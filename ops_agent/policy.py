"""The store's customer-service policy, as code.

The same rules are written in plain English in POLICY.md and in the agent's system prompt. The prompt
*explains* the policy to the model; these functions *enforce* it in the guarded agent, no matter what the
model has been persuaded to do. Every function is pure: it reads the database and returns a decision.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

from .config import TODAY

RETURN_WINDOW_DAYS = 14          # change-of-mind returns, counted from delivery
DEFECT_WINDOW_DAYS = 30          # damaged or wrong item, counted from delivery
APPROVAL_THRESHOLD = 10_000      # refunds above this amount (PKR) need a supervisor
CANCELLABLE = ("pending", "processing")
ADDRESS_CHANGEABLE = ("pending", "processing")
DEFECT_REASONS = ("damaged", "wrong_item")
REFUND_REASONS = ("changed_mind", "damaged", "wrong_item")


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    amount: int = 0
    needs_approval: bool = False


def order(con: sqlite3.Connection, order_id: str):
    return con.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()


def items(con: sqlite3.Connection, order_id: str) -> list:
    return con.execute("SELECT oi.*, p.name, p.returnable FROM order_items oi JOIN products p USING (product_id) "
                       "WHERE order_id = ?", (order_id,)).fetchall()


def refunded_products(con: sqlite3.Connection, order_id: str) -> set[str]:
    rows = con.execute("SELECT product_ids FROM refunds WHERE order_id = ? AND status = 'completed'", (order_id,))
    return {p for (ids,) in rows for p in ids.split(",") if p}


def prepaid_amount(con: sqlite3.Connection, order_id: str) -> int:
    """What the customer paid up front (0 for cash on delivery)."""
    if order(con, order_id)["payment_method"] == "cod":
        return 0
    return sum(i["qty"] * i["unit_price"] for i in items(con, order_id))


def owns(con: sqlite3.Connection, customer_id: str | None, order_id: str) -> Decision:
    o = order(con, order_id)
    if o is None:
        return Decision(False, f"order {order_id} does not exist")
    if customer_id is None:
        return Decision(False, "the customer is not verified yet: ask for the email address and an order number, then call verify_customer")
    if o["customer_id"] != customer_id:
        # Same message as a missing order, so the reply does not confirm that the order exists.
        return Decision(False, f"order {order_id} is not on this customer's account")
    return Decision(True)


def can_cancel(con: sqlite3.Connection, order_id: str) -> Decision:
    o = order(con, order_id)
    if o["status"] not in CANCELLABLE:
        hint = " It can be returned after delivery." if o["status"] == "shipped" else ""
        return Decision(False, f"orders that are {o['status']} cannot be cancelled.{hint}")
    return Decision(True, amount=prepaid_amount(con, order_id))


def can_refund(con: sqlite3.Connection, order_id: str, product_ids: list[str], reason: str,
               today: date = TODAY) -> Decision:
    o = order(con, order_id)
    if reason not in REFUND_REASONS:
        return Decision(False, f"reason must be one of {', '.join(REFUND_REASONS)}")
    if o["status"] != "delivered" or not o["delivered_at"]:
        return Decision(False, f"only delivered orders can be returned (this order is {o['status']})")
    window = DEFECT_WINDOW_DAYS if reason in DEFECT_REASONS else RETURN_WINDOW_DAYS
    age = (today - date.fromisoformat(o["delivered_at"])).days
    if age > window:
        return Decision(False, f"delivered {age} days ago; the limit for '{reason}' is {window} days")
    lines = {i["product_id"]: i for i in items(con, order_id)}
    if not product_ids:
        return Decision(False, "say which products are being returned")
    unknown = [p for p in product_ids if p not in lines]
    if unknown:
        return Decision(False, f"not in this order: {', '.join(unknown)}")
    done = refunded_products(con, order_id) & set(product_ids)
    if done:
        return Decision(False, f"already refunded: {', '.join(sorted(done))}")
    final_sale = [lines[p]["name"] for p in product_ids if not lines[p]["returnable"]]
    if final_sale:
        return Decision(False, f"not returnable (final sale): {', '.join(final_sale)}")
    amount = sum(lines[p]["qty"] * lines[p]["unit_price"] for p in product_ids)
    return Decision(True, amount=amount, needs_approval=amount > APPROVAL_THRESHOLD)


def can_change_address(con: sqlite3.Connection, order_id: str) -> Decision:
    o = order(con, order_id)
    if o["status"] not in ADDRESS_CHANGEABLE:
        return Decision(False, f"the address cannot be changed once an order is {o['status']}")
    return Decision(True)
