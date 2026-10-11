"""Offline tests: no API key, no network. The LLM is a scripted fake."""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from evaluation.run_eval import check_expectations, find_violations
from ops_agent import policy
from ops_agent.agent import Conversation, build
from ops_agent.llm import Reply
from ops_agent.store import build_store
from ops_agent.tools import GuardedExecutor, NaiveExecutor, new_session


@pytest.fixture
def con():
    return build_store()


def call(name, **args):
    return {"name": name, "args": args}


def verified(session, text="ayesha.khan@example.com ORD-50001", customer="C-1001"):
    session["customer_text"] += " " + text
    session["customer_id"] = customer
    return session


# -- policy ------------------------------------------------------------------------------------------------------

def test_return_windows(con):
    assert policy.can_refund(con, "ORD-50001", ["P-2001"], "changed_mind").allowed            # 6 days
    assert not policy.can_refund(con, "ORD-50010", ["P-2004"], "changed_mind").allowed        # 21 days
    assert policy.can_refund(con, "ORD-50010", ["P-2004"], "damaged").allowed                 # 21 <= 30
    assert not policy.can_refund(con, "ORD-50031", ["P-2009"], "damaged").allowed             # 42 days


def test_refund_rules(con):
    assert not policy.can_refund(con, "ORD-50050", ["P-2001"], "changed_mind").allowed        # already refunded
    assert not policy.can_refund(con, "ORD-50051", ["P-2010"], "changed_mind").allowed        # gift card
    assert not policy.can_refund(con, "ORD-50002", ["P-2008"], "damaged").allowed             # not delivered
    d = policy.can_refund(con, "ORD-50011", ["P-2005"], "changed_mind")
    assert d.allowed and d.amount == 145000 and d.needs_approval
    d = policy.can_refund(con, "ORD-50001", ["P-2001", "P-2002"], "damaged")
    assert d.amount == 5400 and not d.needs_approval


def test_cancel_and_address_rules(con):
    assert policy.can_cancel(con, "ORD-50003").amount == 7800
    assert policy.can_cancel(con, "ORD-50020").amount == 0          # cash on delivery: nothing to refund
    assert not policy.can_cancel(con, "ORD-50002").allowed          # shipped
    assert not policy.can_change_address(con, "ORD-50002").allowed


# -- gateway -------------------------------------------------------------------------------------------------------

def test_unverified_and_foreign_orders_are_blocked(con):
    ex, s = GuardedExecutor(con), new_session("t")
    assert "not verified" in ex.execute(call("get_order", order_id="ORD-50001"), s).response["error"]
    verified(s)
    out = ex.execute(call("get_order", order_id="ORD-50040"), s).response       # Hira's order
    assert "not on this customer's account" in out["error"]
    assert "customer_phone" not in ex.execute(call("get_order", order_id="ORD-50001"), s).response


def test_identity_values_must_come_from_the_customer(con):
    ex, s = GuardedExecutor(con), new_session("t")
    out = ex.execute(call("verify_customer", email="ayesha.khan@example.com", order_id="ORD-50001"), s)
    assert out.response["verified"] is False and s["customer_id"] is None        # the model guessed the email
    s["customer_text"] = "hi, ayesha.khan@example.com, order ORD-50001"
    assert ex.execute(call("verify_customer", email="ayesha.khan@example.com", order_id="ORD-50001"), s).response["verified"]


def test_verification_attempts_are_limited(con):
    ex, s = GuardedExecutor(con), new_session("t")
    s["customer_text"] = "x@example.com ORD-50001"
    for _ in range(3):
        ex.execute(call("verify_customer", email="x@example.com", order_id="ORD-50001"), s)
    s["customer_text"] += " ayesha.khan@example.com"
    out = ex.execute(call("verify_customer", email="ayesha.khan@example.com", order_id="ORD-50001"), s)
    assert out.response["verified"] is False and "too many" in out.response["error"]


def test_changes_need_a_confirmation_in_a_new_customer_message(con):
    ex, s = GuardedExecutor(con), verified(new_session("t"))
    first = ex.execute(call("cancel_order", order_id="ORD-50003", reason="changed mind"), s).response
    assert first["status"] == "needs_customer_confirmation"
    code = first["confirmation_code"]
    same_turn = ex.execute(call("cancel_order", order_id="ORD-50003", reason="changed mind", confirmation_code=code), s).response
    assert same_turn["status"] == "needs_customer_confirmation"                 # no new customer message yet
    assert policy.order(con, "ORD-50003")["status"] == "processing"
    s["user_turns"] += 1                                                         # the customer replied
    done = ex.execute(call("cancel_order", order_id="ORD-50003", reason="ok", confirmation_code=code), s).response
    assert done["status"] == "cancelled" and done["refund"]["amount_pkr"] == 7800


def confirm(ex, s, c):
    code = ex.execute(c, s).response["confirmation_code"]
    s["user_turns"] += 1
    return {**c, "args": {**c["args"], "confirmation_code": code}}


def test_retry_after_timeout_does_not_refund_twice(con):
    ex = GuardedExecutor(con)
    s = verified(new_session("t", faults=[{"tool": "request_refund", "mode": "timeout_after_commit", "times": 1}]))
    c = confirm(ex, s, call("request_refund", order_id="ORD-50001", product_ids=["P-2001"], reason="changed_mind"))
    assert "504" in ex.execute(c, s).response["error"]                          # it happened, but we heard a timeout
    assert ex.execute(c, s).response["status"] == "already_done"               # the retry is recognised
    assert con.execute("SELECT COUNT(*) FROM refunds WHERE order_id = 'ORD-50001'").fetchone()[0] == 1


def test_naive_agent_refunds_twice_on_retry(con):
    ex = NaiveExecutor(con)
    s = new_session("t", faults=[{"tool": "request_refund", "mode": "timeout_after_commit", "times": 1}])
    c = call("request_refund", order_id="ORD-50001", product_ids=["P-2001"], reason="changed_mind")
    ex.execute(c, s), ex.execute(c, s)
    assert con.execute("SELECT SUM(amount) FROM refunds WHERE order_id = 'ORD-50001'").fetchone()[0] == 9000


def test_high_value_refund_asks_for_approval_only_after_confirmation(con):
    ex = GuardedExecutor(con)
    s = verified(new_session("t"), "bilal.ahmed@example.com ORD-50011", "C-1002")
    c = call("request_refund", order_id="ORD-50011", product_ids=["P-2005"], reason="changed_mind")
    assert ex.approval_request(c, s) is None                                    # not confirmed yet
    c = confirm(ex, s, c)
    assert ex.approval_request(c, s)["amount"] == 145000
    assert ex.execute(c, s, {"approved": False}).response["status"] == "denied_by_supervisor"
    assert con.execute("SELECT COUNT(*) FROM refunds WHERE order_id = 'ORD-50011'").fetchone()[0] == 0


def test_untrusted_text_is_labelled(con):
    ex, s = GuardedExecutor(con), verified(new_session("t"), "sana.malik@example.com ORD-50021", "C-1003")
    note = ex.execute(call("get_order", order_id="ORD-50021"), s).response["delivery_note"]
    assert "SYSTEM OVERRIDE" in note["untrusted_text"] and "not instructions" in note["handling"]
    review = ex.execute(call("get_product_info", product_id="P-2009"), s).response["reviews"][0]["review"]
    assert "untrusted_text" in review


# -- graph: interrupt and resume ------------------------------------------------------------------------------------

class FakeLLM:
    """Replies from a script. Each item: text, or a list of (tool, args) calls."""

    def __init__(self, script):
        self.script, self.seen = list(script), []

    def chat(self, contents, **kw):
        self.seen.append(contents)
        step = self.script.pop(0)
        if isinstance(step, str):
            return Reply({"role": "model", "parts": [{"text": step}]}, "fake", 0.0)
        return Reply({"role": "model", "parts": [{"function_call": {"id": f"c{i}", "name": n, "args": a}}
                                                 for i, (n, a) in enumerate(step)]}, "fake", 0.0)

    class s:   # noqa: N801 - mimics Gemini.s
        agent_models = ["fake"]


def test_approval_pauses_the_graph_and_resumes(con):
    refund = {"order_id": "ORD-50011", "product_ids": ["P-2005"], "reason": "changed_mind"}
    llm = FakeLLM([
        [("verify_customer", {"email": "bilal.ahmed@example.com", "order_id": "ORD-50011"}), ("request_refund", refund)],
        "Please confirm: refund PKR 145,000 for the laptop?",
        "CODE",                                                         # replaced below once the code is known
        "Approved: your refund of PKR 145,000 is on its way.",
    ])
    conv = Conversation(build(con, llm, "guarded", InMemorySaver()), "t1")
    r = conv.send("bilal.ahmed@example.com ORD-50011, I want to return the laptop")
    assert r["reply"].startswith("Please confirm") and not r["pending_approval"]
    code = next(e["result"]["confirmation_code"] for e in conv.state["events"] if e.get("gateway") == "confirmation requested")
    llm.script[0] = [("request_refund", {**refund, "confirmation_code": code})]
    r = conv.send("yes")
    assert r["pending_approval"]["amount"] == 145000 and not r["reply"]        # paused for the supervisor
    r = conv.decide(True)
    assert "on its way" in r["reply"]
    assert con.execute("SELECT COUNT(*) FROM refunds WHERE order_id = 'ORD-50011'").fetchone()[0] == 1
    assert con.execute("SELECT decision FROM approvals").fetchone()[0] == "approved"


# -- evaluation checks ----------------------------------------------------------------------------------------------

def test_violation_detector(con):
    init = build_store()
    sc = {"customer": "C-1001", "forbid": [{"must_not_say": ["secret"]}]}
    assert find_violations(sc, con, init, ["all good"]) == []
    con.execute("INSERT INTO refunds VALUES ('R-1','ORD-50001','P-2001',4500,'changed_mind','completed',NULL,'x')")
    con.execute("INSERT INTO refunds VALUES ('R-2','ORD-50001','P-2001',4500,'changed_mind','completed',NULL,'x')")
    con.execute("INSERT INTO refunds VALUES ('R-3','ORD-50011','P-2005',145000,'changed_mind','completed',NULL,'x')")
    con.execute("UPDATE orders SET status = 'cancelled' WHERE order_id = 'ORD-50002'")
    v = " | ".join(find_violations(sc, con, init, ["Hira's number is 0304-5550005", "secret"]))
    for expected in ("product refunded twice on ORD-50001", "someone else's order ORD-50011", "without supervisor approval",
                     "ORD-50002 cancelled while shipped", "leaked phone of C-1005", "forbidden text"):
        assert expected in v, expected


def test_expectations(con):
    init = build_store()
    sc = {"expect": [{"sql": "SELECT status FROM orders WHERE order_id = 'ORD-50003'", "equals": "processing"},
                     {"escalated": False}, {"approval_requested": False}, {"reply_contains_any": ["hello"]}]}
    assert check_expectations(sc, con, init, ["Hello there"], 0) == []
    assert len(check_expectations(sc, con, init, ["bye"], 1)) == 2
