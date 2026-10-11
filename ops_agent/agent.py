"""The support agent as a LangGraph graph with a human-in-the-loop interrupt.

    customer message -> receive -> agent --tool calls--> tools --> agent ... --> reply (END)
                                                          (after MAX_ROUNDS tool rounds: final, answer without tools)
                                                          |
                                                interrupt(): supervisor approves / denies, graph resumes here

One graph invocation handles one customer message. The checkpointer keeps the conversation, the session (who is
verified, pending confirmations) and any paused approval, so a conversation can stop for a supervisor and
continue later, even after the app restarts (with the SQLite checkpointer).

Resume detail that matters: on resume LangGraph re-runs the interrupted node from the start. So the tools node
first asks every approval question (pure checks, no side effects) and only then executes the calls. Nothing
runs twice, and the gateway's idempotency keys would catch it if it did.
"""
from __future__ import annotations

import copy
import operator
import sqlite3
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from . import prompts
from .llm import Gemini, user_turn
from .tools import GuardedExecutor, NaiveExecutor, new_session

MAX_ROUNDS = 6       # tool rounds per customer message before the agent must answer


class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    session: dict
    events: Annotated[list, operator.add]       # trace for the UI and the evaluation


def build(con: sqlite3.Connection, llm: Gemini, variant: str = "guarded", checkpointer=None, models: list[str] | None = None):
    ex = GuardedExecutor(con) if variant == "guarded" else NaiveExecutor(con)
    system = prompts.GUARDED if variant == "guarded" else prompts.NAIVE
    models = models or llm.s.agent_models

    def receive(s: State) -> State:
        session = copy.deepcopy(s["session"])
        session["user_turns"] += 1
        session["rounds"] = 0
        last = s["messages"][-1]
        session["customer_text"] += " " + " ".join(p.get("text", "") for p in last.get("parts", []))
        return {"session": session}

    def agent(s: State) -> State:
        r = llm.chat(s["messages"], models=models, system=system, tools=ex.tools)
        session = copy.deepcopy(s["session"])
        session["rounds"] += 1
        ev = {"type": "model", "model": r.model, "seconds": round(r.seconds, 2), "text": r.text,
              "calls": [{"name": c["name"], "args": c.get("args", {})} for c in r.function_calls]}
        return {"messages": [r.content], "session": session, "events": [ev]}

    def tools(s: State) -> State:
        calls = [p["function_call"] for p in s["messages"][-1].get("parts", []) if p.get("function_call")]
        session = copy.deepcopy(s["session"])
        # 1) every approval question first: interrupt() pauses here and returns the supervisor's decision on resume
        approvals = {}
        for i, call in enumerate(calls):
            req = ex.approval_request(call, session)
            if req:
                approvals[i] = interrupt({"type": "supervisor_approval", **req})
        # 2) then execute, in order
        parts, events = [], []
        for i, call in enumerate(calls):
            out = ex.execute(call, session, approvals.get(i))
            resp = {"name": call["name"], "response": out.response}
            if call.get("id"):
                resp["id"] = call["id"]
            parts.append({"function_response": resp})
            events.append({"type": "tool", "name": call["name"], "args": call.get("args", {}), "result": out.response,
                           **({"approval": approvals[i]} if i in approvals else {}), **out.event})
        return {"messages": [{"role": "user", "parts": parts}], "session": session, "events": events}

    def final(s: State) -> State:
        """Round cap reached: answer with what is known, without more tool calls (no unanswered calls are left)."""
        r = llm.chat(s["messages"], models=models, system=system + "\n\nAnswer the customer now. Do not call tools.")
        return {"messages": [r.content], "events": [{"type": "model", "model": r.model, "seconds": round(r.seconds, 2),
                                                     "text": r.text, "calls": [], "note": "round cap"}]}

    def route(s: State) -> str:
        last = s["messages"][-1]
        return "tools" if any(p.get("function_call") for p in last.get("parts", [])) else END

    def after_tools(s: State) -> str:
        return "final" if s["session"]["rounds"] >= MAX_ROUNDS else "agent"

    g = StateGraph(State)
    g.add_node("receive", receive)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("final", final)
    g.set_entry_point("receive")
    g.add_edge("receive", "agent")
    g.add_conditional_edges("agent", route, ["tools", END])
    g.add_conditional_edges("tools", after_tools, ["agent", "final"])
    g.add_edge("final", END)
    return g.compile(checkpointer=checkpointer)


class Conversation:
    """Convenience wrapper: send customer messages, see pending approvals, resume with a decision."""

    def __init__(self, graph, thread_id: str, faults: list[dict] | None = None):
        self.graph, self.config = graph, {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
        self.thread_id, self.faults, self.started = thread_id, faults, False

    def send(self, text: str) -> dict:
        update: dict = {"messages": [user_turn(text)]}
        if not self.started:
            update["session"] = new_session(self.thread_id, self.faults)
            self.started = True
        return self._result(self.graph.invoke(update, self.config))

    def decide(self, approved: bool, note: str = "") -> dict:
        return self._result(self.graph.invoke(Command(resume={"approved": approved, "note": note}), self.config))

    def _result(self, out: dict) -> dict:
        pending = [i.value for i in out.get("__interrupt__", [])]
        reply = ""
        if not pending:
            last = out["messages"][-1]
            reply = "".join(p.get("text", "") for p in last.get("parts", []) if not p.get("thought")).strip()
            if not reply:     # the model stopped mid-tool-loop (round cap): never leave the customer without an answer
                reply = "Sorry, I could not finish that. Let me pass this to a colleague."
        return {"reply": reply, "pending_approval": pending[0] if pending else None, "state": out}

    @property
    def state(self) -> dict:
        return self.graph.get_state(self.config).values
