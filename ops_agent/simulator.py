"""A simulated customer: an LLM playing a persona with a goal, as in tau-bench.

It runs on a different model from the agent (separate free-tier quota, and a customer played by the same model
as the agent tends to agree with it too easily). In the simulator's own history the roles are flipped: the
agent's replies are its "user" turns and its own messages are "model" turns.
"""
from __future__ import annotations

from . import prompts
from .llm import Gemini, user_turn

STOP = "###STOP###"


class SimulatedCustomer:
    def __init__(self, llm: Gemini, persona: str, models: list[str] | None = None):
        self.llm, self.models = llm, models or llm.s.user_models
        self.system = prompts.USER_SIMULATOR.format(persona=persona.strip())
        self.history: list[dict] = [user_turn("(The support chat window opens. Write your first message.)")]
        self.done = False

    def say(self, agent_reply: str | None = None) -> str:
        if agent_reply is not None:
            self.history.append(user_turn(agent_reply or "(no reply)"))
        r = self.llm.chat(self.history, models=self.models, system=self.system, temperature=0.0, thinking="minimal")
        text = r.text
        self.history.append({"role": "model", "parts": [{"text": text}]})
        if STOP in text:
            self.done = True
            text = text.replace(STOP, "").strip()
        return text
