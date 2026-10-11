"""Gemini client for the free tier: two keys, strict model preference, rate limiting, and a local cache.

Conversation history is kept as plain dicts (`types.Content.model_dump(mode="json")`) so it can be stored in
the LangGraph checkpoint, hashed for the cache, and sent back to the API unchanged. Sending the model's own
turns back unchanged matters: Gemini 3 attaches a `thought_signature` to function calls and expects it back.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass

from google import genai
from google.genai import types

from .config import get_settings

log = logging.getLogger("ops.llm")


class LLMUnavailable(RuntimeError):
    """Every key and model failed or is out of quota."""


@dataclass
class Reply:
    content: dict          # the model turn as a JSON-safe dict
    model: str             # "gemini-3.5-flash-lite@key1" or "cache"
    seconds: float

    @property
    def text(self) -> str:
        return "".join(p.get("text", "") for p in self.content.get("parts", []) if not p.get("thought")).strip()

    @property
    def function_calls(self) -> list[dict]:
        return [p["function_call"] for p in self.content.get("parts", []) if p.get("function_call")]


class _RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute, self.calls, self.blocked_until = per_minute, {}, {}
        self.lock = threading.Lock()

    def _window(self, pair: str, now: float) -> float:
        q = self.calls.setdefault(pair, deque())
        while q and now - q[0] > 60:
            q.popleft()
        return 0.0 if len(q) < self.per_minute else 60 - (now - q[0])

    def blocked(self, pair: str) -> float:
        with self.lock:
            return max(self.blocked_until.get(pair, 0) - time.monotonic(), 0.0)

    def try_acquire(self, pair: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if self.blocked_until.get(pair, 0) > now or self._window(pair, now) > 0:
                return False
            self.calls[pair].append(now)
            return True

    def ready_in(self, pair: str) -> float:
        now = time.monotonic()
        with self.lock:
            return max(self._window(pair, now), self.blocked_until.get(pair, 0) - now, 0.0)

    def block(self, pair: str, seconds: float) -> None:
        with self.lock:
            self.blocked_until[pair] = time.monotonic() + seconds


class _Cache:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("create table if not exists llm(k text primary key, content text, model text, ts real)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            return self.db.execute("select content, model from llm where k=?", (k,)).fetchone()

    def put(self, k, content, model):
        with self.lock:
            self.db.execute("insert or replace into llm values(?,?,?,?)", (k, content, model, time.time()))
            self.db.commit()


def _retry_after(err: Exception) -> float:
    """Seconds the API asks us to wait: "retryDelay: 41s" (per-minute limit) or "retry in 6h38m13s" (daily
    limit, which resets at midnight Pacific time). Daily limits block that key/model pair for hours."""
    msg = str(err)
    m = re.search(r"retry in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", msg, re.I)
    if m and any(m.groups()):
        h, mi, se = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mi * 60 + se + 1
    m = re.search(r"retryDelay\W+(\d+(?:\.\d+)?)s", msg, re.I)
    if m:
        return float(m.group(1)) + 1
    if re.search(r"PerDay|per day|daily", msg, re.I):
        return 3600.0
    return 30.0


def _thinking(model: str, level: str):
    if model.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level=level)
    if model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=0 if level == "minimal" else 1024)
    return None


def user_turn(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


class Gemini:
    def __init__(self, settings=None):
        self.s = settings or get_settings()
        if not self.s.api_keys:
            raise LLMUnavailable("No API key found. Put GEMINI_API_KEY in .env (see .env.example).")
        self.clients = [genai.Client(api_key=k) for k in self.s.api_keys]
        self.limiter = _RateLimiter(self.s.requests_per_minute)
        self.cache = _Cache(self.s.cache_dir / "llm_cache.sqlite")
        self.calls = 0

    def chat(self, contents: list[dict], *, models: list[str], system: str = "", tools: list[dict] | None = None,
             temperature: float = 0.0, thinking: str = "low", use_cache: bool = True, max_wait_s: float = 300) -> Reply:
        """One model turn. `tools` are function declarations as JSON schema dicts."""
        base = json.dumps([system, contents, tools, temperature, thinking], sort_keys=True, ensure_ascii=False)
        key = lambda m: hashlib.sha256((base + m).encode()).hexdigest()  # noqa: E731
        if use_cache:
            for m in models:
                if hit := self.cache.get(key(m)):
                    return Reply(json.loads(hit[0]), "cache", 0.0)

        decls = [types.Tool(function_declarations=[types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]) for t in tools])] if tools else None
        api_contents = [types.Content.model_validate(c) for c in contents]
        deadline = time.monotonic() + max_wait_s
        last: Exception | None = None
        while time.monotonic() < deadline:
            nap = None
            for model in models:
                pairs = [(f"{model}@key{i + 1}", c) for i, c in enumerate(self.clients)]
                if all(self.limiter.blocked(lbl) > 0 for lbl, _ in pairs):
                    continue                     # every key failing for this model: fall back to the next one
                for label, client in pairs:
                    if not self.limiter.try_acquire(label):
                        continue
                    t0 = time.monotonic()
                    try:
                        r = client.models.generate_content(
                            model=model, contents=api_contents,
                            config=types.GenerateContentConfig(
                                system_instruction=system or None, temperature=temperature, tools=decls,
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                                thinking_config=_thinking(model, thinking),
                                http_options=types.HttpOptions(timeout=60_000)))
                        if not r.candidates or not r.candidates[0].content or not r.candidates[0].content.parts:
                            raise ValueError(f"empty response (finish reason: {r.candidates[0].finish_reason if r.candidates else '?'})")
                        content = r.candidates[0].content.model_dump(mode="json", exclude_none=True)
                        content["role"] = "model"
                        self.calls += 1
                        self.cache.put(key(model), json.dumps(content, ensure_ascii=False), label)
                        return Reply(content, label, time.monotonic() - t0)
                    except Exception as e:  # noqa: BLE001
                        last = e
                        quota = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                        gone = "404" in str(e) or "no longer available" in str(e)    # model retired for this key
                        self.limiter.block(label, _retry_after(e) if quota else 86_400 if gone else 5)
                        log.warning("%s failed: %s", label, str(e)[:1500])
                nap = min(self.limiter.ready_in(lbl) for lbl, _ in pairs)   # busy, not broken: wait for it
                break
            if nap is None:
                nap = min(self.limiter.ready_in(f"{m}@key{i + 1}") for m in models for i in range(len(self.clients)))
            if time.monotonic() + nap > deadline:
                break
            time.sleep(min(max(nap, 0.2), 5))
        raise LLMUnavailable(f"All Gemini keys/models failed or are rate-limited. Last error: {last}")
