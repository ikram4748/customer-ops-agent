# Customer-Ops Agent with Guardrails

A customer-support agent for a fictional online store that can **take real actions**: cancel orders, refund returns,
change delivery addresses. Every tool call goes through a **gateway** that the model cannot talk its way past:

| Layer | What it does |
|---|---|
| **Identity** | Order tools only work on the verified customer's own orders. |
| **Argument provenance** | The email and order number used to verify must appear in the customer's own messages. (During development the model once "verified" a made-up email.) |
| **Policy in code** | Return windows, order states, refund amounts and final-sale items are checked in `policy.py`, not left to the prompt. |
| **Confirmation** | A change returns a summary and a confirmation code; the code is only accepted after a *new* customer message, so the model cannot confirm on the customer's behalf. |
| **Human approval** | Refunds above PKR 10,000 pause the conversation with a LangGraph `interrupt()` until a supervisor approves or denies. With the SQLite checkpointer, a paused conversation survives an app restart. |
| **Idempotency** | Every change is recorded under a key derived from the request, in the same transaction as the change, so a retried refund after a timeout can never pay twice. |
| **Untrusted text** | Delivery notes and reviews are passed as labelled data, never as instructions. |

```
customer message ─► receive ─► agent (Gemini + tools) ──tool calls──► gateway + tools ──► agent ... ──► reply
                                                                         │
                                                       interrupt(): supervisor approves / denies
```

> **Results: in progress.** The agent, gateway, 45 scenarios, evaluation harness, tests and demo are complete. The
> full evaluation (90 simulated conversations) is being run; the table below will be filled from
> `results/RESULTS_test.md`.

## Evaluation

Modelled on [tau-bench](https://github.com/sierra-research/tau-bench): an LLM plays the customer (persona + goal, a
different model from the agent), talks to the agent, and **success is checked on the final state of the store
database**, not on wording. 45 scenarios (15 dev used while building, 30 held-out test) in seven groups: normal tasks,
policy refusals, approvals, identity and privacy, prompt injection (in delivery notes, reviews and chat), social
engineering, and injected tool failures (including "the refund succeeded but the response timed out").

Two variants, same model and same tools:
- **naive:** the policy is only in the prompt; the tools do what they are asked (how tau-bench's tools work, and how
  most agent demos are built)
- **guarded:** the gateway above

Two numbers per scenario, both computed from data, not by an LLM judge:
- **Task success:** every expected outcome holds and there is no violation.
- **Violation:** something that must never happen happened: a refund or change against policy, a double refund, a
  change to someone else's order, a high-value refund without approval, or another customer's personal data in a
  reply. These checks run on every scenario automatically, by comparing the final database with the initial one.

## Results

*To be filled from `results/RESULTS_test.md`.*

## Run it

**Windows:** double-click `run.bat`. **macOS:** double-click `run.command`. The first run installs packages, asks for a
free Gemini API key and opens the demo: chat as one of the demo customers (sidebar), watch the gateway trace, approve
or deny refunds in the supervisor queue, and replay red-team attacks with one click. Switch between `guarded` and
`naive` to compare.

```bash
pip install -r requirements.txt
cp .env.example .env                       # add GEMINI_API_KEY
streamlit run app.py
python -m evaluation.run_eval --split test # simulated conversations for both variants
python -m pytest                           # 14 offline tests (scripted fake LLM)
```

## Project layout

```
ops_agent/
  agent.py      LangGraph graph: receive → agent ⇄ tools (interrupt for approval) → reply; Conversation wrapper
  tools.py      tool schemas, raw store operations, fault injection, NaiveExecutor, GuardedExecutor (the gateway)
  policy.py     the policy as pure functions
  store.py      synthetic store database (6 demo customers, 150 generated ones, injection fixtures)
  simulator.py  the simulated customer
  prompts.py    system prompts (shared policy text) and the simulator prompt
  llm.py        Gemini client: two keys, strict model preference, rate limiting, local cache
scenarios/scenarios.yaml   45 scenarios with machine-checkable expectations
evaluation/run_eval.py     conversations, automatic violation detection, report
POLICY.md                  the store policy and the values enforced in code
```

All data is synthetic; the store, customers and orders are fictional.

## Stack

Python · LangGraph (interrupts, SQLite checkpointer) · Gemini function calling (`google-genai`) · SQLite · Streamlit · pytest

## License

MIT
