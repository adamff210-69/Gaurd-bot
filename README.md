# 🛡️ GuardBot — Prompt-Injection-Guarded Chatbot

A single conversational chatbot (LangGraph + hosted LLM API + Streamlit) where
**every user turn passes through two lightweight prompt-injection detectors**
before it may reach the main LLM. Scope: **direct** injection only (attacks
arriving via the user's own messages).

```
user turn
   │
   ▼
detector1_node  ── DistilBERT fine-tuned on deepset/prompt-injections
   │                 → {label, score}
   ▼
detector2_node  ── tiny instruction-tuned LLM-as-judge (Qwen2.5-0.5B GGUF,
   │                 JSON-schema-constrained output) → {label, reason}
   ▼
decision (conditional edge, policy = OR / AND)
   ├─ flagged → blocked_response_node  → safe fallback + logs/blocked_events.jsonl
   └─ clean   → llm_node (Groq / OpenRouter chat model + chat_history) → END
```

## Layout

| Path | What |
|---|---|
| `guardbot/config.py` | providers, model ids, policy, persona (with eval canary) |
| `guardbot/detectors.py` | Detector 1 (classifier) + Detector 2 (LLM judge, local/api/heuristic fallback chain) |
| `guardbot/graph.py` | LangGraph state schema, nodes, conditional edge, multi-turn helper |
| `guardbot/llm.py` | main chat LLM factory (Groq / OpenRouter / mock) |
| `app.py` | Streamlit chat UI: policy toggle, live scores, blocked banner, trace panel, attack buttons |
| `eval/run_eval.py` | metrics (P/R/F1 per detector, OR vs AND ensemble, latency) + behaviour-change probe |
| `eval/custom_cases.json` | 15 benign + 15 direct-injection hand-written cases |
| `writeup.md` | results: what got through, why, and fixes |

## Run

```bash
pip install -r requirements.txt          # llama-cpp-python compiles from source
cp .env.example .env                     # add GROQ_API_KEY or OPENROUTER_API_KEY
streamlit run app.py
```

No API key? The app still runs: both detectors are fully local; the main LLM
node answers in a clearly-labelled mock mode.

## Evaluation

```bash
python -m eval.run_eval              # full: deepset test split + custom cases
python -m eval.run_eval --limit 40   # quick pass
python -m eval.run_eval --skip-probe # metrics only, no live-LLM probe
```

Outputs `eval/results.json` and a console summary. The behaviour-change probe
re-sends every injection **missed** by the OR-ensemble through the full
pipeline and checks for system-prompt/canary leaks (`ZX-CANARY-4471`), obeyed
overrides ("DAN mode on"), and persona breaks — it needs a real API key.

## Design notes

- **Policy toggle** — `OR` blocks if either detector fires (recall-favoring,
  default); `AND` needs both (precision-favoring). Switchable live in the
  sidebar; also `POLICY=` in `.env`.
- **Multi-turn** — `chat_history` persists in state across turns. Blocked
  turns are *not* appended to the LLM-visible history, so a caught attack can
  never poison later context.
- **Detector 2 fallback chain** — local GGUF judge → API judge (if a key
  exists) → regex heuristic, so a judge failure degrades gracefully instead of
  crashing a turn.
- **Structured judge output** — the local judge uses llama.cpp's JSON-schema
  constrained decoding; the API judge uses LangChain `with_structured_output`.
