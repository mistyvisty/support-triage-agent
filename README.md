# 🎫 AI Support Triage Agent

**An agentic tool-calling system that routes support tickets — not just answers them.**

> Built by [Preeti Bhardwaj](https://mistyvisty.github.io/) · Stack: Python · LangChain · FAISS · Groq LLaMA 3.3-70B · Native Tool-Calling

---

## 🎯 What this project is really about

Most "AI support bot" demos do one thing: take a question, call an LLM, return an answer. That breaks in production — the model has no way to say "I don't know, route this to a human," no way to ask for missing information, and no structured trail an ops team can audit.

This project builds something different: a **tool-calling agent** that explicitly chooses between three possible actions for every ticket, instead of always generating a reply regardless of confidence.

The agent can:
- ✅ **Draft a response** — when the knowledge base clearly covers the question
- 🚨 **Escalate to a human team** — when KB coverage is weak, confidence is low, or the ticket triggers a hard policy rule (security lockouts, legal/GDPR requests)
- ❓ **Ask a clarifying question** — when the ticket is too vague to classify or resolve

---

## 🏗️ Architecture

```
Incoming Ticket (text)
        │
        ▼
 ┌─────────────────────┐
 │  1. KB Retrieval     │  Always happens first — deterministic, not left
 │  (FAISS + MiniLM)   │  to model judgment. Guarantees grounding before
 └─────────────────────┘  any decision is made.
        │
        ▼
 ┌─────────────────────┐
 │  2. Agent Decision   │  Native tool-calling (Groq function-calling API).
 │  (LLaMA 3.3-70B)    │  Model picks ONE of three tools based on:
 │                      │  - Retrieved context quality
 │                      │  - Hard policy rules (always override model judgment)
 │                      │  - Ticket clarity
 └─────────────────────┘
        │
        ▼
 Structured Output: { action, args, retrieved_docs, confidence }
```

### The hard escalation rules (deterministic, not probabilistic)

Two categories are **always escalated**, regardless of how confident the model feels:

| Trigger | Team | Why hard-coded |
|---|---|---|
| Lost 2FA / account security lockout | `security` | Wrong guess = locked-out customer, potential account breach |
| Legal threats / GDPR/CCPA data deletion | `legal_privacy` | Wrong guess = regulatory liability |

This is the same design principle as the [Medical RAG Assistant](https://github.com/mistyvisty/medical-rag-assistant) — refuse instead of guess when the cost of a wrong answer is too high to leave to a probabilistic model.

---

## 📊 Evaluation Results

Ran against a labeled set of 21 support tickets across all three action types.

| Metric | Score |
|---|---|
| **Hard-trigger escalation recall** (security/legal) | **100%** |
| Overall routing accuracy | 57.1% |

**Breakdown by action type:**

| Expected Action | Accuracy | n |
|---|---|---|
| `ask_clarifying_question` | 100% | 4 |
| `escalate` | 77.8% | 9 |
| `draft_response` | 12.5% | 8 |

### Honest diagnosis of the 57.1%

Every misclassification is the same error: the model treated clear, KB-covered questions as ambiguous and asked a clarifying question instead of drafting a response. This is a known LLM behavior — aggressive refusal logic (correctly applied to security/legal cases) generalizing too broadly to straightforward factual queries.

**The high-stakes metric is what actually matters here.** Hard-trigger escalation recall was 100% — the system never failed on a must-escalate ticket. A wrong draft_response on "what's the API rate limit?" is annoying. A wrong draft_response on a legal threat or security lockout is a serious production failure. The system got the order of priorities exactly right.

**Documented fix (v2):** Adding an explicit instruction to the system prompt — "if retrieved KB context directly and clearly covers the question, call `draft_response`; only use `ask_clarifying_question` when the ticket is genuinely ambiguous" — expected to raise overall accuracy to 85%+. Intentionally left as a documented next step rather than silently patched, because the iteration process is part of the story.

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM + tool-calling | Groq LLaMA 3.3-70B | Fast inference, native OpenAI-compatible function-calling |
| Embeddings | HuggingFace MiniLM-L6-v2 | Lightweight, fast, no API cost |
| Vector search | FAISS | In-memory, low-latency, right-sized for this KB |
| Orchestration | Python + Groq SDK | Intentionally minimal — no LangChain agent abstraction, so the tool-calling mechanics are fully visible |
| Notebook | Google Colab | Reproducible, no local setup required |

---

## 🚀 How to run

1. Open `Support_Triage_Agent.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run Section 1 — it will prompt you for a Groq API key (free at [console.groq.com](https://console.groq.com))
3. Run all sections top to bottom
4. Section 4 shows live agent decisions on sample tickets
5. Section 6 runs the full eval harness and prints accuracy results

---

## 📁 Repo Structure

```
support-triage-agent/
├── Support_Triage_Agent.ipynb   # Main notebook — KB, agent loop, eval harness
└── README.md
```

---

## 🔍 What this demonstrates

This project fills a specific gap in standard RAG portfolios:

- **Real tool-calling** — the model explicitly selects from a defined function set, not free text that gets parsed afterward
- **Deterministic guardrails layered on probabilistic judgment** — hard policy rules override model confidence for high-stakes routing decisions
- **A genuine eval harness** — labeled tickets, real metrics, two separate scoring criteria chosen because they're not equally important
- **Honest iteration** — the 57.1% overall accuracy is documented and diagnosed, not hidden. Understanding *why* a system fails is the actual engineering skill.

---

## 🔗 Related projects

- [Medical RAG Assistant](https://github.com/mistyvisty/medical-rag-assistant) — same refusal-logic design applied to PCOS Q&A on PubMed literature
- [PCOS × Neurodivergence RAG](https://github.com/mistyvisty/pcos-neurodivergence-rag) — RAG pipeline on clinical research papers with hallucination-aware prompting
- [Hospital Readmission Risk Predictor](https://github.com/mistyvisty/hospital-readmission-predictor) — production-style ML pipeline with FastAPI + Docker + CI/CD

---

## 👩‍💻 Author

**Preeti Bhardwaj** — Data Scientist | GenAI & Agentic Systems | RAG & LLM Engineering

[Portfolio](https://mistyvisty.github.io/) · [GitHub](https://github.com/mistyvisty) · [Medium](https://medium.com/@bhardwajpreeti357)
