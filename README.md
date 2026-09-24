# 🎫 AI Support Triage Agent

**An agentic tool-calling system that routes support tickets — not just answers them.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mistyvisty/support-triage-agent/blob/main/Support_Triage_Agent.ipynb)

> Built by [Preeti Bhardwaj](https://mistyvisty.github.io/) · Stack: Python · Groq SDK (native tool-calling) · FAISS · MiniLM · Groq LLaMA 3.3-70B

---

## 🎯 What this project is really about

Most "AI support bot" demos do one thing: take a question, call an LLM, return an answer. That breaks in production — the model has no way to say "I don't know, route this to a human," no way to ask for missing information, and no structured trail an ops team can audit.

This project builds something different: a **tool-calling agent** that explicitly chooses between three possible actions for every ticket, instead of always generating a reply regardless of confidence.

The agent can:
- ✅ **Draft a response** — when the knowledge base clearly covers the question
- 🚨 **Escalate to a human team** — when KB coverage is weak, confidence is low, or the ticket matches a policy trigger (security lockouts, legal/GDPR requests)
- ❓ **Ask a clarifying question** — when the ticket is too vague to classify or resolve

---

## 🏗️ Architecture

```
Incoming Ticket (text)
        │
        ▼
 ┌─────────────────────┐
 │  1. KB Retrieval    │  Always runs first, called directly by code —
 │  (FAISS + MiniLM)   │  never left to the model. Guarantees grounding
 └─────────────────────┘  before any decision is made.
        │
        ▼
 ┌─────────────────────┐
 │  2. Agent Decision  │  Native tool-calling (Groq function-calling API,
 │  (LLaMA 3.3-70B)    │  tool_choice="required"). Model must call exactly
 │                     │  ONE of three tools, based on:
 │                     │  - Retrieved context quality
 │                     │  - Policy rules in the system prompt
 │                     │  - Ticket clarity
 └─────────────────────┘
        │
        ▼
 Structured Output: { action, args, retrieved_docs, retrieval_top_distance }
```

### Why a fixed two-step pipeline?

Retrieval is deterministic; the decision is agentic. Letting the model decide *whether* to search adds risk for no real benefit, while letting it decide *what to do with* the results is where the real reasoning value is.

### Policy rules for high-stakes tickets

Two categories must **always** be escalated, and the system prompt instructs the model to do so regardless of how confident it feels:

| Trigger | Team | Why it's a must-escalate |
|---|---|---|
| Lost 2FA / account security lockout | `security` | Wrong guess = locked-out customer, potential account breach |
| Legal threats / GDPR/CCPA data deletion / data breach | `legal_privacy` | Wrong guess = regulatory liability |

This follows the same principle as my [PCOS × Neurodivergence RAG](https://github.com/mistyvisty/pcos-neurodivergence-rag) project: refuse instead of guess when the cost of a wrong answer is too high.

---

## 📊 Evaluation Results

Ran against a hand-labeled set of 21 support tickets across all three action types.

| Metric | Score |
|---|---|
| **Must-escalate recall** (security/legal tickets) | **100%** (6/6) |
| Overall routing accuracy | 57.1% (12/21) |

**Breakdown by action type:**

| Expected Action | Accuracy | n |
|---|---|---|
| `ask_clarifying_question` | 100% | 4 |
| `escalate` | 77.8% | 9 |
| `draft_response` | 12.5% | 8 |

### Honest diagnosis of the 57.1%

7 of the 9 misclassifications are the same error: the model treated clear, KB-covered questions as ambiguous and asked a clarifying question instead of drafting a response. This is a known LLM behavior — cautious rules that are correct for security/legal cases generalizing too broadly to straightforward factual queries.

The other 2 were out-of-KB tickets (not security/legal) that should have been escalated but weren't.

**The high-stakes metric is what matters most here.** All 6 security and legal tickets were escalated. A wrong `draft_response` on "what's the API rate limit?" is annoying; a wrong `draft_response` on a legal threat or security lockout is a serious production failure. The system's errors fall on the low-cost side: it never under-escalated a high-stakes ticket.

### Next step: fixing over-clarification

The planned fix is one explicit instruction in the system prompt: *"If retrieved KB context directly and clearly covers the question, call `draft_response`; only use `ask_clarifying_question` when the ticket is genuinely ambiguous."* The eval harness will measure whether it improves `draft_response` accuracy without lowering must-escalate recall.

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM + tool-calling | Groq LLaMA 3.3-70B | Fast inference, native OpenAI-compatible function-calling |
| Embeddings | HuggingFace MiniLM-L6-v2 | Lightweight, fast, no API cost |
| Vector search | FAISS | In-memory, low-latency, right-sized for this KB |
| Orchestration | Python + Groq SDK | Intentionally minimal — no agent framework, so the tool-calling mechanics are fully visible |
| Notebook | Google Colab | Reproducible, no local setup required |

---

## 🚀 How to run

1. Click **Open in Colab** above
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

## ⚠️ Limitations

- **Policy rules are prompt-enforced**, not code-enforced — an unusually phrased security or legal ticket could slip past
- **Small eval set** (21 tickets), so per-class accuracy can swing a lot from one or two tickets
- **Synthetic, hand-labeled data** — tickets and labels were written by one person; a production system would validate on real historical tickets with multiple labelers
- **Small hand-built knowledge base** (14 docs), not real support documentation
- **One tool call per ticket** and no conversation memory — `ask_clarifying_question` ends the interaction instead of continuing it

## 🔧 What I'd build next

- A code-level policy check that runs before the LLM, so security/legal escalation never depends on model judgment
- Multi-turn handling, so a clarifying question continues the loop when the customer replies
- Confidence calibration — compare the model's self-reported `confidence` against actual correctness

---

## 🔍 What this demonstrates

- **Real tool-calling** — the model explicitly selects from a defined function set, not free text that gets parsed afterward
- **Grounding by design** — retrieval is always done by code before the model decides anything
- **A genuine eval harness** — labeled tickets, real metrics, and two separate scoring criteria chosen because they're not equally important
- **Honest iteration** — the 57.1% overall accuracy is documented and diagnosed, not hidden. Understanding *why* a system fails is the actual engineering skill.

---

## 🔗 Related projects

- [PCOS × Neurodivergence RAG](https://github.com/mistyvisty/pcos-neurodivergence-rag) — RAG pipeline on clinical research papers with hallucination-aware refusal prompting
- [Medical Research Assistant](https://github.com/mistyvisty/medical-research-agent) — 3-agent LangGraph pipeline with a dedicated Fact-Checker agent
- [Hospital Readmission Risk Predictor](https://github.com/mistyvisty/hospital-readmission-predictor) — production-style ML pipeline with FastAPI + Docker + CI/CD

---

## 👩‍💻 Author

**Preeti Bhardwaj** — Software Developer | GenAI & Agentic Systems | RAG & LLM Engineering

[Portfolio](https://mistyvisty.github.io/) · [GitHub](https://github.com/mistyvisty) · [Medium](https://medium.com/@bhardwajpreeti357)
