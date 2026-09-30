# 🎫 AI Support Triage Agent

**An agentic tool-calling system that routes support tickets — not just answers them.**

🔗 **[Live demo →](https://support-triage-agent-3rdpekkbtyma2bhjafya9z.streamlit.app/)**  ·  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mistyvisty/support-triage-agent/blob/main/Support_Triage_Agent.ipynb)

> Built by [Preeti Bhardwaj](https://mistyvisty.github.io/) · Stack: Python · Groq SDK (native tool-calling) · FAISS · MiniLM · Streamlit

---

## 🎯 What this project is really about

Most "AI support bot" demos do one thing: take a question, call an LLM, return an answer. That breaks in production — the model has no way to say "I don't know, route this to a human," no way to ask for missing information, and no structured trail an ops team can audit.

This project builds something different: a **tool-calling agent** that explicitly chooses between three possible actions for every ticket, instead of always generating a reply regardless of confidence.

The agent can:
- ✅ **Draft a response** — when the knowledge base clearly covers the question
- 🚨 **Escalate to a human team** — when KB coverage is weak, confidence is low, or the ticket matches a policy trigger (security lockouts, legal/GDPR requests)
- ❓ **Ask a clarifying question** — when the ticket is too vague to classify or resolve

---

## 🕹️ Try it

**[Open the live app](https://support-triage-agent-3rdpekkbtyma2bhjafya9z.streamlit.app/)** and pick a sample ticket, or write your own:

| Sample | Expected behavior |
|---|---|
| Simple question | Draft a reply, grounded in a KB article |
| Security lockout | Escalate to `security` |
| Legal / GDPR | Escalate to `legal_privacy` |
| Vague | Ask a clarifying question |
| Not in the KB | Escalate instead of inventing an answer |

The app also shows the help articles the agent retrieved, and flags when the code policy guard overrode the model.

> The live app runs on a free tier, so it may take a minute to wake up, and it can hit a daily request limit.

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
 │  (Groq LLM)         │  tool_choice="required"). Model must call exactly
 │                     │  ONE of three tools, based on:
 │                     │  - Retrieved context quality
 │                     │  - Rules in the system prompt
 │                     │  - Ticket clarity
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │  3. Policy Guard    │  Plain code (regex), not a prompt. Security and
 │  (code)             │  legal tickets are ALWAYS escalated, even if the
 └─────────────────────┘  model chose otherwise. The model's original
        │                 choice is kept for the audit trail.
        ▼
 Structured Output: { action, args, model_action, policy_override, retrieved }
```

### Why a fixed pipeline?

Retrieval is deterministic; the decision is agentic. Letting the model decide *whether* to search adds risk for no real benefit, while letting it decide *what to do with* the results is where the real reasoning value is.

### Policy rules for high-stakes tickets

Two categories must **always** be escalated. This is enforced twice: the system prompt tells the model to do it, and a code-level guard checks the ticket text afterwards so a model mistake can't skip it.

| Trigger | Team | Why it's a must-escalate |
|---|---|---|
| Lost 2FA / account security lockout | `security` | Wrong guess = locked-out customer, potential account breach |
| Legal threats / GDPR/CCPA data deletion / data breach | `legal_privacy` | Wrong guess = regulatory liability |

This follows the same principle as my [PCOS × Neurodivergence RAG](https://github.com/mistyvisty/pcos-neurodivergence-rag) project: refuse instead of guess when the cost of a wrong answer is too high.

The system prompt also requires that every claim in a drafted reply come directly from the retrieved KB text. If the reply would need facts that aren't in the KB, the agent escalates instead.

---

## 📊 Evaluation Results

Ran against a hand-labeled set of 21 support tickets across all three action types, using the original notebook (Groq LLaMA 3.3-70B, before the code policy guard and the grounding rule).

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

### What changed since these numbers

- Added the **code policy guard** (security/legal escalation no longer depends on model judgment).
- Added a **grounding rule** after the live app drafted a confident reply to an out-of-KB ticket using details that weren't in the knowledge base.
- The live app now runs `openai/gpt-oss-120b` on Groq, because `llama-3.3-70b-versatile` was no longer available on my account.

These numbers have **not** been re-measured on the new setup. Re-running `python eval.py` is next, and this table will be updated with the results, whether or not they look better.

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM + tool-calling | Groq (LLaMA 3.3-70B in the notebook, `gpt-oss-120b` in the live app) | Fast inference, native OpenAI-compatible function-calling |
| Embeddings | HuggingFace MiniLM-L6-v2 | Lightweight, fast, no API cost |
| Vector search | FAISS | In-memory, low-latency, right-sized for this KB |
| Web app | Streamlit | Quick to build and free to host |
| Orchestration | Python + Groq SDK | Intentionally minimal — no agent framework, so the tool-calling mechanics are fully visible |
| Notebook | Google Colab | Reproducible, no local setup required |

---

## 🚀 How to run

### Live app
Open the [live demo](https://support-triage-agent-3rdpekkbtyma2bhjafya9z.streamlit.app/).

### Locally
```bash
git clone https://github.com/mistyvisty/support-triage-agent.git
cd support-triage-agent
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"   # free at console.groq.com
streamlit run app.py
```

To run the evaluation harness: `python eval.py`

### In Colab
1. Click **Open in Colab** above
2. Run Section 1 — it will prompt you for a Groq API key (free at [console.groq.com](https://console.groq.com))
3. Run all sections top to bottom
4. Section 4 shows live agent decisions on sample tickets
5. Section 6 runs the full eval harness and prints accuracy results

---

## 📁 Repo Structure

```
support-triage-agent/
├── app.py                       # Streamlit web app
├── agent.py                     # Tool schemas, system prompt, policy guard, run_agent()
├── kb.py                        # Knowledge base + FAISS retrieval
├── eval.py                      # Evaluation harness (21 labeled tickets)
├── Support_Triage_Agent.ipynb   # Original notebook — KB, agent loop, eval
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- **Policy guard uses keyword rules.** It is enforced in code, but an unusually phrased security or legal ticket that avoids the keywords could still slip past it.
- **Small eval set** (21 tickets), so per-class accuracy can swing a lot from one or two tickets
- **Synthetic, hand-labeled data** — tickets and labels were written by one person; a production system would validate on real historical tickets with multiple labelers
- **Small hand-built knowledge base** (14 docs), not real support documentation
- **One tool call per ticket** and no conversation memory — `ask_clarifying_question` ends the interaction instead of continuing it
- **Metrics predate the current setup** (see "What changed since these numbers")

## 🔧 What I'd build next

- Re-run the eval on the current model and prompt, and update the results above
- Replace keyword rules with a small classifier, or combine both, for the policy check
- Multi-turn handling, so a clarifying question continues the loop when the customer replies
- Confidence calibration — compare the model's self-reported `confidence` against actual correctness

---

## 🔍 What this demonstrates

- **Real tool-calling** — the model explicitly selects from a defined function set, not free text that gets parsed afterward
- **Grounding by design** — retrieval is always done by code before the model decides anything
- **Safety in code, not just prompts** — high-stakes escalation is checked by a rule the model can't skip
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
