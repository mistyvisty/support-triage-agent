# 🎫 AI Support Triage Agent

**An agentic tool-calling system that routes support tickets — not just answers them.**

🔗 **[Live demo →](https://support-triage-agent-3rdpekkbtyma2bhjafya9z.streamlit.app/)**  ·  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mistyvisty/support-triage-agent/blob/main/Support_Triage_Agent.ipynb) . 
## Write-ups
- [Part 1 — I Built an AI Agent That Knows When Not to Answer](https://medium.com/@bhardwajpreeti357/i-built-an-ai-agent-that-knows-when-not-to-answer-%EF%B8%8F-804776968ff9)
- [Part 2 — How I Improved My AI Agent's Ticket Routing Accuracy from 57% to 86%](https://medium.com/@bhardwajpreeti357/how-i-improved-my-ai-agents-ticket-routing-accuracy-from-57-to-86-part-2-27d87f5b9d16)


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

Re-run on an expanded set of **64 labeled tickets** (up from the original 21), on `openai/gpt-oss-120b` via Groq, with the code policy guard and grounding rule in place. The expansion specifically adds **adversarial tickets**: security/legal escalations phrased *without* the guard's exact trigger words, and tickets that mention a trigger word (lawyer, 2FA, data breach) but aren't actually hard-trigger situations — a stress test for whether the guard generalizes past its literal keyword list.

| Metric | Score |
|---|---|
| **Must-escalate recall** (security/legal tickets) | **100%** (0 missed) |
| Overall routing accuracy | 85.9% (55/64) |
| False escalations (alert-fatigue risk) | 3/64 |
| Mean cost-weighted error per ticket | 0.41 |

**Breakdown by action type:**

| Expected Action | Accuracy | n |
|---|---|---|
| `ask_clarifying_question` | 100% | 10 |
| `draft_response` | 88.0% | 25 |
| `escalate` | 79.3% | 29 |

### Cost-weighted scoring, not just accuracy/recall

Not all errors cost the same. A missed hard-trigger escalation (security/legal) is a safety-critical failure; a false escalation just costs a human's attention (alert fatigue) — cheaper, but not free. `eval.py` now scores each ticket with that asymmetry built in (10 points for a missed hard escalation, 3 for a missed low-stakes escalation, 2 for a false escalation, 1 for other mix-ups), rather than treating every wrong answer the same.

### Honest diagnosis of the 9 misclassifications

- **3 false escalations**: tickets that merely *mention* a trigger-adjacent word (a lawyer joke, 2FA sync question, a "totally different company" data breach) got escalated when they should have been drafted. This is the guard erring toward caution — safer than the alternative, but it is the real alert-fatigue cost, not a hidden one.
- **6 missed low-stakes escalations**: billing disputes and out-of-KB requests (SSO, custom integrations) that the model answered or asked about instead of escalating. None of these were security or legal tickets.

**Zero of the 9 errors touched a hard-trigger ticket.** Every security and legal case in the 64-ticket set — including the adversarially-phrased ones — was escalated correctly. The system's errors stay on the low-cost side of the cost matrix.

### What changed since the original 57.1%/21-ticket numbers

- Added the **code policy guard** (security/legal escalation no longer depends on model judgment).
- Added a **grounding rule** after the live app drafted a confident reply to an out-of-KB ticket using details that weren't in the knowledge base.
- Switched to `openai/gpt-oss-120b` on Groq, because `llama-3.3-70b-versatile` was no longer available on my account.
- Expanded the eval set from 21 to 64 tickets, with adversarial phrasing added specifically to stress-test the guard beyond its literal keywords.
- Added cost-weighted scoring.

This set is still synthetic and hand-labeled by one person — a few hundred real adversarial tickets, reviewed against actual historical traffic, is the next honest step up from here.

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM + tool-calling | Groq (`openai/gpt-oss-120b`) | Fast inference, native OpenAI-compatible function-calling |
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
5. Section 6 runs the full eval harness (64 tickets) and prints accuracy, recall, and cost-weighted results

---

## 📁 Repo Structure

```
support-triage-agent/
├── app.py                       # Streamlit web app
├── agent.py                     # Tool schemas, system prompt, policy guard, run_agent()
├── kb.py                        # Knowledge base + FAISS retrieval
├── eval.py                      # Evaluation harness (64 labeled tickets, cost-weighted scoring)
├── Support_Triage_Agent.ipynb   # Original notebook — KB, agent loop, eval
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- **Policy guard uses keyword rules.** It is enforced in code, and the 64-ticket eval now specifically tests adversarial phrasing around it — but an even more unusual phrasing could still slip past it.
- **Synthetic, hand-labeled data** — tickets and labels were written by one person; a production system would validate on real historical tickets with multiple labelers
- **Small hand-built knowledge base** (14 docs), not real support documentation
- **One tool call per ticket** and no conversation memory — `ask_clarifying_question` ends the interaction instead of continuing it

## 🔧 What I'd build next

- Validate against a few hundred real (or more rigorously adversarial) tickets, reviewed by more than one labeler
- Replace keyword rules with a small classifier, or combine both, for the policy check
- Multi-turn handling, so a clarifying question continues the loop when the customer replies
- Confidence calibration — compare the model's self-reported `confidence` against actual correctness

---

## 🔍 What this demonstrates

- **Real tool-calling** — the model explicitly selects from a defined function set, not free text that gets parsed afterward
- **Grounding by design** — retrieval is always done by code before the model decides anything
- **Safety in code, not just prompts** — high-stakes escalation is checked by a rule the model can't skip
- **A genuine eval harness** — labeled tickets, real metrics, cost-weighted scoring, and adversarial test cases chosen because not every failure mode costs the same
- **Honest iteration** — the original 57.1% overall accuracy was documented and diagnosed, not hidden, and the fix was tested against a harder, adversarial set rather than just re-measured on the same easy one. Understanding *why* a system fails — and proving a fix actually holds under pressure — is the real engineering skill.

---

## 🔗 Related projects

- [PCOS × Neurodivergence RAG](https://github.com/mistyvisty/pcos-neurodivergence-rag) — RAG pipeline on clinical research papers with hallucination-aware refusal prompting
- [Medical Research Assistant](https://github.com/mistyvisty/medical-research-agent) — 3-agent LangGraph pipeline with a dedicated Fact-Checker agent
- [Hospital Readmission Risk Predictor](https://github.com/mistyvisty/hospital-readmission-predictor) — production-style ML pipeline with FastAPI + Docker + CI/CD

---

## 👩‍💻 Author

**Preeti Bhardwaj** — Software Developer | GenAI & Agentic Systems | RAG & LLM Engineering

[Portfolio](https://mistyvisty.github.io/) · [GitHub](https://github.com/mistyvisty) · [Medium](https://medium.com/@bhardwajpreeti357)
