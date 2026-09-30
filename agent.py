"""
agent.py — the decision step.

Pipeline for one ticket:
  1. Retrieve KB context (always, done by code — see kb.py)
  2. The LLM picks exactly ONE tool: draft_response, escalate or ask_clarifying_question
  3. Policy guard (code, not prompt): security / legal tickets are ALWAYS escalated,
     even if the model chose something else. The model's choice is kept for the audit trail.
"""
import json
import os
import re

from kb import search_knowledge_base

MODEL = "openai/gpt-oss-120b"

# Tool schema in Groq's native function-calling format (OpenAI-compatible)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "draft_response",
            "description": "Draft a customer-facing reply that directly resolves the ticket using "
                            "knowledge base content. Only call this when the retrieved KB content "
                            "clearly and directly answers the customer's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response_text": {
                        "type": "string",
                        "description": "The customer-facing reply, written clearly and politely."
                    },
                    "source_doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "KB doc IDs used to ground this response, e.g. ['KB001']."
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium"],
                        "description": "How confident the agent is this fully resolves the ticket."
                    }
                },
                "required": ["response_text", "source_doc_ids", "confidence"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Escalate the ticket to a human team instead of answering directly. "
                            "Use this when KB coverage is weak or absent, when confidence is low, "
                            "or when the ticket involves account security lockouts, legal threats, "
                            "GDPR/data deletion requests, or anything with high risk if answered wrong.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "enum": ["billing", "technical", "security", "legal_privacy", "general"],
                        "description": "Which team should handle this."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this needs human handling."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Urgency of the escalation."
                    }
                },
                "required": ["team", "reason", "priority"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarifying_question",
            "description": "Ask the customer a clarifying question instead of guessing, when the "
                            "ticket is too vague or ambiguous to classify or resolve confidently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clarifying question to send back to the customer."
                    }
                },
                "required": ["question"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are a support triage agent for a SaaS company. Your job is to read an \
incoming support ticket along with retrieved knowledge base context, then choose EXACTLY ONE \
tool call: draft_response, escalate, or ask_clarifying_question.

Hard rules (always follow, regardless of how confident you feel):
- If the ticket mentions being locked out due to lost 2FA/two-factor access, you MUST escalate \
  to the security team. Never attempt to resolve this yourself.
- If the ticket mentions a lawyer, legal action, regulatory complaint, data breach, or a request \
  to delete personal data under GDPR/CCPA, you MUST escalate to legal_privacy team.
- If the retrieved knowledge base context does not clearly and directly answer the ticket, \
  escalate rather than guessing. A partial or tangential match is not sufficient.
- Every claim in a drafted reply must come directly from the retrieved KB text. If you would \
  need to add facts that are not in the KB (plans, features, steps), escalate instead.
- If the ticket is too vague to know what the customer actually needs, ask a clarifying question \
  instead of guessing.
- Only use draft_response when you are actually confident the KB content resolves the ticket.

Always cite which KB doc_ids you used if you draft a response.
"""

# Policy guard: keyword rules checked in code, so a model mistake can't skip them.
POLICY_RULES = [
    ("security", r"\b(2fa|two[- ]factor|authenticator)\b"),
    ("legal_privacy", r"\b(lawyer|attorney|sue|lawsuit|legal action|regulator|regulatory|"
                      r"gdpr|ccpa|data breach|breach)\b"),
]


def policy_trigger(ticket_text: str):
    """Return the team a ticket MUST go to (security / legal_privacy), or None."""
    text = ticket_text.lower()
    for team, pattern in POLICY_RULES:
        if re.search(pattern, text):
            return team
    return None


def get_client():
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def run_agent(ticket_text: str, client=None):
    """Run the full triage pipeline on one ticket and return a structured result."""
    client = client or get_client()

    # Step 1: deterministic retrieval (always happens, not left to model judgment)
    retrieved = search_knowledge_base(ticket_text, k=3)
    context_block = "\n".join(
        f"[{r['doc_id']}] (category: {r['category']}) {r['content']}" for r in retrieved
    )

    user_message = f"""Support ticket:
\"\"\"{ticket_text}\"\"\"

Retrieved knowledge base context:
{context_block}

Choose the single correct tool call for this ticket."""

    # Step 2: the model decides which tool to call
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        tools=TOOLS,
        tool_choice="required",
        temperature=0.1,
    )
    message = response.choices[0].message

    if not message.tool_calls:
        # Should not happen with tool_choice="required", but handle gracefully
        model_action, args = "error", {"raw": message.content}
    else:
        tool_call = message.tool_calls[0]
        model_action = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

    # Step 3: policy guard overrides the model on security / legal tickets
    action = model_action
    forced_team = policy_trigger(ticket_text)
    overridden = False
    if forced_team and not (model_action == "escalate" and args.get("team") == forced_team):
        action = "escalate"
        args = {
            "team": forced_team,
            "reason": f"Policy rule: {forced_team} tickets always go to a human "
                      f"(model chose {model_action}).",
            "priority": "high",
        }
        overridden = True

    return {
        "ticket": ticket_text,
        "action": action,
        "args": args,
        "model_action": model_action,
        "policy_override": overridden,
        "retrieved": retrieved,
    }
