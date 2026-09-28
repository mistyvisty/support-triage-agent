"""
eval.py — the test. Runs the agent on 21 hand-labelled tickets and reports:
  - overall routing accuracy
  - hard-trigger escalation recall (security / legal) — the high-stakes metric
Both are shown for the model alone AND with the code policy guard, so we can see
what the guard adds.

Run:  GROQ_API_KEY=... python eval.py
"""
import pandas as pd

from agent import get_client, run_agent

# Labelled eval set: (ticket_text, expected_action, expected_team_if_escalate)
EVAL_SET = [
    # --- Should draft_response (clear KB coverage) ---
    ("How do I update my credit card on file?", "draft_response", None),
    ("Is there a dark mode option in the app?", "draft_response", None),
    ("What's the API rate limit on the Pro plan?", "draft_response", None),
    ("Can I export more than 50000 rows in one CSV?", "draft_response", None),
    ("How many people can I add to my team workspace on Pro?", "draft_response", None),
    ("I forgot my password, how do I reset it?", "draft_response", None),
    ("Can I downgrade my annual plan to monthly right now?", "draft_response", None),
    ("Does the mobile app work on Android 9?", "draft_response", None),

    # --- Should escalate: hard policy triggers (security / legal) ---
    ("I lost my phone and can't access my 2FA codes, I'm completely locked out.", "escalate", "security"),
    ("My authenticator app broke and now I can't get into my account at all.", "escalate", "security"),
    ("I'm going to contact my attorney about how you've been storing my information.", "escalate", "legal_privacy"),
    ("Please delete all my personal data, I'm invoking my GDPR rights.", "escalate", "legal_privacy"),
    ("This is a formal complaint, I'm filing with the regulator if this isn't fixed.", "escalate", "legal_privacy"),
    ("There's been a data breach on your end and I want to know what you're doing about it.", "escalate", "legal_privacy"),

    # --- Should escalate: outside KB coverage / low confidence ---
    ("Can you migrate my data from a competitor's tool automatically?", "escalate", "general"),
    ("I want a custom enterprise contract with different SLA terms.", "escalate", "general"),
    ("Your billing charged me twice this month and the amounts don't match my invoice.", "escalate", "billing"),

    # --- Should ask_clarifying_question: too vague ---
    ("It's not working.", "ask_clarifying_question", None),
    ("the app is weird", "ask_clarifying_question", None),
    ("help", "ask_clarifying_question", None),
    ("something is broken with my account", "ask_clarifying_question", None),
]

HARD_TRIGGER_TEAMS = ["security", "legal_privacy"]


def evaluate():
    client = get_client()
    rows = []
    for ticket_text, expected_action, expected_team in EVAL_SET:
        result = run_agent(ticket_text, client=client)
        rows.append({
            "ticket": ticket_text,
            "expected_action": expected_action,
            "expected_team": expected_team,
            "model_action": result["model_action"],
            "final_action": result["action"],
            "final_team": result["args"].get("team"),
            "policy_override": result["policy_override"],
        })
    df = pd.DataFrame(rows)
    df["model_correct"] = df["model_action"] == df["expected_action"]
    df["final_correct"] = df["final_action"] == df["expected_action"]
    return df


def summarise(df):
    hard = df[df["expected_team"].isin(HARD_TRIGGER_TEAMS)]
    return {
        "routing_accuracy_model": df["model_correct"].mean(),
        "routing_accuracy_final": df["final_correct"].mean(),
        "hard_trigger_recall_model": hard["model_correct"].mean(),
        "hard_trigger_recall_final": hard["final_correct"].mean(),
    }


if __name__ == "__main__":
    df = evaluate()
    df.to_csv("eval_results.csv", index=False)

    print("=" * 60)
    for name, value in summarise(df).items():
        print(f"{name:32s} {value:.1%}")
    print("=" * 60)

    print("\nAccuracy by expected action (final):")
    print(df.groupby("expected_action")["final_correct"].agg(["mean", "count"]))

    errors = df[~df["final_correct"]]
    print(f"\n{len(errors)} misclassified ticket(s):")
    print(errors[["ticket", "expected_action", "final_action"]].to_string(index=False))
