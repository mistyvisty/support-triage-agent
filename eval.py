"""
eval.py — the test. Runs the agent on labelled tickets and reports:
  - overall routing accuracy
  - hard-trigger escalation recall (security / legal) — the high-stakes metric
  - policy guard false-trigger rate — how often the regex guard fires on
    tickets that didn't actually need hard-trigger escalation
Both accuracy metrics are shown for the model alone AND with the code policy
guard, so we can see what the guard adds.

Run:  GROQ_API_KEY=... python eval.py
"""
import pandas as pd

from agent import get_client, run_agent, policy_trigger

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

    # --- Adversarial: security lockouts phrased WITHOUT the guard's exact keywords (2fa/two-factor/authenticator) ---
    ("I set up that extra login code thing a while back and now I don't have access to the device it goes to. I'm completely locked out.", "escalate", "security"),
    ("My phone broke and now the app won't let me log in because it wants some kind of verification code sent to that phone.", "escalate", "security"),
    ("I lost the device I use for that extra verification step when logging in, and the backup codes don't work either.", "escalate", "security"),
    ("I can't log in — it's asking for a code from an authentication app I don't have installed anymore.", "escalate", "security"),
    ("Someone changed the security settings on my account and now I'm locked out of the verification step I normally use.", "escalate", "security"),
    ("I reset my phone and lost access to the app that generates my login codes. Account's locked now.", "escalate", "security"),

    # --- Adversarial: legal/privacy threats phrased WITHOUT the guard's exact keywords ---
    ("I want all my personal information removed from your systems permanently, as is my right under privacy regulations in my country.", "escalate", "legal_privacy"),
    ("Please delete everything you have on me. I know my rights under European data protection law and expect this handled properly.", "escalate", "legal_privacy"),
    ("I'm considering taking this further with outside counsel if this isn't resolved soon.", "escalate", "legal_privacy"),
    ("My company's compliance team is now looking into how you handled my information after this incident.", "escalate", "legal_privacy"),
    ("I'll be filing a formal complaint with the authorities about how this was handled.", "escalate", "legal_privacy"),
    ("This needs to stop or I'll have to take formal action against your company.", "escalate", "legal_privacy"),

    # --- Adversarial: contains a trigger keyword but is NOT actually a hard-trigger situation ---
    ("My cousin, who happens to be a lawyer, said this cancellation fee seems a bit much. Can you waive it?", "draft_response", None),
    ("I read on the news that a totally different company had a data breach recently, which reminded me to ask — do you store my credit card number?", "draft_response", None),
    ("Is two-factor authentication required or optional? Just want to know before I turn it on.", "draft_response", None),
    ("If I don't get a response soon I'll have to sue you guys lol, jk but seriously can someone help with my export limit?", "draft_response", None),
    ("I'm not threatening legal action or anything, just wondering what your refund policy is.", "draft_response", None),
    ("What happens if my 2FA authenticator app loses sync — is that something you can just fix for me, or is it a bigger deal?", "draft_response", None),

    # --- More draft_response: clear KB-covered questions, varied phrasing ---
    ("I canceled my subscription 3 days ago, when will I get my refund and how?", "draft_response", None),
    ("My card on file expired, how do I add a new one?", "draft_response", None),
    ("Can I switch my annual plan to monthly right now, mid-cycle?", "draft_response", None),
    ("I forgot my password, how do I get back in?", "draft_response", None),
    ("I accidentally requested account deletion, can I undo it?", "draft_response", None),
    ("The app crashes every time I open it on my Android phone.", "draft_response", None),
    ("I keep getting 429 errors from the API, what's the limit on the Starter plan?", "draft_response", None),
    ("My CSV export got cut off at 50,000 rows, is that expected?", "draft_response", None),
    ("How many people can I add to my workspace on the Pro plan?", "draft_response", None),
    ("Where do I turn on dark mode in the mobile app?", "draft_response", None),
    ("Are webhooks available if I'm on the Starter plan?", "draft_response", None),

    # --- More ask_clarifying_question: genuinely too vague to act on ---
    ("It's not working.", "ask_clarifying_question", None),
    ("Can you fix my account issue from last week?", "ask_clarifying_question", None),
    ("I need this changed please.", "ask_clarifying_question", None),
    ("Something's wrong with my plan, can you check?", "ask_clarifying_question", None),
    ("I have a question about the thing we discussed.", "ask_clarifying_question", None),
    ("This is broken, please help.", "ask_clarifying_question", None),

    # --- More escalations: general/billing, outside self-service but not hard-trigger ---
    ("I was charged twice for the same month, I need an actual refund processed beyond the normal window, it's been 3 months.", "escalate", "billing"),
    ("I want a refund for my annual plan from 6 months ago, I just didn't notice the charge.", "escalate", "billing"),
    ("Can someone manually review my account because I think I'm being charged for a plan I downgraded from already?", "escalate", "billing"),
    ("I need someone to personally look into a billing discrepancy across the last 4 invoices, the numbers don't add up and I've checked twice.", "escalate", "billing"),

    # --- More escalations: out-of-KB topics ---
    ("Do you support single sign-on (SSO) with Okta for enterprise customers?", "escalate", "general"),
    ("I need a custom integration built for our internal tool, is that something you offer?", "escalate", "general"),
    ("What's your roadmap for AI features this year?", "escalate", "general"),
    ("Can I get a hardware device reader to work with your product?", "escalate", "general"),
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


def false_trigger_rate():
    """Of tickets where the regex guard fires at all, what fraction
    didn't actually need hard-trigger escalation? Independent of the
    model -- this measures the guard's own precision."""
    triggered = [(text, exp_team) for text, _, exp_team in EVAL_SET if policy_trigger(text)]
    if not triggered:
        return 0.0, []
    false_positives = [(text, exp_team) for text, exp_team in triggered if exp_team not in HARD_TRIGGER_TEAMS]
    return len(false_positives) / len(triggered), false_positives


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

    ft_rate, ft_examples = false_trigger_rate()
    print(f"policy_guard_false_trigger_rate {ft_rate:.1%}  ({len(ft_examples)} false positives)")
    for text, exp_team in ft_examples:
        print(f"  FALSE TRIGGER (expected {exp_team or 'no escalation'}): {text[:70]}...")
    print("=" * 60)

    print("\nAccuracy by expected action (final):")
    print(df.groupby("expected_action")["final_correct"].agg(["mean", "count"]))

    errors = df[~df["final_correct"]]
    print(f"\n{len(errors)} misclassified ticket(s):")
    print(errors[["ticket", "expected_action", "final_action"]].to_string(index=False))
