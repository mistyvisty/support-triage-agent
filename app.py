"""
app.py — the website (Streamlit).

Run locally:  streamlit run app.py
The Groq key comes from Streamlit secrets (on the cloud) or the GROQ_API_KEY env var.
"""
import os

import streamlit as st

from agent import policy_trigger, run_agent
from kb import search_knowledge_base

st.set_page_config(page_title="Support Triage Agent", page_icon="🎫")

# On Streamlit Cloud the key is stored in Secrets; copy it to the env var agent.py reads
if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("No Groq API key found. Add it to .streamlit/secrets.toml or set GROQ_API_KEY.")
        st.stop()

SAMPLE_TICKETS = {
    "Simple question": "How do I update my credit card on file?",
    "Security lockout": "I lost my phone and can't access my 2FA codes, I'm completely locked out.",
    "Legal / GDPR": "Please delete all my personal data, I'm invoking my GDPR rights.",
    "Vague": "the app is weird",
    "Not in the KB": "Can you migrate my data from a competitor's tool automatically?",
}

st.title("🎫 AI Support Triage Agent")
st.write(
    "Paste a support ticket. The agent searches the help articles, then decides: "
    "**draft a reply**, **escalate to a human**, or **ask a clarifying question**. "
    "Security and legal tickets always go to a human."
)

tab_try, tab_eval = st.tabs(["Try it", "Evaluation"])

with tab_try:
    sample = st.selectbox("Pick a sample ticket, or write your own below",
                          ["(write my own)"] + list(SAMPLE_TICKETS))
    default_text = SAMPLE_TICKETS.get(sample, "")
    ticket = st.text_area("Support ticket", value=default_text, height=100)

    if st.button("Triage", type="primary", disabled=not ticket.strip()):
        with st.spinner("Searching the knowledge base and deciding..."):
            search_knowledge_base("warm up")  # loads the embedding model on first use
            result = run_agent(ticket)

        action, args = result["action"], result["args"]
        if action == "draft_response":
            st.success("✅ **Draft reply**")
            st.write(args.get("response_text", ""))
            st.caption(f"Sources: {', '.join(args.get('source_doc_ids', []))} · "
                       f"confidence: {args.get('confidence', '-')}")
        elif action == "escalate":
            st.error(f"🚨 **Escalated to {args.get('team', 'general')}** "
                     f"(priority: {args.get('priority', '-')})")
            st.write(args.get("reason", ""))
        elif action == "ask_clarifying_question":
            st.warning("❓ **Clarifying question**")
            st.write(args.get("question", ""))
        else:
            st.write("Something went wrong:", args)

        if result["policy_override"]:
            st.info(f"🛡️ Policy guard stepped in: the model chose "
                    f"`{result['model_action']}`, but this ticket matched a "
                    f"`{policy_trigger(ticket)}` rule, so it was escalated.")

        with st.expander("Help articles the agent found"):
            for r in result["retrieved"]:
                st.markdown(f"**{r['doc_id']}** ({r['category']}, distance {r['distance']:.2f}) "
                            f"— {r['content']}")

with tab_eval:
    st.subheader("How well does it route?")
    st.write("Tested on 21 hand-labelled tickets (run `python eval.py`).")
    col1, col2 = st.columns(2)
    col1.metric("Security / legal escalation recall", "100%", help="6 of 6 high-stakes tickets")
    col2.metric("Overall routing accuracy", "57.1%", help="12 of 21 tickets")
    st.write(
        "The agent is cautious on purpose: wrongly answering a security or legal ticket is "
        "dangerous, while asking one extra question is only annoying. Most mistakes were the "
        "model asking a clarifying question when the help article already had the answer."
    )
    st.caption("Numbers are from the original notebook run, before the code policy guard. "
               "Update them after re-running eval.py.")
