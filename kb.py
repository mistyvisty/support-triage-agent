"""
kb.py — the knowledge base and the search over it.

Step 1 of the pipeline: before the LLM decides anything, we always look up
the closest help articles. The embedding model and FAISS index are built once
and reused (lru_cache), so the app doesn't rebuild them on every ticket.
"""
from functools import lru_cache

import numpy as np

# Knowledge base: (doc_id, category, content)
KNOWLEDGE_BASE = [
    ("KB001", "billing", "Refunds are issued for cancellations made within 14 days of purchase. "
        "Refunds are processed to the original payment method within 5-7 business days. "
        "Refunds outside the 14-day window require manager approval."),
    ("KB002", "billing", "To update a payment method, go to Settings > Billing > Payment Methods. "
        "Failed payments retry automatically 3 times over 7 days before the subscription is paused."),
    ("KB003", "billing", "Annual plans can be downgraded to monthly only at the renewal date. "
        "Mid-cycle downgrades are not supported and will not be prorated."),
    ("KB004", "account", "Password resets are self-service via the 'Forgot Password' link on the login page. "
        "Reset links expire after 30 minutes."),
    ("KB005", "account", "Two-factor authentication (2FA) can be enabled in Settings > Security. "
        "If a user is locked out due to lost 2FA access, this requires identity verification "
        "by the Security team and CANNOT be resolved by self-service or by the support bot."),
    ("KB006", "account", "Account deletion is permanent after a 30-day grace period. "
        "Users can cancel a deletion request within that window from Settings > Account."),
    ("KB007", "technical", "The mobile app requires iOS 15+ or Android 10+. "
        "If the app crashes on launch, clearing cache or reinstalling resolves 90% of cases."),
    ("KB008", "technical", "API rate limits are 100 requests/minute on the Starter plan and "
        "1000 requests/minute on the Pro plan. Rate limit errors return HTTP 429."),
    ("KB009", "technical", "CSV export is limited to 50,000 rows per file on all plans. "
        "Larger exports should be split by date range."),
    ("KB010", "feature", "Team workspaces support up to 10 members on the Pro plan and "
        "unlimited members on the Enterprise plan."),
    ("KB011", "feature", "Dark mode is available in Settings > Appearance on web and mobile."),
    ("KB012", "feature", "Custom integrations via webhook are available on Pro and Enterprise plans only."),
    ("KB013", "policy", "Any request involving a legal threat, mention of a lawyer, regulatory complaint, "
        "or data breach concern must be escalated to the Legal/Trust & Safety team immediately, "
        "regardless of how minor the underlying issue seems."),
    ("KB014", "policy", "Requests to delete a user's data under GDPR/CCPA must be escalated to the "
        "Data Privacy team. Support cannot process these directly even if technically possible."),
]


@lru_cache(maxsize=1)
def _load_index():
    """Embed every KB document with MiniLM and put the vectors in a FAISS index."""
    import faiss
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    kb_embeddings = embedder.encode([doc[2] for doc in KNOWLEDGE_BASE])

    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(np.array(kb_embeddings).astype("float32"))
    return embedder, index


def search_knowledge_base(query: str, k: int = 3):
    """
    Return the top-k most relevant KB documents for a query.
    Lower distance = more relevant (L2 distance on embeddings).
    """
    embedder, index = _load_index()
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_id, category, content = KNOWLEDGE_BASE[idx]
        results.append({
            "doc_id": doc_id,
            "category": category,
            "content": content,
            "distance": float(dist),
        })
    return results
