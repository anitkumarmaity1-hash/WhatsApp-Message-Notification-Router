"""
Historical evidence retrieval with enhanced sender behavior summaries.

Improvements:
1. Same-thread boost increased to 0.25 (stronger signal)
2. Recency boost for recent messages
3. Sender behavior summary included with evidence
4. Better handling of empty/short texts
5. Evidence capped at top_k but with richer metadata
"""
import functools
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from config import EMBEDDING_MODEL, EVIDENCE_TOP_K


@functools.lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _embed(texts: List[str]) -> np.ndarray:
    model = _get_embedder()
    # Filter out empty texts
    valid_texts = [t for t in texts if t and str(t).strip()]
    if not valid_texts:
        return np.zeros((len(texts), 384))  # MiniLM-L6-v2 produces 384-dim vectors

    embeddings = model.encode(
        valid_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # If some texts were empty, we need to reconstruct the full array
    if len(valid_texts) != len(texts):
        full_embeddings = np.zeros((len(texts), embeddings.shape[1]))
        valid_idx = 0
        for i, t in enumerate(texts):
            if t and str(t).strip():
                full_embeddings[i] = embeddings[valid_idx]
                valid_idx += 1
        return np.asarray(full_embeddings)

    return np.asarray(embeddings)


def _get_sender_behavior_summary(data_context, user_id, sender_id, group_id, business_id) -> Dict[str, Any]:
    """Get a summary of how the user typically interacts with this sender/thread."""
    history = data_context.history_for_user(user_id)
    if history.empty:
        return {"total_messages": 0, "dominant_reaction": "none"}

    # Filter to relevant thread
    thread = pd.DataFrame()
    if sender_id and "sender_user_id" in history.columns:
        thread = history[history["sender_user_id"] == sender_id]
    elif business_id and "business_id" in history.columns:
        thread = history[history["business_id"] == business_id]
    elif group_id and "group_id" in history.columns:
        thread = history[history["group_id"] == group_id]

    if thread.empty:
        return {"total_messages": 0, "dominant_reaction": "none"}

    events_list = []
    for msg_id in thread["message_id"].tolist():
        ev = data_context.events_for_message(msg_id)
        if not ev.empty and "event_type" in ev.columns:
            events_list.extend(ev["event_type"].tolist())

    if not events_list:
        return {"total_messages": len(thread), "dominant_reaction": "none"}

    event_counts = {}
    for ev in events_list:
        event_counts[ev] = event_counts.get(ev, 0) + 1

    dominant = max(event_counts, key=event_counts.get)
    total = len(events_list)

    return {
        "total_messages": len(thread),
        "dominant_reaction": dominant,
        "open_rate": round(event_counts.get("opened", 0) / total, 2),
        "reply_rate": round(event_counts.get("replied", 0) / total, 2),
        "dismiss_rate": round(event_counts.get("dismissed", 0) / total, 2),
        "mute_rate": round(event_counts.get("muted", 0) / total, 2),
        "report_rate": round(event_counts.get("reported", 0) / total, 2),
    }


def get_evidence(
    data_context,
    message: dict,
    resolved_text: str,
    top_k: int = EVIDENCE_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: {message_id, text, created_at, events, score, sender_behavior}
    representing the most relevant historical messages for this user.
    """
    history = data_context.history_for_user(message.get("user_id"))
    if history.empty:
        return []

    text_col = "message_text" if "message_text" in history.columns else None
    id_col = "message_id" if "message_id" in history.columns else history.columns[0]

    candidates = history.copy()

    # Prefer same-thread history
    same_thread_mask = pd.Series([False] * len(candidates), index=candidates.index)
    for key in ("sender_user_id", "group_id", "business_id"):
        if key in candidates.columns and message.get(key) not in (None, "", float("nan")):
            same_thread_mask = same_thread_mask | (candidates[key] == message.get(key))

    same_thread = candidates[same_thread_mask]
    rest = candidates[~same_thread_mask]

    # Cap candidate pool
    pool = pd.concat([same_thread, rest]).head(200)
    if pool.empty:
        return []

    pool_texts = pool[text_col].fillna("").astype(str).tolist() if text_col else [""] * len(pool)

    try:
        query_vec = _embed([resolved_text or ""])[0]
        pool_vecs = _embed(pool_texts)
        sims = pool_vecs @ query_vec
    except Exception as e:
        print(f"[retrieval] embedding failed ({e}); falling back to recency only")
        sims = np.zeros(len(pool))

    # Boost same-thread candidates
    boost = np.array([0.25 if idx in same_thread.index else 0.0 for idx in pool.index])

    # Recency boost: more recent messages get a small boost
    if "created_at" in pool.columns:
        try:
            pool["_dt"] = pd.to_datetime(pool["created_at"], errors="coerce")
            max_dt = pool["_dt"].max()
            if pd.notna(max_dt):
                days_old = (max_dt - pool["_dt"]).dt.total_seconds() / 86400
                recency_boost = np.exp(-days_old.fillna(30) / 7) * 0.1  # half-life of 7 days
                boost = boost + recency_boost.fillna(0).values
        except Exception:
            pass

    scores = sims + boost
    ranked_idx = np.argsort(-scores)[:top_k]

    # Get sender behavior summary
    sender_behavior = _get_sender_behavior_summary(
        data_context,
        message.get("user_id"),
        message.get("sender_user_id"),
        message.get("group_id"),
        message.get("business_id"),
    )

    evidence = []
    for i in ranked_idx:
        row = pool.iloc[i]
        msg_id = row[id_col]
        events_df = data_context.events_for_message(msg_id)
        events = []
        if not events_df.empty and "event_type" in events_df.columns:
            events = events_df["event_type"].dropna().tolist()

        text = row.get(text_col, "") if text_col else ""
        # Skip evidence with empty text unless it's the only option
        if not text or not str(text).strip():
            continue

        evidence.append(
            {
                "message_id": msg_id,
                "text": text,
                "created_at": row.get("created_at", ""),
                "events": events,
                "score": float(scores[i]),
                "sender_behavior": sender_behavior if i == ranked_idx[0] else None,
            }
        )

    return evidence
