"""
Orchestrates the end-to-end flow:
  resolve media -> gather context -> retrieve evidence -> decide (batched) -> collect rows

Messages are grouped into batches of LLM_BATCH_SIZE before hitting the LLM
so the system prompt / JSON schema overhead is paid once per batch instead
of once per message — this is one of the two biggest levers (along with
the small-model failover in router/llm_client.py) for staying under a
tight daily token budget.
"""
from tqdm import tqdm

from config import LLM_BATCH_SIZE
from media.ocr import extract_text as ocr_extract_text
from media.asr import transcribe as asr_transcribe
from retrieval.evidence import get_evidence
from router.decision_engine import decide_batch
from router.token_budget import get_tracker
from features import (
    compute_content_signals,
    compute_user_sender_features,
    compute_evidence_summary,
)


def _resolve_content(data_context, message: dict):
    """Returns (resolved_text, media_kind) for a message, running OCR/ASR
    for media messages and falling back to message_text for text messages."""
    media_type = str(message.get("media_type") or "").strip().lower()
    media_id = message.get("media_id")

    if media_type == "image" and media_id:
        img_row = data_context.image_row(media_id) or {}
        filename = img_row.get("file_path") or img_row.get("filename")
        if filename:
            text = ocr_extract_text(str(filename).split("/")[-1])
            return text, "OCR text extracted from image"
        return "", "image (file not found)"

    if media_type == "voice" and media_id:
        vn_row = data_context.voice_row(media_id) or {}
        filename = vn_row.get("file_path") or vn_row.get("filename")
        if filename:
            text = asr_transcribe(str(filename).split("/")[-1])
            return text, "ASR transcript from voice note"
        return "", "voice note (file not found)"

    return str(message.get("message_text") or ""), "text"


def _build_item(data_context, message: dict) -> dict:
    resolved_text, media_kind = _resolve_content(data_context, message)

    user_ctx = data_context.user(message.get("user_id"))
    group_ctx = data_context.group(message.get(
        "group_id")) if message.get("group_id") else None
    group_member_ctx = (
        data_context.group_member(message.get(
            "group_id"), message.get("user_id"))
        if message.get("group_id") else None
    )
    business_ctx = data_context.business(message.get(
        "business_id")) if message.get("business_id") else None
    business_history_ctx = (
        data_context.user_business_history_row(
            message.get("user_id"), message.get("business_id"))
        if message.get("business_id") else None
    )
    daily_summary_ctx = data_context.daily_summary(message.get("user_id"))

    evidence = get_evidence(data_context, message, resolved_text)

    # Computed signals: keyword/content signals, user-sender relationship
    # signals, and a rollup of the retrieved evidence. These feed both the
    # pre/post-LLM heuristics in router/decision_engine.py and the
    # "=== COMPUTED SIGNALS ===" block at the top of every LLM prompt.
    content_signals = compute_content_signals(
        resolved_text, message.get("forwarded_count"))
    user_signals = compute_user_sender_features(data_context, message)
    evidence_summary = compute_evidence_summary(evidence)

    return {
        "message": message,
        "resolved_text": resolved_text,
        "media_kind": media_kind,
        "user_ctx": user_ctx,
        "group_ctx": group_ctx,
        "group_member_ctx": group_member_ctx,
        "business_ctx": business_ctx,
        "business_history_ctx": business_history_ctx,
        "daily_summary_ctx": daily_summary_ctx,
        "evidence": evidence,
        "content_signals": content_signals,
        "user_signals": user_signals,
        "evidence_summary": evidence_summary,
    }


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run_pipeline(data_context) -> list:
    messages = data_context.messages.to_dict(orient="records")

    # Media resolution / context lookup / evidence retrieval don't touch
    # the LLM at all, so they're done up front for every message before
    # any batching decision is made.
    items = [
        _build_item(data_context, message)
        for message in tqdm(messages, desc="Preparing messages")
    ]

    batch_size = max(1, LLM_BATCH_SIZE)
    decisions: dict = {}
    for batch in tqdm(list(_chunks(items, batch_size)), desc="Routing message batches"):
        decisions.update(decide_batch(batch))

    tracker = get_tracker()
    print(f"[pipeline] token budget after run: {tracker.status_line()}")

    results = []
    for message in messages:
        mid = str(message.get("message_id"))
        decision = decisions.get(mid)
        if decision is None:
            # Should not happen (decide_batch resolves every input item),
            # but never silently drop a required output row.
            decision = {
                "action": "digest",
                "message_type": "unknown",
                "reason": "Fallback default (no decision produced).",
                "confidence": 0.2,
                "evidence_message_ids": [],
                "model_used": "none",
            }
        results.append(
            {
                "message_id": message.get("message_id"),
                "action": decision["action"],
                "message_type": decision["message_type"],
                "reason": decision["reason"],
                "confidence": decision["confidence"],
                "evidence_message_ids": (
                    ";".join(decision["evidence_message_ids"])
                    if decision["evidence_message_ids"] else "none"
                ),
                # Debug-only field: which tier actually answered
                # (GROQ_MODEL_PRIMARY / GROQ_MODEL_FALLBACK / ollama:.../
                # pre_llm_heuristic / rule_based_fallback). output_writer.py
                # ignores this for the real submission CSV.
                "model_used": decision.get("model_used", "unknown"),
            }
        )

    return results
