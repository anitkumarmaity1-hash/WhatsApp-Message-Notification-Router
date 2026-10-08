"""
Validates and normalises the LLM's raw decision into the exact contract
required by output.csv, with a deterministic rule-based safety net so a
single bad LLM call can never produce an invalid row.

Improvements:
1. Pre-LLM heuristics for obvious cases (saves tokens, guarantees accuracy)
2. Post-LLM correction based on computed signals
3. Better confidence calibration
4. Stronger scam/spam detection
5. Conservative event/promotion heuristics to avoid false positives
"""
from router.prompts import (
    SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT,
    build_user_prompt, build_batch_user_prompt,
    ALLOWED_ACTIONS, ALLOWED_TYPES,
)
from router.llm_client import call_llm_json

# ---------------------------------------------------------------------------
# Pre-LLM heuristic rules — for obvious cases we can decide without the LLM
# ---------------------------------------------------------------------------

from typing import Optional


def _pre_llm_heuristic(
    content_signals: dict,
    user_signals: dict,
    evidence_summary: dict,
    resolved_text: str,
    message: dict,
) -> Optional[dict]:
    """
    Returns a decision dict if the case is obvious enough to skip the LLM,
    otherwise returns None to let the LLM handle it.

    These rules are CONSERVATIVE — they only fire when confidence is very high.
    """
    text = (resolved_text or "").lower()
    fwd_count = content_signals.get("forwarded_count", 0)

    # RULE 1: Clear scam — always mute
    if content_signals.get("has_scam") or content_signals.get("fake_urgency"):
        scam_kw = content_signals.get("scam_keywords", [])
        reason_kw = scam_kw[0] if scam_kw else "suspicious content"
        return {
            "action": "mute",
            "message_type": "scam",
            "reason": f"Pre-heuristic: clear scam pattern detected ({reason_kw}). Muted for safety.",
            "confidence": 0.95,
            "evidence_message_ids": [],
        }

    # RULE 2: heavily forwarded + low engagement -> mute is right regardless,
    # but the TYPE should reflect what the content actually is (a forwarded
    # greeting is still a "greeting", not "spam") rather than being hardcoded.
    if fwd_count >= 5 and user_signals.get("sender_reply_rate", 0) < 0.1:
        if content_signals.get("has_greeting"):
            inferred_type = "greeting"
        elif content_signals.get("has_forward"):
            inferred_type = "forward"
        else:
            inferred_type = "spam"
        return {
            "action": "mute",
            "message_type": inferred_type,
            "reason": f"Pre-heuristic: heavily forwarded message (x{fwd_count}) from low-engagement sender. Muted.",
            "confidence": 0.9,
            "evidence_message_ids": [],
        }

    # RULE 3: Clear spam — dial/press patterns WITHOUT a legitimate promo signal.
    # "dial 1800-xxx to book now" is a normal promotional CTA, not spam — only
    # fire this when has_promo is NOT also true, otherwise let the LLM (which
    # sees the full flowchart's promotion-vs-spam distinction) decide.
    if (
        content_signals.get("has_spam")
        and not content_signals.get("has_promo")
        and ("dial" in text or "press" in text or "unsubscribe" in text)
    ):
        return {
            "action": "mute",
            "message_type": "spam",
            "reason": "Pre-heuristic: bulk broadcast with dial/press instructions. Muted as spam.",
            "confidence": 0.9,
            "evidence_message_ids": [],
        }

    # RULE 4: Group muted + not urgent/payment -> mute
    if user_signals.get("group_muted") and not content_signals.get("has_urgent") and not content_signals.get("has_payment"):
        return {
            "action": "mute",
            "message_type": "spam" if content_signals.get("has_spam") else "forward" if content_signals.get("has_forward") else "unknown",
            "reason": "Pre-heuristic: user has muted this group and message is not urgent/payment.",
            "confidence": 0.85,
            "evidence_message_ids": [],
        }

    # RULE 5: Business opted out + promotion -> mute
    if user_signals.get("business_opted_out") and content_signals.get("has_promo"):
        return {
            "action": "mute",
            "message_type": "promotion",
            "reason": "Pre-heuristic: user opted out of this business and message is promotional.",
            "confidence": 0.9,
            "evidence_message_ids": [],
        }

    # RULE 6: Clear urgent health/family emergency
    health_emergency_kws = ["dad is unwell", "mom is unwell", "not well", "going to hospital",
                            "going to clinic", "accident", "blood needed", "emergency"]
    if any(kw in text for kw in health_emergency_kws) and not content_signals.get("has_scam"):
        return {
            "action": "notify",
            "message_type": "urgent",
            "reason": "Pre-heuristic: family health emergency requires immediate attention.",
            "confidence": 0.95,
            "evidence_message_ids": [],
        }

    # RULE 7: Clear event with STRONG keywords and time (pickup, meeting, flight, gate)
    # Only fire for unambiguous schedule changes, NOT delivery notifications
    event_kw = content_signals.get("event_keywords", [])
    strong_event_present = any(kw in text for kw in ["pickup", "gate", "meeting at", "flight", "appointment",
                                                     "rescheduled", "moved to", "changed to", "cab", "driver"])
    if content_signals.get("has_event") and content_signals.get("has_time_date") and strong_event_present and not content_signals.get("has_scam"):
        changed = any(kw in text for kw in [
            "changed", "change", "moved", "rescheduled", "postponed", "cancelled", "canceled",
            "delayed", "preponed"])
        same_day_early = "early" in text and any(kw in text for kw in ["today", "mins", "minutes", "tonight"])
        action = "notify" if (changed or same_day_early) else "digest"
        return {
            "action": action,
            "message_type": "event",
            "reason": f"Pre-heuristic: schedule change/event detected ({event_kw[0] if event_kw else 'time-specific'}).",
            "confidence": 0.88,
            "evidence_message_ids": [],
        }

    # RULE 8: Clear promotion with STRONG keywords
    # Only fire if there are strong promo signals (discount, sale, % off, etc.)
    promo_kw = content_signals.get("promo_keywords", [])
    strong_promo_present = any(kw in text for kw in ["% off", "percent off", "discount", "sale", "cashback",
                                                     "offer", "deal", "buy now", "book now", "prime day",
                                                     "free for lifetime", "starting from", "up to"])
    if strong_promo_present and not content_signals.get("has_scam"):
        if user_signals.get("business_has_history") or user_signals.get("sender_open_rate", 0) > 0.3:
            action = "digest"
        else:
            dismiss = max(user_signals.get("sender_dismiss_rate", 0),
                          user_signals.get("user_dismissal_rate", 0))
            action = "mute" if dismiss > 0.5 else "digest"
        return {
            "action": action,
            "message_type": "promotion",
            "reason": f"Pre-heuristic: promotional content detected ({promo_kw[0] if promo_kw else 'offer'}).",
            "confidence": 0.85,
            "evidence_message_ids": [],
        }

    # RULE 9: Clear payment with failure/overdue
    if content_signals.get("has_payment") and not content_signals.get("has_scam"):
        if any(kw in text for kw in ["failed", "overdue", "blocked", "declined", "rejected"]):
            return {
                "action": "notify",
                "message_type": "payment",
                "reason": "Pre-heuristic: failed/overdue payment requires immediate attention.",
                "confidence": 0.9,
                "evidence_message_ids": [],
            }
        elif any(kw in text for kw in ["invoice", "bill", "amount due", "total due", "pending"]):
            return {
                "action": "digest",
                "message_type": "payment",
                "reason": "Pre-heuristic: payment-related message (invoice/bill).",
                "confidence": 0.8,
                "evidence_message_ids": [],
            }

    # RULE 10: Clear greeting
    if content_signals.get("has_greeting") and not any([
        content_signals.get("has_urgent"), content_signals.get("has_event"),
        content_signals.get("has_payment"), content_signals.get("has_promo"),
        content_signals.get("has_scam"), content_signals.get("has_spam"),
    ]):
        return {
            "action": "digest",
            "message_type": "greeting",
            "reason": "Pre-heuristic: purely social greeting with no actionable content.",
            "confidence": 0.9,
            "evidence_message_ids": [],
        }

    # No obvious case — let the LLM decide
    return None


# ---------------------------------------------------------------------------
# Post-LLM correction rules
# ---------------------------------------------------------------------------

def _post_llm_correction(
    decision: dict,
    content_signals: dict,
    user_signals: dict,
    evidence_summary: dict,
) -> dict:
    """
    Apply corrections to the LLM's decision based on hard signals.
    This catches cases where the LLM ignored the computed features.
    """
    action = decision.get("action")
    msg_type = decision.get("message_type")
    confidence = decision.get("confidence", 0.5)
    reason = decision.get("reason", "")
    evidence_ids = decision.get("evidence_message_ids", [])

    # Pre-compute strong event flag for use in multiple corrections
    event_kw = content_signals.get("event_keywords", [])
    strong_event = any(kw in (" ".join(event_kw)).lower() for kw in [
                       "pickup", "gate", "meeting", "flight", "appointment", "rescheduled"])

    # CORRECTION 1: LLM missed scam
    if (content_signals.get("has_scam") or content_signals.get("fake_urgency")) and msg_type != "scam":
        return {
            "action": "mute",
            "message_type": "scam",
            "reason": f"Corrected: LLM missed scam signals. {reason}",
            "confidence": 0.92,
            "evidence_message_ids": evidence_ids,
        }

    # CORRECTION 2: LLM said business_update but it's clearly promotion
    if msg_type == "business_update" and content_signals.get("has_promo"):
        promo_kw = content_signals.get("promo_keywords", [])
        return {
            "action": action,
            "message_type": "promotion",
            "reason": f"Corrected: message contains promotional keywords ({promo_kw[0] if promo_kw else 'offer'}), not pure business update. {reason}",
            "confidence": max(confidence, 0.75),
            "evidence_message_ids": evidence_ids,
        }

    # CORRECTION 3: LLM said business_update but it's clearly event
    # Only override if there are STRONG event signals (not just a delivery date)
    if msg_type == "business_update" and content_signals.get("has_event") and content_signals.get("has_time_date") and strong_event:
        return {
            "action": "notify" if action == "digest" and content_signals.get("has_urgent") else action,
            "message_type": "event",
            "reason": f"Corrected: message contains event/time-specific information, not pure business update. {reason}",
            "confidence": max(confidence, 0.75),
            "evidence_message_ids": evidence_ids,
        }

    # CORRECTION 4: LLM said business_update but it's clearly spam
    if msg_type == "business_update" and content_signals.get("has_spam") and content_signals.get("forwarded_count", 0) >= 3:
        return {
            "action": "mute",
            "message_type": "spam",
            "reason": f"Corrected: bulk broadcast spam detected, not business update. {reason}",
            "confidence": 0.85,
            "evidence_message_ids": evidence_ids,
        }

    # CORRECTION 5: LLM said urgent but it's a scam
    if msg_type == "urgent" and content_signals.get("has_scam"):
        return {
            "action": "mute",
            "message_type": "scam",
            "reason": f"Corrected: fake urgency used in scam. {reason}",
            "confidence": 0.92,
            "evidence_message_ids": evidence_ids,
        }

    # CORRECTION 6: LLM said notify for spam/scam
    if action == "notify" and msg_type in ("spam", "scam"):
        return {
            "action": "mute",
            "message_type": msg_type,
            "reason": f"Corrected: spam/scam should never notify. {reason}",
            "confidence": 0.9,
            "evidence_message_ids": evidence_ids,
        }

    # CORRECTION 7: LLM said digest for clear urgent
    if action == "digest" and msg_type == "urgent" and content_signals.get("has_urgent") and not content_signals.get("has_scam"):
        # Only override if it's a real urgent situation
        urgent_kw = content_signals.get("urgent_keywords", [])
        real_urgent = any(kw in (" ".join(urgent_kw)).lower() for kw in [
                          "emergency", "unwell", "hospital", "incident", "failing", "blocked", "deadline", "outage", "error"])
        if real_urgent:
            return {
                "action": "notify",
                "message_type": "urgent",
                "reason": f"Corrected: real urgent situation should notify. {reason}",
                "confidence": 0.85,
                "evidence_message_ids": evidence_ids,
            }

    # CORRECTION 8: LLM said digest for clear event happening soon
    if action == "digest" and msg_type == "event" and content_signals.get("has_urgent"):
        return {
            "action": "notify",
            "message_type": "event",
            "reason": f"Corrected: time-critical event should notify. {reason}",
            "confidence": 0.8,
            "evidence_message_ids": evidence_ids,
        }

    # CORRECTION 9: LLM said mute for legitimate business update from verified sender with history
    if action == "mute" and msg_type in ("business_update", "payment") and user_signals.get("business_verified") and user_signals.get("business_has_history"):
        return {
            "action": "digest",
            "message_type": msg_type,
            "reason": f"Corrected: legitimate verified business with user history should not be muted. {reason}",
            "confidence": 0.8,
            "evidence_message_ids": evidence_ids,
        }

    # CORRECTION 10: LLM said unknown but we have clear signals
    if msg_type == "unknown":
        if content_signals.get("has_promo"):
            return {
                "action": action,
                "message_type": "promotion",
                "reason": f"Corrected: promotional content detected. {reason}",
                "confidence": 0.75,
                "evidence_message_ids": evidence_ids,
            }
        if content_signals.get("has_event") and strong_event:
            return {
                "action": action,
                "message_type": "event",
                "reason": f"Corrected: event/time-specific content detected. {reason}",
                "confidence": 0.75,
                "evidence_message_ids": evidence_ids,
            }
        if content_signals.get("has_payment"):
            return {
                "action": action,
                "message_type": "payment",
                "reason": f"Corrected: payment content detected. {reason}",
                "confidence": 0.75,
                "evidence_message_ids": evidence_ids,
            }
        if content_signals.get("has_greeting") and not any([
            content_signals.get("has_urgent"), content_signals.get("has_scam"),
            content_signals.get("has_spam"), content_signals.get("has_promo"),
        ]):
            return {
                "action": action,
                "message_type": "greeting",
                "reason": f"Corrected: greeting content detected. {reason}",
                "confidence": 0.75,
                "evidence_message_ids": evidence_ids,
            }

    # CORRECTION 11: mute is reserved for unwanted content.
    # In the labeled data, mute only ever appears for scam / spam / forward / promotion / greeting
    # (the greeting being in a muted group) and never for personal, event, business_update,
    # unknown, payment or urgent messages. If the LLM mutes one of those without a concrete reason
    # (muted group, opt-out, spam/scam/promo/forward signals, or a sender the user ignores),
    # the safe, useful action is digest - not silently dropping it.
    weak_mute_types = {"personal", "event", "business_update", "unknown", "payment", "urgent", "greeting"}
    has_mute_reason = any([
        user_signals.get("group_muted"), user_signals.get("business_opted_out"),
        content_signals.get("has_scam"), content_signals.get("has_spam"),
        content_signals.get("has_promo"), content_signals.get("has_forward"),
        user_signals.get("sender_dismiss_rate", 0) > 0.6,
        user_signals.get("sender_mute_rate", 0) > 0.3,
    ])
    if decision.get("action") == "mute" and decision.get("message_type") in weak_mute_types and not has_mute_reason:
        decision = {**decision, "action": "digest", "confidence": min(float(confidence), 0.75),
                    "reason": f"Corrected mute->digest: nothing here is unwanted. {reason}"}

    return decision


# ---------------------------------------------------------------------------
# Validation and fallback
# ---------------------------------------------------------------------------

def _rule_based_fallback(
    content_signals: dict,
    user_signals: dict,
    evidence_summary: dict,
    resolved_text: str,
    forwarded_count,
) -> dict:
    """Fallback when LLM call fails entirely."""
    text_lower = (resolved_text or "").lower()

    # Priority 1: Scam
    if content_signals.get("has_scam") or content_signals.get("fake_urgency"):
        return {
            "action": "mute",
            "message_type": "scam",
            "reason": "Fallback: scam pattern detected (LLM call failed).",
            "confidence": 0.5,
            "evidence_message_ids": [],
        }

    # Priority 2: Spam
    if content_signals.get("has_spam") or content_signals.get("forwarded_count", 0) >= 5:
        return {
            "action": "mute",
            "message_type": "spam",
            "reason": "Fallback: spam pattern detected (LLM call failed).",
            "confidence": 0.5,
            "evidence_message_ids": [],
        }

    # Priority 3: Group muted
    if user_signals.get("group_muted"):
        return {
            "action": "mute",
            "message_type": "unknown",
            "reason": "Fallback: group is muted (LLM call failed).",
            "confidence": 0.4,
            "evidence_message_ids": [],
        }

    # Priority 4: Business opted out
    if user_signals.get("business_opted_out") and content_signals.get("has_promo"):
        return {
            "action": "mute",
            "message_type": "promotion",
            "reason": "Fallback: opted-out business promotion (LLM call failed).",
            "confidence": 0.4,
            "evidence_message_ids": [],
        }

    # Default
    return {
        "action": "digest",
        "message_type": "unknown",
        "reason": "Fallback default: routed to digest pending review (LLM call failed).",
        "confidence": 0.2,
        "evidence_message_ids": [],
    }


def _validate(decision: dict, evidence: list) -> dict:
    valid_ids = {str(e["message_id"]) for e in evidence}

    action = decision.get("action")
    if action not in ALLOWED_ACTIONS:
        action = "digest"

    message_type = decision.get("message_type")
    if message_type not in ALLOWED_TYPES:
        message_type = "unknown"

    reason = str(decision.get("reason") or "").strip() or "No reason provided."

    try:
        confidence = float(decision.get("confidence", 0.5))
    except (ValueError, TypeError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    raw_ids = decision.get("evidence_message_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [x.strip() for x in raw_ids.split(
            ";") if x.strip() and x.strip().lower() != "none"]
    clean_ids = [str(i) for i in raw_ids if str(i) in valid_ids]

    return {
        "action": action,
        "message_type": message_type,
        "reason": reason,
        "confidence": confidence,
        "evidence_message_ids": clean_ids,
    }


# ---------------------------------------------------------------------------
# Main decision functions
# ---------------------------------------------------------------------------

def decide(
    message: dict,
    resolved_text: str,
    media_kind: str,
    user_ctx,
    group_ctx,
    group_member_ctx,
    business_ctx,
    business_history_ctx,
    daily_summary_ctx,
    evidence: list,
    content_signals: dict,
    user_signals: dict,
    evidence_summary: dict,
) -> dict:

    # STEP 1: Try pre-LLM heuristic
    heuristic = _pre_llm_heuristic(
        content_signals, user_signals, evidence_summary, resolved_text, message
    )
    if heuristic is not None:
        result = _validate(heuristic, evidence)
        result["model_used"] = "pre_llm_heuristic"
        return result

    # STEP 2: Call LLM
    user_prompt = build_user_prompt(
        message, resolved_text, media_kind, user_ctx, group_ctx,
        group_member_ctx, business_ctx, business_history_ctx,
        daily_summary_ctx, evidence, content_signals, user_signals, evidence_summary,
    )

    try:
        raw_decision, model_used = call_llm_json(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        print(
            f"[decision_engine] LLM failed for {message.get('message_id')}: {e}")
        raw_decision = _rule_based_fallback(
            content_signals, user_signals, evidence_summary, resolved_text, message.get(
                "forwarded_count")
        )
        model_used = "rule_based_fallback"

    # STEP 3: Validate
    validated = _validate(raw_decision, evidence)

    # STEP 4: Post-LLM correction
    corrected = _post_llm_correction(
        validated, content_signals, user_signals, evidence_summary)

    # Re-validate after correction
    result = _validate(corrected, evidence)
    result["model_used"] = model_used
    return result


def decide_batch(items: list) -> dict:
    """
    `items`: list of dicts with keys for build_user_prompt plus
    "content_signals", "user_signals", "evidence_summary".

    Returns {message_id: validated_decision_dict}.
    """
    if not items:
        return {}

    if len(items) == 1:
        item = items[0]
        decision = decide(
            item["message"], item["resolved_text"], item["media_kind"],
            item["user_ctx"], item["group_ctx"], item["group_member_ctx"],
            item["business_ctx"], item["business_history_ctx"],
            item["daily_summary_ctx"], item["evidence"],
            item.get("content_signals", {}),
            item.get("user_signals", {}),
            item.get("evidence_summary", {}),
        )
        return {str(item["message"].get("message_id")): decision}

    # Try to resolve as many as possible with heuristics first
    results: dict = {}
    unresolved = []

    for item in items:
        mid = str(item["message"].get("message_id"))
        heuristic = _pre_llm_heuristic(
            item.get("content_signals", {}),
            item.get("user_signals", {}),
            item.get("evidence_summary", {}),
            item.get("resolved_text", ""),
            item["message"],
        )
        if heuristic is not None:
            validated = _validate(heuristic, item["evidence"])
            validated["model_used"] = "pre_llm_heuristic"
            results[mid] = validated
        else:
            unresolved.append(item)

    # If all resolved by heuristics, return early
    if not unresolved:
        return results

    # Batch LLM call for remaining items
    batch_prompt = build_batch_user_prompt(unresolved)
    still_unresolved = list(unresolved)

    try:
        raw, model_used = call_llm_json(BATCH_SYSTEM_PROMPT, batch_prompt)
        decisions = raw.get("decisions", {}) if isinstance(raw, dict) else {}
        new_unresolved = []
        for item in unresolved:
            mid = str(item["message"].get("message_id"))
            raw_decision = decisions.get(mid)
            if raw_decision is None:
                new_unresolved.append(item)
                continue
            validated = _validate(raw_decision, item["evidence"])
            corrected = _post_llm_correction(
                validated,
                item.get("content_signals", {}),
                item.get("user_signals", {}),
                item.get("evidence_summary", {}),
            )
            final = _validate(corrected, item["evidence"])
            final["model_used"] = model_used
            results[mid] = final
        still_unresolved = new_unresolved
    except Exception as e:
        print(
            f"[decision_engine] batch LLM call failed ({len(unresolved)} messages): {e}")
        still_unresolved = unresolved

    # Resolve any remaining items individually
    for item in still_unresolved:
        mid = str(item["message"].get("message_id"))
        print(
            f"[decision_engine] resolving {mid} individually (missing from batch)")
        decision = decide(
            item["message"], item["resolved_text"], item["media_kind"],
            item["user_ctx"], item["group_ctx"], item["group_member_ctx"],
            item["business_ctx"], item["business_history_ctx"],
            item["daily_summary_ctx"], item["evidence"],
            item.get("content_signals", {}),
            item.get("user_signals", {}),
            item.get("evidence_summary", {}),
        )
        results[mid] = decision

    return results