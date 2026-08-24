"""
Feature extraction module.
Computes derived signals from raw data to help the LLM make better decisions.
These features are designed to be explicit, interpretable, and actionable.
"""
import re
from typing import Dict, Any, List
import pandas as pd

# ---------------------------------------------------------------------------
# Keyword lists for content classification
# ---------------------------------------------------------------------------

URGENT_KEYWORDS = [
    "emergency", "urgent", "asap", "immediately", "now", "today", "right now",
    "hurry", "critical", "alert", "breaking", "please call", "please reply",
    "waiting", "deadline", "due today", "expires today", "account blocked",
    "payment failed", "failing", "incident", "outage", "down", "error",
    "bridge", "join now", "pickup", "gate", "changed to", "moved to",
    "rescheduled", "postponed", "cancelled", "dad is unwell", "mom is unwell",
    "not well", "going to hospital", "going to clinic", "unwell", "sick", "accident",
    "blood needed", "help needed", "please come", "reach by", "reach before",
    "don\'t forget", "reminder", "last chance", "final notice", "overdue",
    "blocked", "suspended", "verification needed", "verify now", "act now",
    "only today", "ends today", "last day", "closing soon", "live users",
    "checkout error", "spiking again", "join the bridge",
    "eod", "end of day", "before eod", "escalation", "escalates",
    "retry count", "alert threshold", "crossed the threshold", "come online",
    "need quick help", "on-call", "paging", "page me"
]

FAKE_URGENCY_PATTERNS = [
    "account will be blocked", "verify immediately", "urgent action required",
    "your account has been suspended", "confirm now or lose", "act now to claim",
    "otp received", "share the otp", "share otp", "send otp", "verify otp",
    "bank account will be blocked", "card will be blocked", "immediate action",
    "within 24 hours or", "expires in 24 hours", "limited time offer",
    "you have won", "congratulations you have won", "claim your prize now",
    "urgent: verify", "urgent: update", "urgent: confirm", "share the ott",
    "complete verification immediately"
]

SCAM_KEYWORDS = [
    "otp", "verify your account", "click this link", "suspended", "you have won",
    "lottery", "prize", "congratulations you won", "wire transfer", "crypto investment",
    "bitcoin", "double your money", "guaranteed returns", "act now", "limited time",
    "confirm your password", "bank details", "card details", "cvv",
    "kyc", "verification required", "unusual activity", "login attempt",
    "free gift", "won a prize", "claim your reward", "cash prize", "lucky winner",
    "send money to", "transfer to this account", "click here to verify",
    "your account is locked", "security alert", "unauthorized access",
    "update your details", "re-verify", "confirm identity", "validate account",
    "share the otp you received", "share the ott", "ott you received"
]

PROMO_KEYWORDS_STRONG = [
    "% off", "percent off", "discount", "sale", "cashback", "offer", "deal",
    "buy now", "shop now", "limited period", "flash sale", "mega sale",
    "prime day", "special offer", "exclusive", "free delivery", "extra off",
    "up to", "starting from", "book now", "hurry", "last chance", "grab",
    "free for lifetime", "no cost emi", "zero interest", "flat off",
    "minimum off", "max discount", "redeem now", "coupon", "voucher",
    "promo code", "gift card", "reward points", "loyalty", "membership offer",
    "new launch", "just started", "expression of interest", "eoi"
]

PROMO_KEYWORDS_WEAK = [
    "free", "save", "special", "limited", "exclusive offer", "bonus",
    "cash back", "discounts", "offers", "deals", "promotion",
    "per person", "tap below", "swipe up", "view the itinerary",
    "book your trip", "reply stop to unsubscribe",
]

PAYMENT_KEYWORDS = [
    "payment", "invoice", "bill", "refund", "amount due", "total due",
    "payment received", "payment failed", "transaction", "owed", "send money",
    "request money", "upi", "pay now", "due date", "overdue", "pending payment",
    "payment reminder", "auto-debit", "emi", "installment", "charge",
    "fee", "subscription", "renewal", "billing", "statement"
]

EVENT_KEYWORDS_STRONG = [
    "meeting at", "call at", "pickup at", "pick up at", "appointment",
    "scheduled for", "event on", "join us at", "starts at", "begins at",
    "gate", "venue", "location", "address",
    "conference", "webinar", "seminar", "workshop", "trip", "departure",
    "return time", "field trip", "departing", "arriving", "pick up",
    "drop off", "check-in", "check in", "check-out", "checkout",
    "flight", "train", "bus", "cab", "driver", "hotel booking",
    "reservation", "booking confirmed", "appointment confirmed",
    "meeting link", "calendar invite", "save the date", "rsvp",
    "changed to", "moved to", "rescheduled to", "postponed to"
]

EVENT_KEYWORDS_WEAK = [
    "from", "to", "am", "pm", "delivery on"
]

SPAM_KEYWORDS = [
    "dial", "press 1", "press 2",
    "expression of interest", "call back", "missed call", "offers",
    "bulk", "broadcast", "forwarded as received", "fwd", "chain message",
    "pass this to", "share with 10 people", "send to all", "viral",
    "everyone must know", "don\'t ignore", "read carefully", "important notice",
    "government announcement", "free subscription", "no obligation",
    "act fast", "limited slots", "first come first serve"
]

FORWARD_INDICATORS = [
    "forwarded", "fwd", "sharing this", "as received", "pass this on",
    "send this to", "chain", "viral", "trending", "everyone should know",
    "please share", "don\'t delete", "read and forward", "forward to",
    "circulating", "widely shared", "being shared", "going around"
]

GREETING_KEYWORDS = [
    "happy birthday", "happy new year", "happy diwali", "happy holi",
    "good morning", "good night", "congratulations", "well done",
    "best wishes", "greetings", "thank you", "thanks", "welcome",
    "wishing you", "blessings", "prayers", "get well soon", "happy anniversary",
    "happy holidays", "season\'s greetings", "good luck", "proud of you",
    "miss you", "thinking of you", "take care", "stay safe"
]


def _count_matches(text_lower: str, keywords: List[str]) -> List[str]:
    return [kw for kw in keywords if kw in text_lower]


def compute_content_signals(text: str, forwarded_count: Any) -> Dict[str, Any]:
    """Compute keyword-based signals from message text."""
    text_lower = (text or "").lower()
    if not text_lower:
        return {
            "has_urgent": False, "has_scam": False, "has_promo": False,
            "has_payment": False, "has_event": False, "has_spam": False,
            "has_greeting": False, "has_forward": False,
            "fake_urgency": False, "forwarded_count": 0,
            "content_length": 0,
        }

    fake_urgency = any(p in text_lower for p in FAKE_URGENCY_PATTERNS)

    urgent_hits = _count_matches(text_lower, URGENT_KEYWORDS)
    scam_hits = _count_matches(text_lower, SCAM_KEYWORDS)
    promo_strong = _count_matches(text_lower, PROMO_KEYWORDS_STRONG)
    promo_weak = _count_matches(text_lower, PROMO_KEYWORDS_WEAK)
    payment_hits = _count_matches(text_lower, PAYMENT_KEYWORDS)
    event_strong = _count_matches(text_lower, EVENT_KEYWORDS_STRONG)
    event_weak = _count_matches(text_lower, EVENT_KEYWORDS_WEAK)
    spam_hits = _count_matches(text_lower, SPAM_KEYWORDS)
    greeting_hits = _count_matches(text_lower, GREETING_KEYWORDS)
    forward_hits = _count_matches(text_lower, FORWARD_INDICATORS)

    # Promotion logic: strong keywords trigger immediately, weak need 2+ or context
    has_promo = len(promo_strong) > 0 or (len(promo_weak) >= 2)

    # Event logic: strong keywords trigger immediately, weak need time/date confirmation
    time_pattern = re.search(
        r"\b(\d{1,2}[:.]\d{2}\s*(am|pm|AM|PM)?)\b", text or "")
    date_pattern = re.search(
        r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", text or "")
    has_time_date = bool(time_pattern or date_pattern)
    has_event = len(event_strong) > 0 or (
        len(event_weak) > 0 and has_time_date)

    try:
        fwd = int(forwarded_count) if forwarded_count not in (
            None, "", "nan") else 0
    except (ValueError, TypeError):
        fwd = 0

    return {
        "has_urgent": len(urgent_hits) > 0,
        "urgent_keywords": urgent_hits[:5],
        "has_scam": len(scam_hits) > 0,
        "scam_keywords": scam_hits[:5],
        "has_promo": has_promo,
        "promo_keywords": (promo_strong + promo_weak)[:5],
        "has_payment": len(payment_hits) > 0,
        "payment_keywords": payment_hits[:5],
        "has_event": has_event,
        "event_keywords": (event_strong + event_weak)[:5],
        "has_spam": len(spam_hits) > 0,
        "spam_keywords": spam_hits[:5],
        "has_greeting": len(greeting_hits) > 0,
        "greeting_keywords": greeting_hits[:5],
        "has_forward": len(forward_hits) > 0 or fwd >= 3,
        "forward_keywords": forward_hits[:5],
        "fake_urgency": fake_urgency,
        "forwarded_count": fwd,
        "has_time_date": has_time_date,
        "content_length": len(text or ""),
    }


def compute_user_sender_features(data_context, message: dict) -> Dict[str, Any]:
    """Compute features about the relationship between user and sender."""
    user_id = message.get("user_id")
    sender_id = message.get("sender_user_id")
    group_id = message.get("group_id")
    business_id = message.get("business_id")

    features = {
        "group_muted": False,
        "user_group_role": "none",
        "business_opted_out": False,
        "business_has_history": False,
        "business_recent_orders": 0,
        "business_verified": False,
        "business_reports": 0,
        "sender_reply_rate": 0.0,
        "sender_open_rate": 0.0,
        "sender_dismiss_rate": 0.0,
        "sender_report_rate": 0.0,
        "sender_mute_rate": 0.0,
        "user_quiet_hours": False,
        "user_recent_load": "low",
        "user_dismissal_rate": 0.0,
        "user_report_rate": 0.0,
        "thread_message_count": 0,
    }

    # Group features
    if group_id:
        gm = data_context.group_member(group_id, user_id)
        if gm:
            mute_val = str(gm.get("mute_status", "")).lower()
            features["group_muted"] = mute_val in (
                "muted", "true", "1", "yes", "mute")
            features["user_group_role"] = gm.get("role", "member") or "member"

            counts = {}
            for col in ["messages_read", "messages_replied", "messages_dismissed", "messages_reported", "messages_muted"]:
                if col in gm:
                    try:
                        counts[col] = int(gm[col])
                    except (ValueError, TypeError):
                        counts[col] = 0

            total = sum(counts.values())
            if total > 0:
                features["sender_open_rate"] = counts.get(
                    "messages_read", 0) / total
                features["sender_reply_rate"] = counts.get(
                    "messages_replied", 0) / total
                features["sender_dismiss_rate"] = counts.get(
                    "messages_dismissed", 0) / total
                features["sender_report_rate"] = counts.get(
                    "messages_reported", 0) / total
                features["sender_mute_rate"] = counts.get(
                    "messages_muted", 0) / total

    # Business features
    if business_id:
        ubh = data_context.user_business_history_row(user_id, business_id)
        if ubh:
            features["business_has_history"] = True
            opt_out_val = str(ubh.get("opt_out", "")).lower()
            features["business_opted_out"] = opt_out_val in (
                "true", "1", "yes", "opted_out", "opt_out")
            for col in ["recent_orders", "orders", "bookings", "payments", "purchases"]:
                if col in ubh:
                    try:
                        features["business_recent_orders"] += int(ubh[col])
                    except (ValueError, TypeError):
                        pass

        biz = data_context.business(business_id)
        if biz:
            verified_val = str(biz.get("verified", "")).lower()
            features["business_verified"] = verified_val in (
                "true", "1", "yes", "verified")
            if "reports" in biz:
                try:
                    features["business_reports"] = int(biz["reports"])
                except (ValueError, TypeError):
                    pass
            if "account_age_days" in biz:
                try:
                    features["business_account_age_days"] = int(
                        biz["account_age_days"])
                except (ValueError, TypeError):
                    pass

    # User features
    user = data_context.user(user_id)
    if user:
        qh = str(user.get("quiet_hours", "")).lower()
        features["user_quiet_hours"] = qh in ("true", "1", "yes", "active")

        user_counts = {}
        for col in ["recent_opens", "recent_replies", "recent_dismissals", "recent_reports", "recent_mutes"]:
            if col in user:
                try:
                    user_counts[col] = int(user[col])
                except (ValueError, TypeError):
                    user_counts[col] = 0

        total_user = sum(user_counts.values())
        if total_user > 0:
            features["user_dismissal_rate"] = user_counts.get(
                "recent_dismissals", 0) / total_user
            features["user_report_rate"] = user_counts.get(
                "recent_reports", 0) / total_user

    # Daily summary
    daily = data_context.daily_summary(user_id)
    if daily:
        try:
            count = int(daily.get("notification_count", 0))
            if count > 50:
                features["user_recent_load"] = "very_high"
            elif count > 30:
                features["user_recent_load"] = "high"
            elif count > 15:
                features["user_recent_load"] = "medium"
            else:
                features["user_recent_load"] = "low"
        except (ValueError, TypeError):
            pass

    # Historical message analysis for this specific sender/thread
    history = data_context.history_for_user(user_id)
    if not history.empty:
        thread_history = pd.DataFrame()
        if sender_id and "sender_user_id" in history.columns:
            thread_history = history[history["sender_user_id"] == sender_id]
        elif business_id and "business_id" in history.columns:
            thread_history = history[history["business_id"] == business_id]
        elif group_id and "group_id" in history.columns:
            thread_history = history[history["group_id"] == group_id]

        if not thread_history.empty:
            features["thread_message_count"] = len(thread_history)
            events_list = []
            for msg_id in thread_history["message_id"].tolist():
                ev = data_context.events_for_message(msg_id)
                if not ev.empty and "event_type" in ev.columns:
                    events_list.extend(ev["event_type"].tolist())

            if events_list:
                total_events = len(events_list)
                features["sender_open_rate"] = events_list.count(
                    "opened") / total_events
                features["sender_reply_rate"] = events_list.count(
                    "replied") / total_events
                features["sender_dismiss_rate"] = events_list.count(
                    "dismissed") / total_events
                features["sender_report_rate"] = events_list.count(
                    "reported") / total_events
                features["sender_mute_rate"] = events_list.count(
                    "muted") / total_events

    return features


def compute_evidence_summary(evidence: list) -> Dict[str, Any]:
    """Summarize evidence into actionable signals."""
    if not evidence:
        return {
            "evidence_suggests_mute": False,
            "evidence_suggests_notify": False,
            "top_evidence_action": "none",
            "evidence_open_rate": 0.0,
            "evidence_dismiss_rate": 0.0,
            "evidence_report_rate": 0.0,
            "evidence_count": 0,
        }

    event_counts = {"opened": 0, "replied": 0,
                    "dismissed": 0, "muted": 0, "reported": 0}
    for e in evidence:
        for ev in e.get("events", []):
            if ev in event_counts:
                event_counts[ev] += 1

    total = sum(event_counts.values())
    if total == 0:
        return {
            "evidence_suggests_mute": False,
            "evidence_suggests_notify": False,
            "top_evidence_action": "none",
            "evidence_open_rate": 0.0,
            "evidence_dismiss_rate": 0.0,
            "evidence_report_rate": 0.0,
            "evidence_count": len(evidence),
        }

    dominant = max(event_counts, key=lambda k: event_counts[k])
    dismiss_mute_report = event_counts["dismissed"] + \
        event_counts["muted"] + event_counts["reported"]
    open_reply = event_counts["opened"] + event_counts["replied"]

    return {
        "evidence_suggests_mute": dismiss_mute_report > open_reply,
        "evidence_suggests_notify": open_reply > dismiss_mute_report,
        "top_evidence_action": dominant,
        "evidence_open_rate": round(event_counts["opened"] / total, 2),
        "evidence_dismiss_rate": round(event_counts["dismissed"] / total, 2),
        "evidence_report_rate": round(event_counts["reported"] / total, 2),
        "evidence_count": len(evidence),
    }
