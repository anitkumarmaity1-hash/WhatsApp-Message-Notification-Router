"""
Feature extraction: derived, interpretable signals that help the rules and the LLM.

Changes vs the previous version:
  1. Keyword matching uses WORD BOUNDARIES (regex), so "otp" no longer fires inside
     "hotpot", "sale" inside "wholesale", "bus" inside "business", "am" inside "exam".
  2. Scam detection is TIERED:
       has_scam / fake_urgency  -> HARD, high-precision patterns only (asking someone to
                                   share an OTP/CVV/PIN/bank details, prize/lottery claims,
                                   money-scheme phrases, "account will be blocked" + link/verify,
                                   lookalike hyphenated pay-links). These feed the 0.95 pre-LLM mute.
       scam_suspect             -> SOFT keywords ("security alert", "kyc", "suspended", ...).
                                   Never auto-muted; left for the LLM / business context to judge.
     Negated warnings ("Do not share your OTP with anyone") are stripped first, so genuine
     bank OTP notices are not treated as scams.
  3. User/sender features now read the REAL column names (group_muted_by_user,
     promotions_opted_out_at, do_not_disturb_window, user_reports_30d, ...), compute quiet-hours
     from the message timestamp, and add domain-mismatch / new-domain business signals.
"""
import re
from typing import Any, Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Keyword lists (unchanged content)
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
    "don't forget", "reminder", "last chance", "final notice", "overdue",
    "blocked", "suspended", "verification needed", "verify now", "act now",
    "only today", "ends today", "last day", "closing soon", "live users",
    "checkout error", "spiking again", "join the bridge",
    "eod", "end of day", "before eod", "escalation", "escalates",
    "retry count", "alert threshold", "crossed the threshold", "come online",
    "need quick help", "on-call", "paging", "page me",
]

# SOFT scam words: suspicious, but also used by legitimate banks/services.
SCAM_SUSPECT_KEYWORDS = [
    "otp", "verify your account", "click this link", "suspended", "lottery", "prize",
    "bitcoin", "act now", "limited time", "bank details", "card details", "cvv", "kyc",
    "verification required", "unusual activity", "login attempt", "free gift",
    "claim your reward", "send money to", "click here to verify", "your account is locked",
    "security alert", "unauthorized access", "update your details", "re-verify",
    "confirm identity", "validate account", "confirm your password", "scan and pay",
]

PROMO_KEYWORDS_STRONG = [
    "% off", "percent off", "discount", "sale", "cashback", "offer", "deal",
    "buy now", "shop now", "limited period", "flash sale", "mega sale",
    "prime day", "special offer", "exclusive", "free delivery", "extra off",
    "up to", "starting from", "book now", "hurry", "last chance", "grab",
    "free for lifetime", "no cost emi", "zero interest", "flat off",
    "minimum off", "max discount", "redeem now", "coupon", "voucher",
    "promo code", "gift card", "reward points", "loyalty", "membership offer",
    "new launch", "just started", "expression of interest", "eoi",
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
    "fee", "subscription", "renewal", "billing", "statement",
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
    "changed to", "moved to", "rescheduled to", "postponed to",
]
EVENT_KEYWORDS_WEAK = ["from", "to", "am", "pm", "delivery on"]
SPAM_KEYWORDS = [
    "dial", "press 1", "press 2",
    "expression of interest", "call back", "missed call", "offers",
    "bulk", "broadcast", "forwarded as received", "fwd", "chain message",
    "pass this to", "share with 10 people", "send to all", "viral",
    "everyone must know", "don't ignore", "read carefully", "important notice",
    "government announcement", "free subscription", "no obligation",
    "act fast", "limited slots", "first come first serve",
]
FORWARD_INDICATORS = [
    "forwarded", "fwd", "sharing this", "as received", "pass this on",
    "send this to", "chain", "viral", "trending", "everyone should know",
    "please share", "don't delete", "read and forward", "forward to",
    "circulating", "widely shared", "being shared", "going around",
]
GREETING_KEYWORDS = [
    "happy birthday", "happy new year", "happy diwali", "happy holi",
    "good morning", "good night", "congratulations", "well done",
    "best wishes", "greetings", "thank you", "thanks", "welcome",
    "wishing you", "blessings", "prayers", "get well soon", "happy anniversary",
    "happy holidays", "season's greetings", "good luck", "proud of you",
    "miss you", "thinking of you", "take care", "stay safe",
]


def _compile(words: List[str]) -> re.Pattern:
    """One regex per list; matches whole words/phrases only (longest alternative first)."""
    alts = "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alts})(?![a-z0-9])")


_RX = {name: _compile(words) for name, words in {
    "urgent": URGENT_KEYWORDS, "suspect": SCAM_SUSPECT_KEYWORDS,
    "promo_s": PROMO_KEYWORDS_STRONG, "promo_w": PROMO_KEYWORDS_WEAK,
    "payment": PAYMENT_KEYWORDS, "event_s": EVENT_KEYWORDS_STRONG,
    "event_w": EVENT_KEYWORDS_WEAK, "spam": SPAM_KEYWORDS,
    "forward": FORWARD_INDICATORS, "greeting": GREETING_KEYWORDS,
}.items()}


def _hits(text_lower: str, key: str) -> List[str]:
    """Ordered, de-duplicated list of whole-word matches."""
    return list(dict.fromkeys(_RX[key].findall(text_lower)))


# ---------------------------------------------------------------------------
# HARD scam patterns (high precision)
# ---------------------------------------------------------------------------
_NEGATED_WARNING = re.compile(
    r"\b(?:do not|don't|dont|never|not to)\s+(?:share|send|give|disclose|tell|reveal)\b[^.!?\n]*")
_ASK = r"(?:share|send|enter|reply(?:\s+with)?|provide|forward|tell|give|read\s+out|confirm|submit|fill|type|verify|validate)"
_SENSITIVE = (r"(?:otp|ott|cvv|(?:upi|atm|card|wallet|bank|debit|credit|your)\s+pin|"
              r"(?:your|account|login|bank|email|upi)\s+password|passcode|"
              r"(?:login|verification|security)\s+code|"
              r"\d[- ]?digit\s+(?:login\s+|verification\s+)?code|one[- ]time\s+(?:password|code)|"
              r"(?:bank|card|account|kyc)\s+details|card\s+number|aadhaar)")
_CRED_ASK = re.compile(rf"\b{_ASK}\b[^.!?\n]{{0,60}}?\b{_SENSITIVE}\b")
# Secret code mentioned together with pressure/link words (catches Hinglish: "OTP abhi batao",
# "link open karke code daal do"). Bare OTP notices with only a warning do not match.
_SECRET = re.compile(r"\b(?:otp|ott|cvv|(?:login|verification|security)\s+code|one[- ]time\s+(?:password|code))\b")
_PRESSURE = re.compile(r"\b(?:abhi|immediately|urgently|asap|jaldi|blocked?|restricted|on hold|hold|"
                       r"suspended|locked|leak(?:ed)?|link|click|before midnight)\b")
_PRIZE = re.compile(r"\byou(?:'ve| have)?\s+won\b|\blucky winner\b|\bcash prize\b|"
                    r"\bclaim your (?:prize|reward)\b|\blottery\b")
_MONEY_SCHEME = re.compile(r"\b(?:wire transfer|double your money|guaranteed returns?|"
                           r"crypto investment|transfer to this account)\b")
_THREAT = re.compile(r"\b(?:account|card|bank account|number|sim|wallet|upi|profile|access)\b[^.!?\n]{0,25}?"
                     r"\b(?:will be|has been|is|gets?)\s+(?:blocked|suspended|locked|deactivated|closed|frozen|restricted)\b")
_LINK_ASK = re.compile(r"https?://|www\.|\b(?:click|tap|open|visit)\b[^.!?\n]{0,30}\b(?:link|here|below)\b|"
                       r"\bthis link\b|\bre-?verify\b|\bverify\b|\bkyc\b|\bupdate your (?:details|kyc|pan|aadhaar)\b")
_LOOKALIKE_LINK = re.compile(r"\b(?:pay|fee|charge|clearance|confirm|verify|login|log in|sign in|update|enter)\b"
                             r"[^.!?\n]{0,60}?\b[a-z0-9]+-[a-z0-9-]+\.(?:in|com|co|net|xyz|top|link|online|info)\b")
# Text addressed to the notification router itself (prompt injection). Real chat never does this.
_INJECTION = re.compile(r"routing override|system note|router metadata|internal router|notification router|"
                        r"\baction\s*=\s*(?:notify|mute|digest)|always mark this|confidence\s*=\s*\d|"
                        r"user_priority\s*=|verified_business\s*=|ignore (?:all |any )?(?:previous|prior|above) instructions")


def _hard_scam(text_lower: str) -> Dict[str, Any]:
    t = _NEGATED_WARNING.sub(" ", text_lower)   # "do not share your OTP" is a warning, not an ask
    labels = []
    if _INJECTION.search(text_lower):
        labels.append("prompt-injection attempt")
    if _CRED_ASK.search(t) or (_SECRET.search(t) and _PRESSURE.search(t)):
        labels.append("credential/OTP request")
    if _PRIZE.search(t):
        labels.append("prize/lottery claim")
    if _MONEY_SCHEME.search(t):
        labels.append("money scheme")
    if _LOOKALIKE_LINK.search(t):
        labels.append("lookalike link")
    fake_urgency = bool(_THREAT.search(t) and (_LINK_ASK.search(t) or _CRED_ASK.search(t)))
    if fake_urgency:
        labels.append("blocked-account threat + verify/link")
    return {"labels": labels, "fake_urgency": fake_urgency}


_TIME_RX = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", re.I)
_DATE_RX = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
_MENTION_RX = re.compile(r"@\w+")


def compute_content_signals(text: str, forwarded_count: Any) -> Dict[str, Any]:
    """Whole-word keyword signals plus tiered scam detection."""
    text_lower = (text or "").lower()
    try:
        fwd = int(forwarded_count) if forwarded_count not in (None, "", "nan") else 0
    except (ValueError, TypeError):
        fwd = 0
    if not text_lower.strip():
        return {k: False for k in (
            "has_urgent", "has_scam", "scam_suspect", "has_promo", "has_payment", "has_event",
            "has_spam", "has_greeting", "has_forward", "fake_urgency", "has_time_date",
            "has_direct_mention")} | {"forwarded_count": fwd, "content_length": 0}

    hard = _hard_scam(text_lower)
    urgent, suspect = _hits(text_lower, "urgent"), _hits(text_lower, "suspect")
    promo_s, promo_w = _hits(text_lower, "promo_s"), _hits(text_lower, "promo_w")
    payment, spam = _hits(text_lower, "payment"), _hits(text_lower, "spam")
    event_s, event_w = _hits(text_lower, "event_s"), _hits(text_lower, "event_w")
    greeting, forward = _hits(text_lower, "greeting"), _hits(text_lower, "forward")

    has_time_date = bool(_TIME_RX.search(text) or _DATE_RX.search(text))
    return {
        "has_urgent": bool(urgent), "urgent_keywords": urgent[:5],
        "has_scam": bool(hard["labels"]), "scam_keywords": hard["labels"][:5],
        "scam_suspect": bool(suspect) and not hard["labels"], "scam_suspect_keywords": suspect[:5],
        "has_promo": bool(promo_s) or len(promo_w) >= 2, "promo_keywords": (promo_s + promo_w)[:5],
        "has_payment": bool(payment), "payment_keywords": payment[:5],
        "has_event": bool(event_s) or (bool(event_w) and has_time_date),
        "event_keywords": (event_s + event_w)[:5],
        "has_spam": bool(spam), "spam_keywords": spam[:5],
        "has_greeting": bool(greeting), "greeting_keywords": greeting[:5],
        "has_forward": bool(forward) or fwd >= 3, "forward_keywords": forward[:5],
        "fake_urgency": hard["fake_urgency"],
        "forwarded_count": fwd, "has_time_date": has_time_date,
        "has_direct_mention": bool(_MENTION_RX.search(text or "")),
        "content_length": len(text or ""),
    }


# ---------------------------------------------------------------------------
# User / sender / business relationship features
# ---------------------------------------------------------------------------
def _num(value, default=0.0) -> float:
    try:
        v = float(value)
        return default if v != v else v
    except (ValueError, TypeError):
        return default


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def _in_dnd(window, created_at) -> bool:
    """True if created_at falls inside a 'HH:MM-HH:MM' do-not-disturb window (may cross midnight)."""
    try:
        start_s, end_s = str(window).split("-")
        to_min = lambda s: int(s.strip()[:2]) * 60 + int(s.strip()[3:5])
        start, end = to_min(start_s), to_min(end_s)
        ts = pd.to_datetime(created_at)
        now = ts.hour * 60 + ts.minute
        return (start <= now < end) if start <= end else (now >= start or now < end)
    except Exception:
        return False


def compute_user_sender_features(data_context, message: dict) -> Dict[str, Any]:
    user_id = message.get("user_id")
    sender_id = message.get("sender_user_id")
    group_id = message.get("group_id")
    business_id = message.get("business_id")

    f: Dict[str, Any] = {
        "group_muted": False, "user_group_role": "none",
        "business_opted_out": False, "business_allows_promotions": True,
        "business_has_history": False, "business_recent_orders": 0,
        "business_verified": False, "business_reports": 0,
        "business_domain_mismatch": False, "business_new_domain": False,
        "sender_reply_rate": 0.0, "sender_open_rate": 0.0, "sender_dismiss_rate": 0.0,
        "sender_report_rate": 0.0, "sender_mute_rate": 0.0,
        "user_quiet_hours": False, "user_recent_load": "low",
        "user_dismissal_rate": 0.0, "user_report_rate": 0.0, "thread_message_count": 0,
    }

    # --- group relationship (real columns: group_muted_by_user, messages_read_30d, ...) ---
    if group_id:
        gm = data_context.group_member(group_id, user_id)
        if gm:
            f["group_muted"] = _num(gm.get("group_muted_by_user")) > 0
            f["user_group_role"] = gm.get("role") or "member"
            read, replies = _num(gm.get("messages_read_30d")), _num(gm.get("replies_sent_30d"))
            dism = _num(gm.get("notifications_dismissed_30d"))
            f["sender_open_rate"] = _ratio(read, read + dism)
            f["sender_dismiss_rate"] = _ratio(dism, read + dism)
            f["sender_reply_rate"] = min(1.0, _ratio(replies, read))

    # --- business relationship + sender authenticity ---
    if business_id:
        ubh = data_context.user_business_history_row(user_id, business_id)
        if ubh:
            f["business_has_history"] = True
            f["business_opted_out"] = bool(ubh.get("opted_out"))
            f["business_allows_promotions"] = _num(ubh.get("allows_promotions")) > 0
            f["business_recent_orders"] = int(_num(ubh.get("activity_count_180d")))
            opened, dism = _num(ubh.get("messages_opened_30d")), _num(ubh.get("messages_dismissed_30d"))
            f["sender_open_rate"] = _ratio(opened, opened + dism)
            f["sender_dismiss_rate"] = _ratio(dism, opened + dism)
            f["sender_reply_rate"] = min(1.0, _ratio(_num(ubh.get("messages_replied_30d")), opened))
        biz = data_context.business(business_id)
        if biz:
            f["business_verified"] = _num(biz.get("verified")) > 0
            f["business_reports"] = int(_num(biz.get("user_reports_30d")))
            f["business_account_age_days"] = int(_num(biz.get("account_age_days")))
            f["business_domain_age_days"] = int(_num(biz.get("domain_used_by_sender_age_days")))
            f["business_domain_mismatch"] = not biz.get("domain_matches_official", True)
            f["business_new_domain"] = 0 < f["business_domain_age_days"] < 30

    # --- user-level behaviour (real columns: *_30d) and quiet hours ---
    user = data_context.user(user_id)
    if user:
        opened, dism = _num(user.get("messages_opened_30d")), _num(user.get("notifications_dismissed_30d"))
        f["user_dismissal_rate"] = _ratio(dism, opened + dism)
        f["user_report_rate"] = _ratio(_num(user.get("messages_reported_30d")), opened)
        f["user_quiet_hours"] = _in_dnd(user.get("do_not_disturb_window"), message.get("created_at"))

    daily = data_context.daily_summary(user_id)
    if daily:
        avg = _num(daily.get("avg_notifications_per_day"))
        f["user_recent_load"] = "very_high" if avg >= 10 else "high" if avg >= 8 else "medium" if avg >= 6 else "low"
        f["user_notification_dismiss_rate"] = _num(daily.get("notification_dismiss_rate"))

    # --- per-thread history: overrides the coarse rates above when real events exist ---
    history = data_context.history_for_user(user_id)
    if not history.empty:
        thread = pd.DataFrame()
        if sender_id and "sender_user_id" in history.columns:
            thread = history[history["sender_user_id"] == sender_id]
        elif business_id and "business_id" in history.columns:
            thread = history[history["business_id"] == business_id]
        elif group_id and "group_id" in history.columns:
            thread = history[history["group_id"] == group_id]
        if not thread.empty:
            f["thread_message_count"] = len(thread)
            events: List[str] = []
            for mid in thread["message_id"].tolist():
                ev = data_context.events_for_message(mid)
                if not ev.empty:
                    events.extend(ev["event_type"].tolist())
            if events:
                n = len(events)
                for key, label in (("open", "opened"), ("reply", "replied"), ("dismiss", "dismissed"),
                                   ("report", "reported"), ("mute", "muted")):
                    f[f"sender_{key}_rate"] = events.count(label) / n
    return f


def compute_evidence_summary(evidence: list) -> Dict[str, Any]:
    """Roll retrieved historical evidence up into actionable signals."""
    empty = {"evidence_suggests_mute": False, "evidence_suggests_notify": False,
             "top_evidence_action": "none", "evidence_open_rate": 0.0,
             "evidence_dismiss_rate": 0.0, "evidence_report_rate": 0.0,
             "evidence_count": len(evidence or [])}
    if not evidence:
        return empty
    counts = {"opened": 0, "replied": 0, "dismissed": 0, "muted": 0, "reported": 0}
    for e in evidence:
        for ev in e.get("events", []):
            if ev in counts:
                counts[ev] += 1
    total = sum(counts.values())
    if total == 0:
        return empty
    negative = counts["dismissed"] + counts["muted"] + counts["reported"]
    positive = counts["opened"] + counts["replied"]
    return {
        "evidence_suggests_mute": negative > positive,
        "evidence_suggests_notify": positive > negative,
        "top_evidence_action": max(counts, key=counts.get),
        "evidence_open_rate": round(counts["opened"] / total, 2),
        "evidence_dismiss_rate": round(counts["dismissed"] / total, 2),
        "evidence_report_rate": round(counts["reported"] / total, 2),
        "evidence_count": len(evidence),
    }