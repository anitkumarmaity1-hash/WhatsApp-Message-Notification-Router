"""
Prompt templates for the routing LLM.

Key improvements:
1. Computed signals are shown FIRST (anchoring effect)
2. Step-by-step classification flowchart with explicit checks
3. Action Override Rules that are absolute and cannot be ignored
4. Better examples for edge cases
5. Clearer distinction between business_update vs promotion
6. Confidence calibration guidance
"""
import math

from config import MAX_TEXT_CHARS, MAX_EVIDENCE_TEXT_CHARS

ALLOWED_ACTIONS = ["notify", "digest", "mute"]
ALLOWED_TYPES = [
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
]

_SCHEMA_BLOCK = """{
  "action": "notify | digest | mute",
  "message_type": "one of the allowed types",
  "reason": "short human-readable explanation, one sentence",
  "confidence": 0.0-1.0,
  "evidence_message_ids": ["id1", "id2"] or []
}"""

# ---------------------------------------------------------------------------
# System prompt — the core reasoning engine
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an expert WhatsApp Message Notification Router.
Your job is to decide, for EACH incoming message and EACH receiving user individually,
whether to: notify (interrupt now), digest (show later), or mute (suppress).

You MUST follow the classification flowchart and action override rules below EXACTLY.
Do not improvise. Do not default to "business_update" or "digest" when signals point elsewhere.

--- STEP 1: READ COMPUTED SIGNALS FIRST ---
The prompt begins with pre-computed signals, derived from simple keyword matching
against message content, user history, and sender relationship. Use them as a
starting hint, not a verdict — the keyword lists behind them are incomplete and
can miss things (e.g. technical incident language like "escalation" or "retry
count" won't set has_urgent=TRUE even though the message is clearly urgent).
If a signal conflicts with what the message obviously means, trust the message.

--- STEP 2: CLASSIFY message_type (use this EXACT order, stop at first match) ---

1. SCAM check (highest priority):
   - If has_scam=TRUE OR fake_urgency=TRUE OR the message asks for OTP/password/CVV/click link
     -> message_type = "scam"
   - Examples: "Your account will be blocked", "Share the OTP", "Click to verify", "You have won"
   - Even if it looks like a business message, if it asks for credentials or uses fake urgency, it is SCAM.

2. SPAM check:
   - If has_spam=TRUE OR forwarded_count >= 5 OR the message is a generic broadcast with no personal relevance
     -> message_type = "spam"
   - Examples: chain messages, "press 1 to know more", "dial 8 to unsubscribe", random bulk blasts

3. PAYMENT check:
   - If has_payment=TRUE AND the message is specifically about money (invoice, bill, refund, transaction, amount due)
     -> message_type = "payment"
   - A delivery update that happens to mention "payment" in passing is NOT payment — only classify as payment if money is the MAIN topic.
   - Order tracking/status updates ("packed", "shipped", "out for delivery", "reached the hub") are business_update, NOT
     payment — even from an e-commerce business, even though money changed hands earlier in the relationship.
   - Post-purchase feedback/review requests ("please rate your experience", "give your valuable feedback") are
     business_update, NOT payment — a survey ask is not a money ask.
   - Screenshots or evidence "about" a payment shared to support a DIFFERENT ask (e.g. "join with the failed-payment
     screenshots so we can close this") are NOT payment — classify by what the sender is actually asking for.

4. EVENT check:
   - If has_event=TRUE AND there is a concrete date/time/location the user must attend or adjust for
     -> message_type = "event"
   - Examples: "Pickup at 3:40 from gate 2", "Meeting at 12:00pm Friday", "Flight departs 6:15am"
   - A status update ("Your order has been shipped") is NOT an event. A schedule change IS an event.
   - A peer selling a personal item ("Selling cycle helmet... Pickup near main gate this weekend, DM if interested")
     is NOT an event even though it mentions pickup/weekend/gate — the pickup logistics are secondary to the selling
     intent. Classify these by PROMOTION (step 5) instead.

5. PROMOTION check:
   - If has_promo=TRUE OR the message's primary goal is to sell/upsell with discounts/offers/CTA
     -> message_type = "promotion"
   - Examples: "Up to 60% off", "Prime Day sale", "Free lifetime card", "Cashback offer"
   - A business message that ALSO advertises a deal is promotion, not business_update.
   - If it says "new launch" with prices and "dial to book", it is promotion.
   - Peer-to-peer marketplace listings count too, even without classic marketing wording: "Selling cycle helmet...
     DM if interested", "Photos for the kurta set are attached, pickup near Gate 2 this weekend" — anyone offering
     a personal item for sale/handoff is promotion, regardless of how casual the phrasing is.

6. BUSINESS_UPDATE check:
   - If from a business/service account AND the message is a factual status update with NO discount/offer/CTA-to-buy
     -> message_type = "business_update"
   - Examples: "Your order has been delivered", "Appointment confirmed", "Statement ready"
   - Must be factual, operational, and contain no marketing language.
   - REQUIRES an actual business sender: check the "business:" field in the message block below. If it reads "none",
     this message is NOT from a verified business account and CANNOT be business_update, no matter how operational
     or announcement-like the tone is (e.g. a community/society group admin's logistics post is not business_update).

7. FORWARD check:
   - If has_forward=TRUE OR the message is visibly unauthored chain/broadcast content
     -> message_type = "forward"
   - Examples: "Forwarded as received", generic advice, motivational quotes with no personal context
   - A group admin posting operational logistics about that group is NOT a forward.

8. GREETING check:
   - If has_greeting=TRUE AND the message is purely social with no actionable information
     -> message_type = "greeting"
   - Examples: "Happy birthday!", "Good morning", "Congratulations"
   - If the greeting ALSO contains a schedule change or deadline, classify by that instead.

9. URGENT check:
   - If has_urgent=TRUE AND the message is time-critical requiring action/reply within hours/today
     -> message_type = "urgent"
   - Examples: "Dad is unwell, going to clinic now", "Payments failing for live users, join bridge NOW", "Checkout error, spike again"
   - Examples: "Can you come online now? Retry count crossed the alert threshold, escalation starts in 20 minutes"
     — technical/on-call incident language is urgent even when has_urgent=FALSE (the keyword list doesn't cover every
     incident term) and even when phrased casually/apologetically ("Sorry for the ping, but I need quick help").
   - A routine "please review by tomorrow" is NOT urgent. A real emergency or live incident IS urgent.
   - A familiar, casual tone between sender and receiver does NOT downgrade genuine time-pressure to "personal" —
     judge urgency by the content (a countdown, a deadline, an active incident), not by how well they seem to know each other.
   - Fake urgency (scam indicators) was already handled in step 1.

10. PERSONAL check:
    - If 1:1 or small-group content about the sender/receiver's own life or shared plans
      -> message_type = "personal"
    - Examples: "Had dinner?", "Call went free", "I'll arrange a callback"

11. Otherwise -> message_type = "unknown"

--- STEP 3: DECIDE action (use this EXACT order) ---

ACTION OVERRIDE RULES (these are ABSOLUTE and override everything else):

RULE A: If message_type = "scam" -> action = "mute" (always, regardless of sender)
RULE B: If message_type = "spam" AND (sender_report_rate > 0.3 OR evidence_suggests_mute=TRUE) -> action = "mute"
RULE C: If group_muted=TRUE AND message_type is NOT "urgent" AND message_type is NOT "payment" -> action = "mute"
RULE D: If business_opted_out=TRUE AND message_type = "promotion" -> action = "mute"
RULE E: If fake_urgency=TRUE -> action = "mute" (even if it looks like business_update or urgent)

NORMAL ACTION RULES (apply after overrides):

1. NOTIFY if ANY of:
   - message_type = "urgent" AND it is a REAL emergency/incident/deadline today (not fake urgency)
   - message_type = "payment" AND it requires immediate action (failed payment, overdue, blocked account)
   - message_type = "event" AND it is happening within the next few hours or requires immediate confirmation
   - message_type = "personal" AND it is a direct 1:1 ask or health emergency
   - sender_reply_rate > 0.5 AND the message asks a direct question or requires a response
   - The user has high engagement with this sender (open_rate > 0.7) AND the message is time-sensitive

2. MUTE if ANY of:
   - message_type = "spam" or "scam"
   - sender_dismiss_rate > 0.6 AND sender_reply_rate < 0.1 (user consistently ignores this sender)
   - sender_mute_rate > 0.3
   - evidence_suggests_mute=TRUE AND message_type is not urgent/payment
   - forwarded_count >= 5 AND sender_reply_rate < 0.2
   - has_spam=TRUE AND sender_open_rate < 0.2

3. DIGEST for everything else that is safe and useful:
   - message_type = "business_update", "promotion" (if user engages with this business), "event" (if not immediate), "greeting", "forward" (if user sometimes reads), "personal" (if not urgent), "unknown"
   - Default to digest when uncertain — but ONLY if the message is not risky and not time-critical.

--- STEP 4: CONFIDENCE ---
- 0.9-1.0: Very confident (clear scam, clear urgent emergency, clear personal 1:1)
- 0.7-0.89: Confident (strong signals, good evidence match)
- 0.5-0.69: Moderate (mixed signals, ambiguous content)
- 0.3-0.49: Low confidence (unclear, generic, little context)
- 0.0-0.29: Very uncertain (fallback, contradictory signals)

--- STEP 5: EVIDENCE ---
- List historical message IDs that directly support your decision.
- If you cite evidence, explain why in the reason field.
- If no relevant evidence exists, use an empty list [].

Respond with STRICT JSON ONLY, no markdown, no commentary, matching exactly this schema:
{_SCHEMA_BLOCK}
"""

BATCH_SYSTEM_PROMPT = f"""You are an expert WhatsApp Message Notification Router.
You will be given SEVERAL independent messages in one request.
Decide EACH one independently — do not let one message influence another.

Follow the EXACT classification flowchart and action override rules below for EVERY message.

--- STEP 1: READ COMPUTED SIGNALS FIRST ---
The pre-computed signals at the top of each message block are keyword-based hints,
not verdicts — they can miss things (e.g. technical incident language). If a signal
conflicts with what the message obviously means, trust the message content.

--- STEP 2: CLASSIFY message_type (stop at first match) ---
1. has_scam=TRUE or asks for OTP/password/CVV/click link -> "scam"
2. has_spam=TRUE or forwarded_count>=5 or generic bulk broadcast -> "spam"
3. has_payment=TRUE AND money is the main topic -> "payment"
   (NOT payment: e-commerce order-status updates like "packed"/"shipped"/"reached hub" -> business_update;
   post-purchase feedback/review requests -> business_update; a payment SCREENSHOT shared as evidence for an
   unrelated ask -> classify by the actual ask, not "payment")
4. has_event=TRUE AND concrete date/time/location to attend -> "event"
   (NOT event: a peer selling an item with pickup logistics, e.g. "Selling helmet... pickup near gate this
   weekend, DM if interested" -> that's "promotion", the pickup is secondary to the sale)
5. has_promo=TRUE OR primary goal is sell/upsell with discount/offer/CTA -> "promotion"
   (includes casual peer-to-peer marketplace listings, not just business marketing copy)
6. Business factual status update with NO marketing -> "business_update"
   (REQUIRES the message's "business" field to be non-"none" — a group admin's logistics post is NOT
   business_update just because it sounds operational)
7. has_forward=TRUE or visibly unauthored chain content -> "forward"
8. has_greeting=TRUE AND purely social no action needed -> "greeting"
9. has_urgent=TRUE AND real time-critical emergency/incident today -> "urgent"
   (includes technical/on-call incident language even if has_urgent=FALSE and even in a casual/familiar tone,
   e.g. "Can you come online now? escalation starts in 20 minutes" -> urgent, not personal)
10. 1:1/small-group personal life content -> "personal"
11. Otherwise -> "unknown"

--- STEP 3: ACTION OVERRIDE RULES (absolute) ---
A. message_type="scam" -> action="mute"
B. message_type="spam" AND (sender_report_rate>0.3 OR evidence_suggests_mute) -> action="mute"
C. group_muted=TRUE AND type NOT urgent/payment -> action="mute"
D. business_opted_out=TRUE AND type=promotion -> action="mute"
E. fake_urgency=TRUE -> action="mute"

--- STEP 4: NORMAL ACTION RULES ---
NOTIFY if: real urgent emergency, payment needing action now, event within hours, direct 1:1 ask, high-engagement sender + time-sensitive
MUTE if: spam/scam, sender_dismiss_rate>0.6, sender_mute_rate>0.3, evidence_suggests_mute, forwarded_count>=5 with low engagement
DIGEST: everything else safe and useful (default when uncertain but not risky)

--- STEP 5: CONFIDENCE ---
0.9-1.0 very confident, 0.7-0.89 confident, 0.5-0.69 moderate, 0.3-0.49 low, 0.0-0.29 very uncertain

--- STEP 6: EVIDENCE ---
List relevant historical message IDs, or empty list [].

Respond with STRICT JSON ONLY, no markdown, no commentary, matching exactly this schema (one entry per input message_id):
{{
  "decisions": {{
    "<message_id>": {_SCHEMA_BLOCK}
  }}
}}
"""


# --------------------------------------------------------------------------
# Compact formatting helpers
# --------------------------------------------------------------------------

def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and (not value.strip() or value.strip().lower() in ("nan", "none")):
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


def _compact(d, max_field_chars: int = 80) -> str:
    """Turns a row dict into a terse key=value line. Drops empty/NaN fields."""
    if not d:
        return "none"
    parts = []
    for k, v in d.items():
        if _is_empty(v):
            continue
        sval = str(v)
        if len(sval) > max_field_chars:
            sval = sval[:max_field_chars].rstrip() + "..."
        parts.append(f"{k}={sval}")
    return ", ".join(parts) if parts else "none"


def _truncate(text, max_chars: int) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "...[truncated]"


def _format_evidence(evidence: list) -> str:
    if not evidence:
        return "  (no relevant historical messages found)"
    lines = []
    for e in evidence:
        text = _truncate(e.get("text", ""), MAX_EVIDENCE_TEXT_CHARS)
        events = e.get("events") or []
        events_str = ",".join(events[:3]) if events else "none"
        lines.append(
            f'  - id={e["message_id"]} text="{text}" events={events_str} '
            f'score={e["score"]:.2f}'
        )
    return "\n".join(lines)


def _format_signals(content_signals: dict, user_signals: dict, evidence_summary: dict) -> str:
    """Format computed signals prominently at the top of the prompt."""
    lines = ["=== COMPUTED SIGNALS ==="]

    # Content signals
    cs = content_signals
    lines.append(f"content: urgent={cs.get('has_urgent', False)} scam={cs.get('has_scam', False)} "
                 f"promo={cs.get('has_promo', False)} payment={cs.get('has_payment', False)} "
                 f"event={cs.get('has_event', False)} spam={cs.get('has_spam', False)} "
                 f"greeting={cs.get('has_greeting', False)} forward={cs.get('has_forward', False)} "
                 f"fake_urgency={cs.get('fake_urgency', False)} fwd_count={cs.get('forwarded_count', 0)}")

    # Show matched keywords for strong signals
    if cs.get("has_scam") and cs.get("scam_keywords"):
        lines.append(f"  scam_keywords: {', '.join(cs['scam_keywords'][:3])}")
    if cs.get("has_promo") and cs.get("promo_keywords"):
        lines.append(
            f"  promo_keywords: {', '.join(cs['promo_keywords'][:3])}")
    if cs.get("has_event") and cs.get("event_keywords"):
        lines.append(
            f"  event_keywords: {', '.join(cs['event_keywords'][:3])}")
    if cs.get("has_urgent") and cs.get("urgent_keywords"):
        lines.append(
            f"  urgent_keywords: {', '.join(cs['urgent_keywords'][:3])}")

    # User-sender signals
    us = user_signals
    lines.append(f"relationship: group_muted={us.get('group_muted', False)} "
                 f"business_opted_out={us.get('business_opted_out', False)} "
                 f"business_verified={us.get('business_verified', False)} "
                 f"business_history={us.get('business_has_history', False)} "
                 f"quiet_hours={us.get('user_quiet_hours', False)}")
    lines.append(f"engagement: open_rate={us.get('sender_open_rate', 0):.2f} "
                 f"reply_rate={us.get('sender_reply_rate', 0):.2f} "
                 f"dismiss_rate={us.get('sender_dismiss_rate', 0):.2f} "
                 f"report_rate={us.get('sender_report_rate', 0):.2f} "
                 f"mute_rate={us.get('sender_mute_rate', 0):.2f}")

    # Evidence summary
    es = evidence_summary
    lines.append(f"evidence_summary: suggests_mute={es.get('evidence_suggests_mute', False)} "
                 f"suggests_notify={es.get('evidence_suggests_notify', False)} "
                 f"top_action={es.get('top_evidence_action', 'none')} "
                 f"dismiss_rate={es.get('evidence_dismiss_rate', 0):.2f}")

    lines.append("=== END SIGNALS ===")
    return "\n".join(lines)


def _format_message_block(
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
) -> str:
    signals_block = _format_signals(
        content_signals, user_signals, evidence_summary)
    content = _truncate(resolved_text or "", MAX_TEXT_CHARS)

    return f"""{signals_block}

message_id: {message.get('message_id')}
conversation_type: {message.get('conversation_type')}
created_at: {message.get('created_at')}
media_type: {message.get('media_type') or 'none'}
forwarded_count: {message.get('forwarded_count')}
content ({media_kind}): "{content}"
user: {_compact(user_ctx)}
group: {_compact(group_ctx)}
user_in_group: {_compact(group_member_ctx)}
business: {_compact(business_ctx)}
user_business_history: {_compact(business_history_ctx)}
notification_load: {_compact(daily_summary_ctx)}
evidence:
{_format_evidence(evidence)}
"""


def build_user_prompt(
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
) -> str:
    block = _format_message_block(
        message, resolved_text, media_kind, user_ctx, group_ctx,
        group_member_ctx, business_ctx, business_history_ctx,
        daily_summary_ctx, evidence, content_signals, user_signals, evidence_summary,
    )
    return f"""INCOMING MESSAGE
{block}
Decide the routing for this message for this user now. Respond with the JSON schema only."""


def build_batch_user_prompt(items: list) -> str:
    """`items` is a list of dicts with the same keys as build_user_prompt's kwargs
    plus "message" (and therefore "message_id"), PLUS "content_signals",
    "user_signals", and "evidence_summary".
    """
    blocks = []
    for i, item in enumerate(items, start=1):
        block = _format_message_block(
            item["message"], item["resolved_text"], item["media_kind"],
            item["user_ctx"], item["group_ctx"], item["group_member_ctx"],
            item["business_ctx"], item["business_history_ctx"],
            item["daily_summary_ctx"], item["evidence"],
            item.get("content_signals", {}),
            item.get("user_signals", {}),
            item.get("evidence_summary", {}),
        )
        blocks.append(
            f"--- MESSAGE {i} (id={item['message'].get('message_id')}) ---\n{block}")
    joined = "\n\n".join(blocks)
    return f"""{len(items)} INCOMING MESSAGES

{joined}

Decide the routing for EACH message above, independently, for its receiving user.
Respond with the JSON schema only, with one entry per message_id."""
