"""
Loads every CSV in dataset/ into pandas DataFrames, validates them against
the schema the pipeline relies on, and exposes lookup helpers.

Why this file changed: the pipeline used to read column names that do not
exist in the real data (mute_status, event_type, opt_out, ...), so features
silently stayed at defaults. Now:
  1. EXPECTED_COLUMNS is checked at load time and any mismatch is printed
     loudly (a schema bug can never be silent again).
  2. message_events.csv is converted from wide flags (message_opened, ...)
     into one row per event with an `event_type` column.
  3. Derived fields (`opted_out`, `domain_matches_official`) are attached to
     rows here, so they also reach the LLM prompt automatically.
  4. daily_summary() aggregates ALL days per user, not just the first row.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import DATASET_DIR

EXPECTED_COLUMNS = {
    "messages.csv": ["message_id", "user_id", "conversation_type", "group_id", "business_id",
                     "sender_user_id", "created_at", "message_text", "media_type", "media_id",
                     "forwarded_count"],
    "users.csv": ["user_id", "do_not_disturb_window", "messages_opened_30d", "messages_replied_30d",
                  "notifications_dismissed_30d", "messages_reported_30d"],
    "group_members.csv": ["group_id", "user_id", "role", "messages_read_30d", "replies_sent_30d",
                          "notifications_dismissed_30d", "group_muted_by_user"],
    "business_accounts.csv": ["business_id", "verified", "official_domain", "domain_used_by_sender",
                              "account_age_days", "user_reports_30d", "domain_used_by_sender_age_days"],
    "user_business_history.csv": ["user_id", "business_id", "allows_promotions",
                                  "promotions_opted_out_at", "activity_count_180d",
                                  "messages_opened_30d", "messages_dismissed_30d",
                                  "messages_replied_30d"],
    "message_history.csv": ["message_id", "user_id", "message_text", "created_at"],
    "message_events.csv": ["user_id", "message_id", "message_opened", "message_replied",
                           "notification_dismissed", "muted_after_message", "message_reported"],
    "daily_notification_summary.csv": ["user_id", "date", "notifications_sent",
                                       "notifications_dismissed"],
    "images.csv": ["image_id", "file_path"],
    "voice_notes.csv": ["voice_note_id", "file_path"],
}

# wide flag column in message_events.csv -> event_type label
_EVENT_FLAGS = {
    "message_opened": "opened",
    "message_replied": "replied",
    "notification_dismissed": "dismissed",
    "muted_after_message": "muted",
    "message_reported": "reported",
}


def _read_csv(name: str) -> pd.DataFrame:
    path = DATASET_DIR / name
    if not path.exists():
        print(f"[data_loader] WARNING: {name} not found at {path}, using empty frame.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS.get(name, []) if c not in df.columns]
    if missing:
        print(f"[data_loader] SCHEMA WARNING: {name} is missing expected columns {missing}. "
              f"Features that depend on them will fall back to defaults. "
              f"Actual columns: {list(df.columns)}")
    return df


def _events_long(wide: pd.DataFrame) -> pd.DataFrame:
    """One row per (user, message, event_type) for every flag that equals 1."""
    cols = ["user_id", "message_id", "event_type", "reaction_time_minutes"]
    if wide.empty:
        return pd.DataFrame(columns=cols)
    parts = []
    for flag, label in _EVENT_FLAGS.items():
        if flag in wide.columns:
            sub = wide[pd.to_numeric(wide[flag], errors="coerce").fillna(0) > 0].copy()
            sub["event_type"] = label
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    if "reaction_time_minutes" not in out.columns:
        out["reaction_time_minutes"] = None
    return out[cols]


def _norm_domain(value) -> str:
    return str(value or "").strip().lower().removeprefix("www.")


@dataclass
class DataContext:
    messages: pd.DataFrame
    sample_messages: pd.DataFrame
    users: pd.DataFrame
    groups: pd.DataFrame
    group_members: pd.DataFrame
    business_accounts: pd.DataFrame
    user_business_history: pd.DataFrame
    message_history: pd.DataFrame
    message_events: pd.DataFrame  # long format: one row per event_type
    images: pd.DataFrame
    voice_notes: pd.DataFrame
    daily_notification_summary: pd.DataFrame

    # ---- generic lookups -------------------------------------------------
    def row_by(self, df: pd.DataFrame, id_col: str, id_val) -> Optional[dict]:
        if df.empty or id_col not in df.columns or id_val in (None, "", "nan") \
                or (isinstance(id_val, float) and id_val != id_val):
            return None
        match = df[df[id_col] == id_val]
        return None if match.empty else match.iloc[0].to_dict()

    def rows_by(self, df: pd.DataFrame, id_col: str, id_val) -> pd.DataFrame:
        if df.empty or id_col not in df.columns or id_val in (None, "", "nan") \
                or (isinstance(id_val, float) and id_val != id_val):
            return pd.DataFrame()
        return df[df[id_col] == id_val]

    def user(self, user_id):
        return self.row_by(self.users, "user_id", user_id)

    def group(self, group_id):
        return self.row_by(self.groups, "group_id", group_id)

    def group_member(self, group_id, user_id):
        rows = self.rows_by(self.group_members, "group_id", group_id)
        if rows.empty or "user_id" not in rows.columns:
            return None
        match = rows[rows["user_id"] == user_id]
        return None if match.empty else match.iloc[0].to_dict()

    def business(self, business_id):
        row = self.row_by(self.business_accounts, "business_id", business_id)
        if row is not None:
            used = _norm_domain(row.get("domain_used_by_sender"))
            official = _norm_domain(row.get("official_domain"))
            # Derived phishing signal: sender's domain differs from the brand's real domain.
            row["domain_matches_official"] = bool(used and official and used == official)
        return row

    def user_business_history_row(self, user_id, business_id):
        rows = self.rows_by(self.user_business_history, "user_id", user_id)
        if rows.empty or "business_id" not in rows.columns:
            return None
        match = rows[rows["business_id"] == business_id]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        row["opted_out"] = bool(pd.notna(row.get("promotions_opted_out_at"))
                                and str(row.get("promotions_opted_out_at")).strip())
        return row

    def daily_summary(self, user_id):
        """Aggregate over all observed days (the old version returned only the first day)."""
        rows = self.rows_by(self.daily_notification_summary, "user_id", user_id)
        if rows.empty or "notifications_sent" not in rows.columns:
            return None
        sent = pd.to_numeric(rows["notifications_sent"], errors="coerce").fillna(0)
        dism = pd.to_numeric(rows.get("notifications_dismissed", 0), errors="coerce").fillna(0)
        return {
            "days_observed": int(len(rows)),
            "avg_notifications_per_day": round(float(sent.mean()), 2),
            "notification_dismiss_rate": round(float(dism.sum() / sent.sum()), 2) if sent.sum() else 0.0,
        }

    def image_row(self, media_id):
        return self.row_by(self.images, "image_id", media_id) or self.row_by(
            self.images, "media_id", media_id)

    def voice_row(self, media_id):
        return self.row_by(self.voice_notes, "voice_note_id", media_id) or self.row_by(
            self.voice_notes, "media_id", media_id)

    def history_for_user(self, user_id) -> pd.DataFrame:
        return self.rows_by(self.message_history, "user_id", user_id)

    def events_for_message(self, message_id) -> pd.DataFrame:
        return self.rows_by(self.message_events, "message_id", message_id)


def load_data() -> DataContext:
    return DataContext(
        messages=_read_csv("messages.csv"),
        sample_messages=_read_csv("sample_messages.csv"),
        users=_read_csv("users.csv"),
        groups=_read_csv("groups.csv"),
        group_members=_read_csv("group_members.csv"),
        business_accounts=_read_csv("business_accounts.csv"),
        user_business_history=_read_csv("user_business_history.csv"),
        message_history=_read_csv("message_history.csv"),
        message_events=_events_long(_read_csv("message_events.csv")),
        images=_read_csv("images.csv"),
        voice_notes=_read_csv("voice_notes.csv"),
        daily_notification_summary=_read_csv("daily_notification_summary.csv"),
    )