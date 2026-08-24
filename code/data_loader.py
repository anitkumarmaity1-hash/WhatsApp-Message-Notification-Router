"""
Loads every CSV in dataset/ into pandas DataFrames and exposes fast,
schema-tolerant lookup helpers so the rest of the pipeline never
re-reads disk and never hard-codes column names it hasn't verified.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from config import DATASET_DIR


def _read_csv(name: str) -> pd.DataFrame:
    path = DATASET_DIR / name
    if not path.exists():
        print(
            f"[data_loader] WARNING: {name} not found at {path}, using empty frame.")
        return pd.DataFrame()
    return pd.read_csv(path)


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
    message_events: pd.DataFrame
    images: pd.DataFrame
    voice_notes: pd.DataFrame
    daily_notification_summary: pd.DataFrame

    # ---- generic lookups -------------------------------------------------
    def row_by(self, df: pd.DataFrame, id_col: str, id_val) -> Optional[dict]:
        if df.empty or id_col not in df.columns or id_val in (None, "", "nan"):
            return None
        match = df[df[id_col] == id_val]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def rows_by(self, df: pd.DataFrame, id_col: str, id_val) -> pd.DataFrame:
        if df.empty or id_col not in df.columns or id_val in (None, "", "nan"):
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
        return self.row_by(self.business_accounts, "business_id", business_id)

    def user_business_history_row(self, user_id, business_id):
        rows = self.rows_by(self.user_business_history, "user_id", user_id)
        if rows.empty or "business_id" not in rows.columns:
            return None
        match = rows[rows["business_id"] == business_id]
        return None if match.empty else match.iloc[0].to_dict()

    def daily_summary(self, user_id):
        return self.row_by(self.daily_notification_summary, "user_id", user_id)

    def image_row(self, media_id):
        return self.row_by(self.images, "media_id", media_id) or self.row_by(
            self.images, "image_id", media_id
        )

    def voice_row(self, media_id):
        return self.row_by(self.voice_notes, "media_id", media_id) or self.row_by(
            self.voice_notes, "voice_note_id", media_id
        )

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
        message_events=_read_csv("message_events.csv"),
        images=_read_csv("images.csv"),
        voice_notes=_read_csv("voice_notes.csv"),
        daily_notification_summary=_read_csv("daily_notification_summary.csv"),
    )
