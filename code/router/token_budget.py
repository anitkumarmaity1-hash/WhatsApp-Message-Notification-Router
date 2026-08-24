"""
Per-model daily token-budget tracker for Groq's "tokens per day" (TPD)
rate limits.

Why this exists: the original pipeline found out it was out of tokens the
same way a blind driver finds a wall — by hitting a 429 from Groq after
already committing to the 70B model for that call, three retries deep. This
module tracks *estimated* usage as calls are planned, and *actual* usage
(from the API's own `usage` field) as calls complete, so the pipeline can
pick the right model BEFORE spending tokens it doesn't have.

State is persisted to disk (TOKEN_BUDGET_STATE_PATH) because Groq's TPD
limit resets once every 24h server-side, not once per `python main.py`
invocation — without persistence, a second run on the same day would
"forget" that the budget was already half-spent.
"""
import json
import time
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Optional

from config import (
    GROQ_TPD_LIMITS,
    TOKEN_BUDGET_SAFETY_MARGIN,
    TOKEN_BUDGET_STATE_PATH,
)

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None


def estimate_tokens(text: str) -> int:
    """Best-effort token estimate. Uses tiktoken's cl100k_base encoding
    (close enough to Llama's tokenizer for budgeting purposes) if available,
    otherwise falls back to the standard ~4-chars-per-token heuristic."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


class TokenBudgetTracker:
    """Tracks estimated + actual token spend per model, per calendar day,
    persisted to a small JSON file so it survives across process runs."""

    def __init__(self, state_path: Path = TOKEN_BUDGET_STATE_PATH,
                 limits: Optional[dict] = None,
                 safety_margin: float = TOKEN_BUDGET_SAFETY_MARGIN):
        self._path = Path(state_path)
        self._limits = limits or GROQ_TPD_LIMITS
        self._safety_margin = safety_margin
        self._lock = Lock()
        self._state = self._load()

    # -- persistence ---------------------------------------------------
    def _today(self) -> str:
        return date.today().isoformat()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if data.get("date") == self._today():
                    return data
            except Exception:
                pass
        return {"date": self._today(), "usage": {}}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._state), encoding="utf-8")
        except Exception as e:
            print(f"[token_budget] WARNING: could not persist state: {e}")

    def _roll_if_new_day(self) -> None:
        if self._state.get("date") != self._today():
            self._state = {"date": self._today(), "usage": {}}
            self._save()

    # -- reads -----------------------------------------------------------
    def used(self, model: str) -> int:
        with self._lock:
            self._roll_if_new_day()
            return int(self._state["usage"].get(model, 0))

    def limit(self, model: str) -> float:
        return float(self._limits.get(model, 0)) or float("inf")

    def remaining(self, model: str) -> float:
        return self.limit(model) - self.used(model)

    def has_headroom(self, model: str, estimated_tokens: int) -> bool:
        """True if spending `estimated_tokens` on `model` keeps it under
        the safety-margin fraction of its daily budget."""
        limit = self.limit(model)
        if limit == float("inf"):
            return True
        projected = self.used(model) + estimated_tokens
        return projected <= limit * self._safety_margin

    # -- writes ------------------------------------------------------------
    def record(self, model: str, tokens: int) -> None:
        if tokens <= 0:
            return
        with self._lock:
            self._roll_if_new_day()
            self._state["usage"][model] = int(
                self._state["usage"].get(model, 0)) + int(tokens)
            self._save()

    def mark_exhausted(self, model: str) -> None:
        """Called when the API itself returns a 429 TPD error — snaps this
        model's recorded usage up to its full limit so nothing else keeps
        trying it for the rest of the day, even if our own estimate was
        off (e.g. Groq's tokenizer counts differ slightly from ours)."""
        limit = self.limit(model)
        if limit == float("inf"):
            return
        with self._lock:
            self._roll_if_new_day()
            self._state["usage"][model] = int(limit)
            self._save()

    # -- decision helper -----------------------------------------------
    def choose_model(self, primary: str, fallback: str, estimated_tokens: int) -> str:
        """Pick which model a call should use, BEFORE making it."""
        if self.has_headroom(primary, estimated_tokens):
            return primary
        print(
            f"[token_budget] {primary} projected to exceed "
            f"{self._safety_margin:.0%} of its daily budget "
            f"({self.used(primary)}+{estimated_tokens}/{self.limit(primary)} tokens) "
            f"-> switching to {fallback}"
        )
        return fallback

    def status_line(self) -> str:
        with self._lock:
            self._roll_if_new_day()
            parts = []
            for model, limit in self._limits.items():
                used = self._state["usage"].get(model, 0)
                parts.append(f"{model}={used}/{limit}")
            return ", ".join(parts)


# Module-level singleton so every caller shares one view of today's spend.
_tracker: Optional[TokenBudgetTracker] = None


def get_tracker() -> TokenBudgetTracker:
    global _tracker
    if _tracker is None:
        _tracker = TokenBudgetTracker()
    return _tracker
