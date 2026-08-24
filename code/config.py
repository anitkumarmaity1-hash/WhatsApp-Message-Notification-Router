"""
Central configuration for the Message Notification Router pipeline.
All secrets are read from environment variables (never hardcoded).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(REPO_ROOT / "dataset")))
MEDIA_DIR = DATASET_DIR / "media"
IMAGES_DIR = MEDIA_DIR / "images"
AUDIO_DIR = MEDIA_DIR / "audio"
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(CODE_DIR / ".cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATASET_DIR / "output.csv"

# --- LLM (Groq) ----------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Primary model does the real reasoning. Fallback is a much smaller/cheaper
# model with its own separate Groq daily-token bucket, used the moment the
# primary bucket is exhausted (or predicted to be) instead of burning
# retries against a 429 that cannot succeed until the quota resets.
GROQ_MODEL_PRIMARY = os.getenv(
    "GROQ_MODEL_PRIMARY", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")
GROQ_MODEL = GROQ_MODEL_PRIMARY  # kept for backwards compatibility

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# --- LLM (Ollama, local fallback) -----------------------------------------
# Final tier once BOTH Groq models are out of daily budget: a fully local,
# unlimited, $0 model via Ollama. Requires `ollama serve` running and the
# model already pulled (`ollama pull llama3.1:8b`). Quality is closer to
# GROQ_MODEL_FALLBACK (8B) than GROQ_MODEL_PRIMARY (70B) — this is a safety
# net so a run never degrades all the way to the blind rule-based default,
# not a replacement for the Groq tiers.
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

# --- Token budget tracking -------------------------------------------------
# Groq free-tier "tokens per day" (TPD) limits are tracked PER MODEL. Set
# these to whatever console.groq.com/settings/limits shows for your account.
GROQ_TPD_LIMITS = {
    GROQ_MODEL_PRIMARY: int(os.getenv("GROQ_TPD_LIMIT_PRIMARY", "100000")),
    GROQ_MODEL_FALLBACK: int(os.getenv("GROQ_TPD_LIMIT_FALLBACK", "500000")),
}
# Proactively stop using a model once estimated usage crosses this fraction
# of its daily budget, rather than finding out via a 429 mid-run.
TOKEN_BUDGET_SAFETY_MARGIN = float(
    os.getenv("TOKEN_BUDGET_SAFETY_MARGIN", "0.9"))
# Persisted so budget tracking survives across separate `python main.py`
# runs on the same calendar day (Groq's limit resets daily, not per-process).
TOKEN_BUDGET_STATE_PATH = Path(
    os.getenv("TOKEN_BUDGET_STATE_PATH", str(CACHE_DIR / "token_budget_state.json")))

# --- Prompt size controls ---------------------------------------------------
# Batching trades accuracy for token savings: asking the model to hold
# several users' contexts in mind at once and keep judgments cleanly
# separated per-message is measurably harder than one-at-a-time reasoning,
# especially for a policy this dependent on PER-USER personalization.
# Default is 1 (no batching) so quality matches the original per-message
# pipeline. Raise this only if you're re-hitting the daily token wall AND
# have verified accuracy is acceptable at higher batch sizes.
LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "1"))
# Hard caps on how much raw text gets embedded into the prompt per message.
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "400"))
MAX_EVIDENCE_TEXT_CHARS = int(os.getenv("MAX_EVIDENCE_TEXT_CHARS", "180"))

# --- ASR (faster-whisper) -------------------------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# --- OCR (EasyOCR) ---------------------------------------------------------
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
OCR_GPU = os.getenv("OCR_GPU", "false").lower() == "true"

# --- Retrieval -------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EVIDENCE_TOP_K = int(os.getenv("EVIDENCE_TOP_K", "5"))

# --- Run behaviour -----------------------------------------------------
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

if not GROQ_API_KEY:
    print("[config] WARNING: GROQ_API_KEY is not set. Set it in your .env file.")
