"""
ASR for voice notes using faster-whisper (a CTranslate2 reimplementation
of OpenAI Whisper) — the standard free, local, production-grade
speech-to-text stack; roughly 4x faster than vanilla Whisper on CPU at
the same accuracy, which is why it's widely used over the reference
implementation in real deployments.
"""
import json
import functools

from config import AUDIO_DIR, CACHE_DIR, WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

_CACHE_PATH = CACHE_DIR / "asr_cache.json"


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


@functools.lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)


def transcribe(audio_filename: str) -> str:
    """
    Returns the transcript for a voice note. Cached to disk by filename.
    """
    cache = _load_cache()
    if audio_filename in cache:
        return cache[audio_filename]

    audio_path = AUDIO_DIR / audio_filename
    text = ""
    if not audio_path.exists():
        print(f"[asr] WARNING: audio not found: {audio_path}")
    else:
        try:
            model = _get_model()
            segments, _info = model.transcribe(
                str(audio_path), beam_size=5, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            print(f"[asr] transcription failed ({e}); returning empty text")
            text = ""

    cache[audio_filename] = text
    _save_cache(cache)
    return text
