"""
ASR for voice notes using faster-whisper (CTranslate2 Whisper) - free, local, CPU-friendly.

Changes:
  * Failed transcriptions are NO LONGER cached. Before, one failure (e.g. a broken PyAV install)
    saved "" to .cache/asr_cache.json permanently, so even after fixing the install the voice
    note stayed empty and got routed as a blank message.
  * Empty transcripts are treated as failures too (only non-empty text is cached).
  * PyAV 19 removed the `metadata_errors` argument that faster-whisper's own audio decoder
    passes to av.open(), so EVERY file failed with
        "open() got an unexpected keyword argument 'metadata_errors'".
    Fix, no matter which PyAV is installed: if that happens we decode the file ourselves with PyAV
    (16 kHz mono float32) and hand the array to Whisper. Pinning also works:
        pip install "av>=11,<19"
"""
import functools
import json

from config import AUDIO_DIR, CACHE_DIR, WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

_CACHE_PATH = CACHE_DIR / "asr_cache.json"
_hint_shown = False


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)


def _decode_with_av(path, sample_rate: int = 16000):
    """Version-proof decoder: any audio file -> mono float32 array at 16 kHz (what Whisper expects)."""
    import av
    import numpy as np

    chunks = []
    with av.open(str(path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)

        def collect(frames):
            for f in frames or []:
                chunks.append(f.to_ndarray().reshape(-1))

        for packet in container.demux(stream):
            for frame in packet.decode():
                collect(resampler.resample(frame))
        collect(resampler.resample(None))          # flush
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _explain(err: Exception) -> None:
    global _hint_shown
    print(f"[asr] transcription failed ({err}); returning empty text (not cached)")
    if "metadata_errors" in str(err) and not _hint_shown:
        _hint_shown = True
        print('[asr] hint: pip install "av>=11,<19"  (PyAV 19 is incompatible with faster-whisper)')


def transcribe(audio_filename: str) -> str:
    """Return the transcript for a voice note. Only successful, non-empty results are cached."""
    cache = _load_cache()
    cached = cache.get(audio_filename)
    if cached:                      # '' from an old failed run is ignored and retried
        return cached

    audio_path = AUDIO_DIR / audio_filename
    if not audio_path.exists():
        print(f"[asr] WARNING: audio not found: {audio_path}")
        return ""
    model = _get_model()

    def run(source) -> str:
        segments, _info = model.transcribe(source, beam_size=5, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()

    try:
        try:
            text = run(str(audio_path))
        except TypeError as e:                      # PyAV 19 vs faster-whisper's decoder
            if "metadata_errors" not in str(e):
                raise
            text = run(_decode_with_av(audio_path))
    except Exception as e:
        _explain(e)
        return ""

    if text:
        cache[audio_filename] = text
        _save_cache(cache)
    else:
        print(f"[asr] WARNING: {audio_filename} produced an empty transcript (silence?); not cached")
    return text