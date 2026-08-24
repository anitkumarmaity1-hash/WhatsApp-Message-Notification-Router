"""
OCR for image posters / screenshots using EasyOCR — a free, local,
CNN-based OCR engine that's substantially more robust than classic
Tesseract on stylised poster text, multi-colour backgrounds, and mixed
fonts, which is why it (or a close cousin like PaddleOCR) is the default
choice for real-world scene-text/document OCR pipelines today.

Falls back to pytesseract if EasyOCR fails to import or run, so the
pipeline never hard-crashes on a single bad image.
"""
import json
import functools
from pathlib import Path

from config import IMAGES_DIR, CACHE_DIR, OCR_LANGS, OCR_GPU

_CACHE_PATH = CACHE_DIR / "ocr_cache.json"


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
def _get_easyocr_reader():
    import easyocr  # heavy import, deferred until first real use
    return easyocr.Reader(OCR_LANGS, gpu=OCR_GPU, verbose=False)


def _ocr_with_easyocr(image_path: Path) -> str:
    reader = _get_easyocr_reader()
    results = reader.readtext(str(image_path), detail=0, paragraph=True)
    return " ".join(str(r) for r in results).strip()


def _ocr_with_tesseract(image_path: Path) -> str:
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(image_path)).strip()


def extract_text(image_filename: str) -> str:
    """
    Returns OCR-extracted text for a poster/screenshot. Cached to disk by
    filename so re-runs during development don't re-run the model.
    """
    cache = _load_cache()
    if image_filename in cache:
        return cache[image_filename]

    image_path = IMAGES_DIR / image_filename
    text = ""
    if not image_path.exists():
        print(f"[ocr] WARNING: image not found: {image_path}")
    else:
        try:
            text = _ocr_with_easyocr(image_path)
        except Exception as e:
            print(f"[ocr] EasyOCR failed ({e}); falling back to pytesseract")
            try:
                text = _ocr_with_tesseract(image_path)
            except Exception as e2:
                print(
                    f"[ocr] pytesseract also failed ({e2}); returning empty text")
                text = ""

    cache[image_filename] = text
    _save_cache(cache)
    return text
