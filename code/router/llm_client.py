"""
Thin wrapper around the Groq API (OpenAI-compatible chat completions).

Compared to the original version, this client is budget-aware:
  1. Before every call it estimates the prompt's token cost and asks the
     TokenBudgetTracker which model to use — falling over to the smaller
     GROQ_MODEL_FALLBACK model BEFORE spending tokens the primary model
     doesn't have left today.
  2. If Groq still returns a 429 token-rate-limit error (e.g. our estimate
     was off, or another process spent the budget), it fails over to the
     fallback model IMMEDIATELY instead of burning all of LLM_MAX_RETRIES
     against a quota that cannot recover until the daily reset.
  3. After every successful call it records the *actual* token usage Groq
     reports (response.usage), not just the estimate, so the budget stays
     accurate over a long run.
"""
import json
import re
import time

import requests

from config import (
    GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK, LLM_TEMPERATURE, LLM_MAX_RETRIES,
    OLLAMA_ENABLED, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS,
)
from router.token_budget import get_tracker, estimate_tokens

_client = None

_RATE_LIMIT_PATTERNS = (
    "rate_limit_exceeded",
    "tokens per day",
    "tpd",
    "429",
)


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        from config import GROQ_API_KEY
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _is_token_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(p in msg for p in _RATE_LIMIT_PATTERNS)


def _record_usage(model: str, response, fallback_prompt_tokens: int) -> None:
    tracker = get_tracker()
    usage = getattr(response, "usage", None)
    total = getattr(usage, "total_tokens", None) if usage else None
    tracker.record(model, total if total else fallback_prompt_tokens)


def _call_once(client, model: str, system_prompt: str, user_prompt: str):
    response = client.chat.completions.create(
        model=model,
        temperature=LLM_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Groq returned an empty completion (content is None)")
    return json.loads(content), response


def _call_ollama(system_prompt: str, user_prompt: str) -> dict:
    """
    Calls a local Ollama server (unlimited, $0, no network calls once the
    model is pulled). Used only after every Groq model in model_order is
    out of daily budget. Requires `ollama serve` running locally and the
    model pulled ahead of time, e.g. `ollama pull llama3.1:8b`.
    """
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": LLM_TEMPERATURE},
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    if not content:
        raise ValueError("Ollama returned an empty completion")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Local models sometimes wrap JSON in prose or ```json fences even
        # with format="json" requested — one cheap repair attempt.
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?|```$", "", cleaned).strip()
        return json.loads(cleaned)


def _call_with_failover(system_prompt: str, user_prompt: str, preferred_model: str) -> tuple[dict, str]:
    client = _get_client()
    tracker = get_tracker()
    estimated = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)

    # Two "lanes": the model we'd like to use first, then the fallback. We
    # only ever try each lane once per retry-loop pass so a hard 429 can't
    # eat all LLM_MAX_RETRIES against a quota that's already dead.
    model_order = [preferred_model]
    if GROQ_MODEL_FALLBACK not in model_order:
        model_order.append(GROQ_MODEL_FALLBACK)
    if GROQ_MODEL_PRIMARY not in model_order:
        model_order.append(GROQ_MODEL_PRIMARY)

    last_err = None
    attempt = 0
    for model in model_order:
        if not tracker.has_headroom(model, estimated) and model != model_order[-1]:
            print(f"[llm_client] skipping {model}: insufficient daily budget "
                  f"({tracker.status_line()})")
            continue

        for local_try in range(1, LLM_MAX_RETRIES + 1):
            attempt += 1
            try:
                parsed, response = _call_once(
                    client, model, system_prompt, user_prompt)
                _record_usage(model, response, estimated)
                return parsed, model
            except Exception as e:
                last_err = e
                if _is_token_rate_limit_error(e):
                    tracker.mark_exhausted(model)
                    print(f"[llm_client] {model} hit its daily token limit "
                          f"-> failing over (attempt {attempt})")
                    break  # stop retrying THIS model, move to next in model_order
                wait = min(2 ** local_try, 10)
                print(f"[llm_client] attempt {attempt} on {model} failed "
                      f"({e}); retrying in {wait}s")
                time.sleep(wait)

    # Both Groq lanes are exhausted (or every attempt errored). Last resort:
    # a fully local, unlimited Ollama model before falling back to the
    # blind rule-based default in decision_engine.py.
    if OLLAMA_ENABLED:
        try:
            print(f"[llm_client] both Groq models exhausted -> trying local "
                  f"Ollama model '{OLLAMA_MODEL}'")
            return _call_ollama(system_prompt, user_prompt), f"ollama:{OLLAMA_MODEL}"
        except Exception as e:
            last_err = e
            print(f"[llm_client] Ollama call failed too: {e}")

    raise RuntimeError(
        f"LLM call failed after {attempt} attempt(s) across "
        f"{model_order}{' + ollama' if OLLAMA_ENABLED else ''}: {last_err}")


def call_llm_json(system_prompt: str, user_prompt: str, model: str | None = None) -> tuple[dict, str]:
    """
    Calls the Groq chat completion endpoint in JSON mode and returns
    (parsed dict, model_actually_used). The model name is returned so
    callers can attach it to the decision and we can measure accuracy
    broken down by which tier actually answered (70B vs 8B vs local
    Ollama), instead of guessing from log timestamps.

    Picks the model proactively from the token budget (unless `model` is
    forced), retries transient errors, and fails over to the smaller
    model immediately on a daily-rate-limit (429) error.
    """
    tracker = get_tracker()
    estimated = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    chosen_model = model or tracker.choose_model(
        GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK, estimated)
    return _call_with_failover(system_prompt, user_prompt, chosen_model)
