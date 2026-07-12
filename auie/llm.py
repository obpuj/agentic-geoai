"""LLM client abstraction (pipeline stages 2 and 9 share this).

One interface, three implementations: Gemini, Groq, and a ScriptedClient for
offline tests. Keeping both live providers behind one protocol is the demo-day
insurance discussed in the risk register — either can serve if the other is
down or rate-limited.

Keys are read from environment variables: GEMINI_API_KEY, GROQ_API_KEY.
Uses plain HTTP (requests) — no SDK version headaches on Kaggle/Colab.
"""
from __future__ import annotations

import os
import time
from typing import Protocol

import requests

_MAX_RETRIES = 4


def _post_with_retry(url: str, **kwargs) -> requests.Response:
    """POST with backoff on 429/5xx. Honors Retry-After when present.

    Rate limits are a fact of life on free tiers (Groq ~30 req/min) and must
    never crash a demo: wait, retry, and only raise after _MAX_RETRIES.
    """
    delay = 2.0
    for attempt in range(_MAX_RETRIES + 1):
        resp = requests.post(url, **kwargs)
        if resp.status_code not in (429, 500, 502, 503, 529):
            resp.raise_for_status()
            return resp
        if attempt == _MAX_RETRIES:
            resp.raise_for_status()
        wait = resp.headers.get("Retry-After")
        wait_s = float(wait) if wait and wait.replace(".", "", 1).isdigit() else delay
        time.sleep(min(wait_s, 30.0))
        delay *= 2
    raise RuntimeError("unreachable")


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class GeminiClient:
    MODEL = "gemini-2.0-flash"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        resp = _post_with_retry(
            self.URL.format(m=self.MODEL),
            params={"key": self.api_key},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            },
            timeout=self.timeout,
        )
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


class GroqClient:
    MODEL = "llama-3.3-70b-versatile"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ["GROQ_API_KEY"]
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        resp = _post_with_retry(
            self.URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.MODEL,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        return resp.json()["choices"][0]["message"]["content"]


class ScriptedClient:
    """Returns pre-scripted responses in order. For offline unit tests."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self._responses:
            raise RuntimeError("ScriptedClient ran out of responses")
        return self._responses.pop(0)


def default_client() -> LLMClient:
    """Prefer Gemini, fall back to Groq, based on available keys."""
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiClient()
    if os.environ.get("GROQ_API_KEY"):
        return GroqClient()
    raise EnvironmentError("Set GEMINI_API_KEY or GROQ_API_KEY")
