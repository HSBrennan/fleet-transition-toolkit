"""AI assistant integration using the OpenAI API (optional)."""

from __future__ import annotations

import os
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for tests
    OpenAI = None  # type: ignore


DEFAULT_MODEL = "gpt-4o-mini"


class FleetAssistant:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = self._build_client()

    def _build_client(self):
        if not self.api_key or OpenAI is None:
            return None
        return OpenAI(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        if not prompt.strip():
            return "Please provide a question or instruction."

        if not self.available:
            return (
                "No language model is configured. Set the OPENAI_API_KEY environment variable "
                "and restart the app to enable the AI assistant."
            )

        system_message = (
            "You are FleetGuide, an expert assistant helping users design low-carbon fleets. "
            "Explain assumptions, compare energy pathways, and guide scenario creation."
        )

        if context:
            prompt = f"Context:\n{context}\n\nUser question:\n{prompt}"

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=800,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            return f"LLM call failed: {exc}"

        if not response.choices:
            return "The language model did not return any output."

        return response.choices[0].message.get("content", "No answer returned.")

