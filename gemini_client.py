"""Gemini client for the Telegram chatbot.

Wraps `google-genai` so the rest of the app depends on this small surface
instead of the SDK directly. Async, because aiogram handlers are async.
"""

import logging
import os

from dotenv import load_dotenv
from google.genai import Client
from google.genai import types
from google.genai.chats import AsyncChat

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant replying inside Telegram. "
    "Keep answers short and plain-spoken."
)


class GeminiError(RuntimeError):
    """Raised when a Gemini request cannot be completed."""


class GeminiClient:
    """Thin async wrapper around the Gemini `generate_content` API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise GeminiError("GEMINI_API_KEY is missing. Set it in your .env file.")

        self._client = Client(api_key=key)
        self._model = model
        self._config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    @property
    def model(self) -> str:
        """Name of the model this client talks to."""
        return self._model

    def create_chat(self) -> AsyncChat:
        """Start a stateful chat session that remembers its own history."""
        return self.create_chat_with_history(None)

    def create_chat_with_history(self, history: list[types.Content] | None) -> AsyncChat:
        """Start a chat session seeded with earlier turns."""
        return self._client.aio.chats.create(
            model=self._model, config=self._config, history=history
        )

    async def generate(self, prompt: str) -> str:
        """Return Gemini's reply to `prompt`.

        Raises:
            GeminiError: the prompt was empty, the API call failed, or the
                model returned no text (for example, a safety block).
        """
        if not prompt or not prompt.strip():
            raise GeminiError("Prompt is empty.")

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._config,
            )
        except Exception as exc:  # SDK raises several unrelated error types.
            logger.exception("Gemini request failed")
            raise GeminiError(f"Gemini request failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise GeminiError("Gemini returned an empty response.")
        return text

    async def close(self) -> None:
        """Release the underlying HTTP session."""
        await self._client.aio.aclose()
