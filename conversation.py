"""Per-chat conversation memory.

Each Telegram chat gets its own Gemini chat session, so follow-up questions
keep their context. Sessions live in memory only: a restart forgets them.
"""

import asyncio
import logging
from collections import OrderedDict

from google.genai.chats import AsyncChat

from gemini_client import GeminiClient, GeminiError

logger = logging.getLogger(__name__)

# One turn is a user message plus the model's reply.
DEFAULT_MAX_TURNS = 20
# Cap the number of remembered chats so a busy bot cannot exhaust memory.
DEFAULT_MAX_CHATS = 500


class ConversationStore:
    """Keeps one Gemini chat session per Telegram chat, with bounded history."""

    def __init__(
        self,
        client: GeminiClient,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_chats: int = DEFAULT_MAX_CHATS,
    ) -> None:
        self._client = client
        self._max_turns = max_turns
        self._max_chats = max_chats
        self._sessions: OrderedDict[int, AsyncChat] = OrderedDict()
        self._locks: dict[int, asyncio.Lock] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    async def reply(self, chat_id: int, prompt: str) -> str:
        """Answer `prompt` in the context of `chat_id`'s history.

        Raises:
            GeminiError: the prompt was empty, the API call failed, or the
                model returned no text.
        """
        if not prompt or not prompt.strip():
            raise GeminiError("Prompt is empty.")

        # One request at a time per chat: concurrent sends corrupt the history.
        async with self._locks.setdefault(chat_id, asyncio.Lock()):
            session = self._session_for(chat_id)
            try:
                response = await session.send_message(prompt)
            except Exception as exc:  # SDK raises several unrelated error types.
                logger.exception("Gemini chat failed for %s", chat_id)
                raise GeminiError(f"Gemini request failed: {exc}") from exc

            text = (response.text or "").strip()
            if not text:
                raise GeminiError("Gemini returned an empty response.")

            self._trim(chat_id, session)
            return text

    def reset(self, chat_id: int) -> bool:
        """Forget `chat_id`'s history. Returns True if there was any."""
        self._locks.pop(chat_id, None)
        return self._sessions.pop(chat_id, None) is not None

    def _session_for(self, chat_id: int) -> AsyncChat:
        """Return the chat's session, creating it and evicting the oldest."""
        session = self._sessions.pop(chat_id, None)
        if session is None:
            session = self._client.create_chat()
            logger.info("New conversation for chat %s", chat_id)
        self._sessions[chat_id] = session  # Re-insert to mark as most recent.

        while len(self._sessions) > self._max_chats:
            evicted, _ = self._sessions.popitem(last=False)
            self._locks.pop(evicted, None)
            logger.info("Evicted conversation for chat %s", evicted)
        return session

    def _trim(self, chat_id: int, session: AsyncChat) -> None:
        """Replace the session when its history outgrows `max_turns`."""
        history = session.get_history()
        limit = self._max_turns * 2
        if len(history) <= limit:
            return

        tail = history[-limit:]
        # Gemini requires the history to start with a user turn.
        while tail and tail[0].role != "user":
            tail = tail[1:]
        self._sessions[chat_id] = self._client.create_chat_with_history(tail)
        logger.info("Trimmed history for chat %s to %d entries", chat_id, len(tail))
