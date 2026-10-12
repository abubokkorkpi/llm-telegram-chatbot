"""Telegram message handlers.

Handlers receive the `ConversationStore` through aiogram's dependency
injection, so this module never builds one itself.
"""

import logging

from aiogram import F, Router, html
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from conversation import ConversationStore
from gemini_client import GeminiError

logger = logging.getLogger(__name__)

router = Router(name="chat")

# Telegram rejects messages longer than 4096 characters.
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

WELCOME_MESSAGE = (
    "👋 Hi, {name}!\n\n"
    "Ask me anything and I'll answer with <b>Gemini</b>.\n\n"
    "I remember our conversation, so you can ask follow-ups.\n\n"
    "<b>Commands</b>\n"
    "/start — show this message\n"
    "/reset — forget our conversation"
)
ERROR_MESSAGE = "⚠️ Sorry, I couldn't answer that right now. Please try again."


def split_message(text: str, limit: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split `text` into chunks Telegram will accept, breaking on newlines."""
    chunks: list[str] = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Greet the user and explain what the bot does."""
    name = html.quote(message.from_user.full_name) if message.from_user else "there"
    logger.info("Start command from chat %s", message.chat.id)
    await message.answer(WELCOME_MESSAGE.format(name=name))


@router.message(Command("reset"))
async def handle_reset(message: Message, conversations: ConversationStore) -> None:
    """Drop the stored history for this chat."""
    had_history = conversations.reset(message.chat.id)
    logger.info("Reset chat %s (had history: %s)", message.chat.id, had_history)
    await message.answer(
        "🧹 Conversation cleared." if had_history else "Nothing to clear yet."
    )


@router.message(F.text)
async def handle_prompt(message: Message, conversations: ConversationStore) -> None:
    """Answer the user's text with a Gemini completion."""
    logger.info("Prompt from chat %s (%d chars)", message.chat.id, len(message.text))
    try:
        async with ChatActionSender(
            bot=message.bot, chat_id=message.chat.id, action=ChatAction.TYPING
        ):
            answer = await conversations.reply(message.chat.id, message.text)
    except GeminiError:
        logger.exception("Gemini failed for chat %s", message.chat.id)
        await message.answer(ERROR_MESSAGE)
        return

    # Escaped because the bot's default parse mode is HTML.
    for chunk in split_message(html.quote(answer)):
        await message.answer(chunk)


@router.message()
async def handle_unsupported(message: Message) -> None:
    """Explain that only text is supported."""
    await message.answer("I can only read text messages for now.")
