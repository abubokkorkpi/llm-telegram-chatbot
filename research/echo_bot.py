"""Minimal aiogram echo bot (research spike)."""

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Set it in your .env file.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Set it in your .env file.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()


WELCOME_MESSAGE = (
    "👋 Hi, {name}!\n\n"
    "I'm an <b>echo bot</b> — send me any text and I'll send it right back.\n\n"
    "<b>Commands</b>\n"
    "/start — show this message"
)


@dp.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Greet the user and explain what the bot does."""
    name = html.quote(message.from_user.full_name) if message.from_user else "there"
    logger.info("Start command from chat %s", message.chat.id)
    await message.answer(WELCOME_MESSAGE.format(name=name))


@dp.message(F.text)
async def handle_echo(message: Message) -> None:
    """Send the user's text back to them."""
    logger.info("Echoing %d chars to chat %s", len(message.text), message.chat.id)
    # Escaped because the bot's default parse mode is HTML.
    await message.answer(html.quote(message.text))


async def main() -> None:
    """Start long polling until the process is stopped."""
    me = await bot.get_me()
    logger.info("Starting @%s", me.username)
    await bot.delete_webhook(drop_pending_updates=True)
    # start_polling closes the bot session on shutdown (close_bot_session=True).
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")
