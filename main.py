"""Entry point: wires the Gemini client into the Telegram bot and runs it."""

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv

from conversation import ConversationStore
from gemini_client import GeminiClient
from handlers import router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
# The SDK warns about automatic function calling on every request.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def read_token() -> str:
    """Return the bot token, failing fast when it is not configured."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Set it in your .env file.")
    return token


async def main() -> None:
    """Start long polling until the process is stopped."""
    bot = Bot(
        token=read_token(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    gemini = GeminiClient()
    conversations = ConversationStore(gemini)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    me = await bot.get_me()
    logger.info("Starting @%s with %s", me.username, gemini.model)
    await bot.delete_webhook(drop_pending_updates=True)
    try:
        # `conversations` is injected into handlers that declare it.
        await dispatcher.start_polling(bot, conversations=conversations)
    finally:
        await gemini.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")
