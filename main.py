import asyncio
import logging
import sys
from config import Settings
from core.runner import AgentRunner
from adapters.memory.stateless import StatelessMemoryStore
from adapters.llm.ollama_plugin import OllamaPlugin
from adapters.telegram.bot import create_bot, create_dispatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app.main")


async def main() -> None:
    logger.info("Initializing Telegram Ollama Gateway...")
    settings = Settings()

    # 1. Driven Adapters
    memory_store = StatelessMemoryStore()
    llm_plugin = OllamaPlugin(
        base_url=settings.ollama_base_url,
        model_name=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )

    # 2. Core Domain Orchestrator
    runner = AgentRunner(
        llm_plugin=llm_plugin,
        memory_store=memory_store,
        system_prompt=settings.system_prompt,
    )

    # 3. Driving Telegram Adapter
    bot = create_bot(token=settings.telegram_bot_token)
    dp = create_dispatcher(runner=runner)

    try:
        logger.info(
            "Connecting to Ollama at %s (model: %s)...",
            settings.ollama_base_url,
            settings.ollama_model,
        )
        await llm_plugin.initialize()
        logger.info("Ollama plugin initialized successfully.")

        logger.info("Starting Telegram Bot long polling...")
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted by user or signal. Initiating graceful shutdown...")
    except Exception as e:
        logger.exception("Fatal error during execution: %s", e)
    finally:
        logger.info("Cleaning up resources...")
        await llm_plugin.shutdown()
        await bot.session.close()
        logger.info("Gateway stopped gracefully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
