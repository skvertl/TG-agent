import asyncio
import logging
import sys
from config import Settings
from core.runner import AgentRunner
from adapters.memory.sqlite import SqliteMemoryStore
from adapters.llm.ollama_plugin import OllamaPlugin
from adapters.telegram.bot import create_bot, create_dispatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app.main")


async def main() -> None:
    logger.info("Initializing Telegram Autonomous Agent Gateway...")
    settings = Settings()

    # 1. Driven Adapters
    memory_store = SqliteMemoryStore()
    llm_plugin = OllamaPlugin(
        base_url=settings.ollama_base_url,
        model_name=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )

    # 2. Observability & Tools
    from core.observability.engine import ObservabilityEngine
    from core.tools.registry import ToolRegistry

    obs_engine = ObservabilityEngine(project_name="tg-agent")
    tool_registry = ToolRegistry(engine=obs_engine)

    # 3. Core Domain Orchestrator
    runner = AgentRunner(
        llm_plugin=llm_plugin,
        memory_store=memory_store,
        system_prompt=settings.system_prompt,
        observability_engine=obs_engine,
        tool_registry=tool_registry,
        skills_dir="skills",
    )

    # 3. Driving Telegram Adapter
    bot = create_bot(token=settings.telegram_bot_token)
    dp = create_dispatcher(
        runner=runner,
        allowed_user_ids=settings.allowed_user_ids,
    )
    if settings.allowed_user_ids:
        logger.info("Access whitelist active: %s", settings.allowed_user_ids)
    else:
        logger.info("No whitelist configured: bot responds to all users.")

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
