"""Read environment settings and print one model reply."""

import asyncio
import os

from llm.config import LLMConfig, Provider
from llm.factory import LLMFactory


async def main() -> None:
    """Build a client from the environment and print its reply."""
    provider = Provider(os.environ.get("LLM_PROVIDER", "openai"))
    config = LLMConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    client = LLMFactory.create_client(provider, config)
    reply = await client.chat("Di hola en una frase.")
    print(reply)


if __name__ == "__main__":
    asyncio.run(main())
