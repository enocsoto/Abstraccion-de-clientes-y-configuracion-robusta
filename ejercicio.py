import asyncio
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMConfig(BaseModel):
    """API keys stay in SecretStr. Only the selected provider must have one."""

    model_config = ConfigDict(extra="forbid")

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def blank_key_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            raw = value.get_secret_value()
        elif isinstance(value, str):
            raw = value
        else:
            return value
        if raw.strip() == "":
            return None
        return value

    def key_for(self, provider: Provider) -> SecretStr:
        """Return the key for this provider, or raise if it is missing."""
        selected = Provider(provider)
        key = (
            self.openai_api_key
            if selected is Provider.OPENAI
            else self.anthropic_api_key
        )
        if key is None or key.get_secret_value().strip() == "":
            raise ValueError(
                f"API key for provider '{selected.value}' is missing or blank."
            )
        return key


class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, prompt: str) -> str:
        pass


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: SecretStr, model: str) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key.get_secret_value())

    async def chat(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or content.strip() == "":
            raise RuntimeError("OpenAI response content is empty.")
        return content


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: SecretStr, model: str) -> None:
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key.get_secret_value())

    async def chat(self, prompt: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        chunks = [
            block.text
            for block in response.content
            if isinstance(getattr(block, "text", None), str) and block.text.strip()
        ]
        reply = "".join(chunks)
        if reply.strip() == "":
            raise RuntimeError("Anthropic response did not include text.")
        return reply


class LLMFactory:
    @staticmethod
    def create_client(provider: Provider, config: LLMConfig) -> BaseLLMClient:
        selected = Provider(provider)
        if selected is Provider.OPENAI:
            client: BaseLLMClient = OpenAIClient(
                config.key_for(selected),
                config.openai_model,
            )
        elif selected is Provider.ANTHROPIC:
            client = AnthropicClient(
                config.key_for(selected),
                config.anthropic_model,
            )
        else:
            raise ValueError(f"Unsupported provider: {selected.value}")

        logger.info("cliente creado para %s", selected.value)
        return client


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
