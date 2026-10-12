"""Async vendor clients behind one chat(prompt) -> str interface."""

from abc import ABC, abstractmethod

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import SecretStr


class BaseLLMClient(ABC):
    """Common async chat contract. Callers never see vendor payloads."""

    @abstractmethod
    async def chat(self, prompt: str) -> str:
        """Send a user prompt and return the assistant text."""


class OpenAIClient(BaseLLMClient):
    """OpenAI chat completions adapter."""

    def __init__(self, api_key: SecretStr, model: str) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key.get_secret_value())

    async def chat(self, prompt: str) -> str:
        """Return choices[0].message.content as a plain string."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or content.strip() == "":
            raise RuntimeError("OpenAI response content is empty.")
        return content


class AnthropicClient(BaseLLMClient):
    """Anthropic messages adapter."""

    def __init__(self, api_key: SecretStr, model: str) -> None:
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key.get_secret_value())

    async def chat(self, prompt: str) -> str:
        """Join text blocks from the messages response into one string."""
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        chunks: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip() != "":
                chunks.append(text)
        reply = "".join(chunks)
        if reply.strip() == "":
            raise RuntimeError("Anthropic response did not include text.")
        return reply
