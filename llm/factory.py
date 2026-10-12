"""Select a concrete LLM client without leaking vendor SDKs to callers."""

import logging

from llm.clients import AnthropicClient, BaseLLMClient, OpenAIClient
from llm.config import LLMConfig, Provider

logger = logging.getLogger(__name__)


class LLMFactory:
    """Build the client that matches the selected provider."""

    @staticmethod
    def create_client(provider: Provider, config: LLMConfig) -> BaseLLMClient:
        """Return an OpenAI or Anthropic client. Raise if provider is unknown."""
        try:
            selected = Provider(provider)
        except ValueError as exc:
            raise ValueError(f"Unsupported provider: {provider}") from exc

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
