"""Validated LLM settings. API keys stay inside SecretStr."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator


class Provider(str, Enum):
    """Supported LLM vendors."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMConfig(BaseModel):
    """Provider settings. A key is required only for the selected provider."""

    model_config = ConfigDict(extra="forbid")

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def blank_key_to_none(cls, value: object) -> object:
        """Treat a blank string as a missing key."""
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
        """Return the key for provider, or raise if it is missing or blank."""
        try:
            selected = Provider(provider)
        except ValueError as exc:
            raise ValueError(f"Unsupported provider: {provider}") from exc

        if selected is Provider.OPENAI:
            key = self.openai_api_key
        elif selected is Provider.ANTHROPIC:
            key = self.anthropic_api_key
        else:
            raise ValueError(f"Unsupported provider: {selected.value}")

        if key is None or key.get_secret_value().strip() == "":
            raise ValueError(
                f"API key for provider '{selected.value}' is missing or blank."
            )
        return key
