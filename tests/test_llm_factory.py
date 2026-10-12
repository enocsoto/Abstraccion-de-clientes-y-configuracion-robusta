"""LLM factory tests. Stdlib and unittest.mock only; no network."""

import asyncio
import inspect
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.clients import AnthropicClient, OpenAIClient
from llm.config import LLMConfig, Provider
from llm.factory import LLMFactory
from llm.__main__ import main

OPENAI_KEY = "sk-test-openai-raw-key"
ANTHROPIC_KEY = "sk-ant-test-raw-key"


def _openai_response(content: str | None) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _anthropic_response(blocks: list[object]) -> MagicMock:
    response = MagicMock()
    response.content = blocks
    return response


class TestLLMFactory(unittest.TestCase):
    def test_factory_returns_openai_client(self) -> None:
        config = LLMConfig(openai_api_key=OPENAI_KEY)
        with patch("llm.clients.AsyncOpenAI") as sdk:
            client = LLMFactory.create_client(Provider.OPENAI, config)
        self.assertIsInstance(client, OpenAIClient)
        sdk.assert_called_once_with(api_key=OPENAI_KEY)

    def test_factory_returns_anthropic_client(self) -> None:
        config = LLMConfig(anthropic_api_key=ANTHROPIC_KEY)
        with patch("llm.clients.AsyncAnthropic") as sdk:
            client = LLMFactory.create_client(Provider.ANTHROPIC, config)
        self.assertIsInstance(client, AnthropicClient)
        sdk.assert_called_once_with(api_key=ANTHROPIC_KEY)

    def test_missing_openai_key_does_not_construct_sdk(self) -> None:
        config = LLMConfig(anthropic_api_key=ANTHROPIC_KEY)
        with patch("llm.clients.AsyncOpenAI") as sdk:
            with self.assertRaisesRegex(ValueError, "missing or blank"):
                LLMFactory.create_client(Provider.OPENAI, config)
        sdk.assert_not_called()

    def test_missing_anthropic_key_does_not_construct_sdk(self) -> None:
        config = LLMConfig(openai_api_key=OPENAI_KEY)
        with patch("llm.clients.AsyncAnthropic") as sdk:
            with self.assertRaisesRegex(ValueError, "missing or blank"):
                LLMFactory.create_client(Provider.ANTHROPIC, config)
        sdk.assert_not_called()

    def test_other_provider_key_can_be_absent(self) -> None:
        openai_only = LLMConfig(openai_api_key=OPENAI_KEY)
        with patch("llm.clients.AsyncOpenAI") as sdk:
            client = LLMFactory.create_client(Provider.OPENAI, openai_only)
        self.assertIsInstance(client, OpenAIClient)
        sdk.assert_called_once()

        anthropic_only = LLMConfig(anthropic_api_key=ANTHROPIC_KEY)
        with patch("llm.clients.AsyncAnthropic") as sdk:
            client = LLMFactory.create_client(Provider.ANTHROPIC, anthropic_only)
        self.assertIsInstance(client, AnthropicClient)
        sdk.assert_called_once()

    def test_unsupported_provider_raises(self) -> None:
        config = LLMConfig(openai_api_key=OPENAI_KEY)
        with patch("llm.clients.AsyncOpenAI") as sdk:
            with self.assertRaises(ValueError):
                LLMFactory.create_client("gemini", config)  # type: ignore[arg-type]
        sdk.assert_not_called()

    def test_openai_chat_normalizes_content(self) -> None:
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(
            return_value=_openai_response("Hola, ¿cómo estás?")
        )
        with patch("llm.clients.AsyncOpenAI", return_value=sdk):
            client = OpenAIClient(config_key(OPENAI_KEY), "gpt-4o-mini")
            reply = asyncio.run(client.chat("Di hola"))
        self.assertEqual(reply, "Hola, ¿cómo estás?")
        sdk.chat.completions.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Di hola"}],
        )

    def test_openai_empty_content_raises(self) -> None:
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=_openai_response(None))
        with patch("llm.clients.AsyncOpenAI", return_value=sdk):
            client = OpenAIClient(config_key(OPENAI_KEY), "gpt-4o-mini")
            with self.assertRaises(RuntimeError):
                asyncio.run(client.chat("Di hola"))

    def test_anthropic_chat_joins_text_blocks(self) -> None:
        blocks = [
            SimpleNamespace(text="Hola "),
            SimpleNamespace(type="tool_use"),
            SimpleNamespace(text=""),
            SimpleNamespace(text="mundo"),
        ]
        sdk = MagicMock()
        sdk.messages.create = AsyncMock(return_value=_anthropic_response(blocks))
        with patch("llm.clients.AsyncAnthropic", return_value=sdk):
            client = AnthropicClient(config_key(ANTHROPIC_KEY), "claude-3-5-haiku-latest")
            reply = asyncio.run(client.chat("Di hola"))
        self.assertEqual(reply, "Hola mundo")
        sdk.messages.create.assert_awaited_once_with(
            model="claude-3-5-haiku-latest",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Di hola"}],
        )

    def test_anthropic_without_text_raises(self) -> None:
        sdk = MagicMock()
        sdk.messages.create = AsyncMock(
            return_value=_anthropic_response([SimpleNamespace(type="tool_use")])
        )
        with patch("llm.clients.AsyncAnthropic", return_value=sdk):
            client = AnthropicClient(config_key(ANTHROPIC_KEY), "claude-3-5-haiku-latest")
            with self.assertRaises(RuntimeError):
                asyncio.run(client.chat("Di hola"))

    def test_secret_str_hides_raw_key(self) -> None:
        config = LLMConfig(
            openai_api_key=OPENAI_KEY,
            anthropic_api_key=ANTHROPIC_KEY,
        )
        self.assertNotIn(OPENAI_KEY, str(config.openai_api_key))
        self.assertNotIn(ANTHROPIC_KEY, str(config.anthropic_api_key))
        dumped = config.model_dump(mode="json")
        blob = str(dumped)
        self.assertNotIn(OPENAI_KEY, blob)
        self.assertNotIn(ANTHROPIC_KEY, blob)

    def test_empty_string_key_is_missing(self) -> None:
        config = LLMConfig(openai_api_key="", anthropic_api_key="   ")
        self.assertIsNone(config.openai_api_key)
        self.assertIsNone(config.anthropic_api_key)
        with patch("llm.clients.AsyncOpenAI") as sdk:
            with self.assertRaisesRegex(ValueError, "missing or blank"):
                LLMFactory.create_client(Provider.OPENAI, config)
        sdk.assert_not_called()

    def test_chat_methods_are_coroutines_and_source_has_no_sleep(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(OpenAIClient.chat))
        self.assertTrue(inspect.iscoroutinefunction(AnthropicClient.chat))
        files = list((ROOT / "llm").rglob("*.py"))
        self.assertTrue(files)
        for file in files:
            text = file.read_text(encoding="utf-8")
            self.assertNotIn("time.sleep", text, file.name)

    def test_factory_log_names_provider_without_key(self) -> None:
        config = LLMConfig(openai_api_key=OPENAI_KEY)
        with (
            patch("llm.clients.AsyncOpenAI"),
            self.assertLogs("llm.factory", level="INFO") as captured,
        ):
            LLMFactory.create_client(Provider.OPENAI, config)
        output = "\n".join(captured.output)
        self.assertIn("cliente creado para openai", output)
        self.assertNotIn(OPENAI_KEY, output)

    def test_main_prints_reply(self) -> None:
        class FakeClient:
            async def chat(self, prompt: str) -> str:
                return "Hola, ¿qué tal?"

        env = {"OPENAI_API_KEY": OPENAI_KEY, "LLM_PROVIDER": "openai"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("llm.__main__.LLMFactory.create_client", return_value=FakeClient()),
            patch("builtins.print") as print_mock,
        ):
            asyncio.run(main())
        print_mock.assert_called_once_with("Hola, ¿qué tal?")

    def test_main_missing_key_propagates(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("llm.clients.AsyncOpenAI") as sdk,
        ):
            with self.assertRaises(ValueError):
                asyncio.run(main())
        sdk.assert_not_called()


def config_key(raw: str):
    from pydantic import SecretStr

    return SecretStr(raw)


if __name__ == "__main__":
    unittest.main()
