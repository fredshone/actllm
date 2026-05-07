from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    provider: str
    name: str
    temperature: float
    max_tokens: int
    max_retries: int

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(
            provider=d["provider"],
            name=d["name"],
            temperature=d["temperature"],
            max_tokens=d["max_tokens"],
            max_retries=d["max_retries"],
        )


class LLMClient:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        if config.provider == "anthropic":
            self._client = anthropic.Anthropic()
        elif config.provider == "local":
            import torch
            from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]

            self._pipeline = hf_pipeline(
                "text-generation",
                model=config.name,
                device="cuda",
                torch_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._config.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        elif self._config.provider == "local":
            return self._generate_local(system_prompt, user_prompt)
        raise ValueError(f"Unsupported provider: {self._config.provider}")

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._config.name,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        usage = response.usage
        logger.debug(
            "Token usage — input: %d, output: %d, cache_read: %s, cache_write: %s",
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", "n/a"),
            getattr(usage, "cache_creation_input_tokens", "n/a"),
        )
        content = response.content[0]
        if content.type != "text":
            raise ValueError(f"Unexpected response content type: {content.type}")
        return content.text

    def _generate_local(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        outputs = self._pipeline(  # type: ignore[attr-defined]
            messages,
            max_new_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            do_sample=True,
        )
        return outputs[0]["generated_text"][-1]["content"]  # type: ignore[no-any-return]
