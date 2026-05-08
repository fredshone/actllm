from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


class _SuppressBPETokenizationWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "clean_up_tokenization_spaces" not in record.getMessage()


@dataclass
class ModelConfig:
    provider: str
    name: str
    temperature: float
    max_tokens: int
    max_retries: int
    adapter_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(
            provider=d["provider"],
            name=d["name"],
            temperature=d["temperature"],
            max_tokens=d["max_tokens"],
            max_retries=d["max_retries"],
            adapter_path=d.get("adapter_path"),
        )


class LLMClient:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        if config.provider == "anthropic":
            self._client = anthropic.Anthropic()
        elif config.provider == "local":
            import torch
            from transformers import AutoTokenizer, pipeline as hf_pipeline  # type: ignore[import-untyped]

            # The pipeline calls tokenizer.decode(..., clean_up_tokenization_spaces=True)
            # internally, which triggers a benign log warning on BPE tokenizers (Gemma).
            # Suppress it via a targeted filter on the emitting logger; the actual decode
            # behaviour is already correct (the True is ignored for BPE).
            logging.getLogger("transformers.tokenization_utils_tokenizers").addFilter(
                _SuppressBPETokenizationWarning()
            )
            tokenizer = AutoTokenizer.from_pretrained(
                config.name, clean_up_tokenization_spaces=False
            )
            if config.adapter_path:
                from peft import PeftModel  # type: ignore[import-untyped]
                from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

                base = AutoModelForCausalLM.from_pretrained(
                    config.name, torch_dtype=torch.bfloat16, device_map="auto"
                )
                model = PeftModel.from_pretrained(base, config.adapter_path)
                self._pipeline = hf_pipeline(
                    "text-generation", model=model, tokenizer=tokenizer
                )
            else:
                self._pipeline = hf_pipeline(
                    "text-generation",
                    model=config.name,
                    tokenizer=tokenizer,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._config.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        elif self._config.provider == "local":
            return self._generate_local(system_prompt, user_prompt)
        raise ValueError(f"Unsupported provider: {self._config.provider}")

    def generate_batch(self, prompts: list[tuple[str, str]]) -> list[str]:
        if self._config.provider != "local":
            raise ValueError("generate_batch is only supported for the local provider")
        from transformers import GenerationConfig  # type: ignore[import-untyped]

        messages_batch = [
            [{"role": "system", "content": sp}, {"role": "user", "content": up}]
            for sp, up in prompts
        ]
        gen_config = GenerationConfig(
            max_new_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            do_sample=True,
        )
        outputs = self._pipeline(messages_batch, generation_config=gen_config)  # type: ignore[attr-defined]
        return [out[0]["generated_text"][-1]["content"] for out in outputs]  # type: ignore[no-any-return]

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
        from transformers import GenerationConfig  # type: ignore[import-untyped]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        gen_config = GenerationConfig(
            max_new_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            do_sample=True,
        )
        outputs = self._pipeline(messages, generation_config=gen_config)  # type: ignore[attr-defined]
        return outputs[0]["generated_text"][-1]["content"]  # type: ignore[no-any-return]
