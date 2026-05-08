"""QLoRA fine-tuning of Gemma 4 E2B for activity schedule generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import typer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help="QLoRA fine-tune Gemma 4 E2B on the activity schedule pool.")

MODEL = "google/gemma-4-E2B-it"
OUTPUT_DIR = Path("outputs/finetune/gemma-4-E2B-it-actllm")

# Gemma 4 uses interleaved local/global attention layers.
# If training fails with "target modules not found", inspect layer names with:
#   for name, _ in model.named_modules(): print(name)
# and update this list to cover both local- and global-attention projections.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Gemma 4 uses <|turn>model\n (not Gemma 3's <start_of_turn>model\n).
# Token IDs are computed at runtime from the loaded tokenizer.
_RESPONSE_TEMPLATE = "<|turn>model\n"


@dataclass
class _CompletionOnlyCollator:
    """Pads a batch and masks prompt tokens so loss is only on assistant completions."""

    tokenizer: PreTrainedTokenizerBase
    response_template_ids: list[int]

    def __call__(self, features: list[dict]) -> dict:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        n = len(self.response_template_ids)

        for i, ids in enumerate(batch["input_ids"].tolist()):
            # Find the model-turn delimiter and mask everything up to and including it
            pos = next(
                (j for j in range(len(ids) - n + 1) if ids[j : j + n] == self.response_template_ids),
                None,
            )
            if pos is None:
                labels[i] = -100  # template not found — skip this example's loss
            else:
                labels[i, : pos + n] = -100

        # Mask padding positions
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


@app.command()
def main(
    data_dir: Path = typer.Option(Path("data/finetune"), "--data-dir"),
    output_dir: Path = typer.Option(OUTPUT_DIR, "--output-dir"),
    model_name: str = typer.Option(MODEL, "--model"),
    epochs: int = typer.Option(3, "--epochs"),
    lr: float = typer.Option(2e-4, "--lr"),
    rank: int = typer.Option(16, "--rank"),
) -> None:
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    if not train_file.exists() or not val_file.exists():
        typer.echo(
            f"Missing {train_file} or {val_file} — run prepare_finetune_data.py first",
            err=True,
        )
        raise typer.Exit(1)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    response_template_ids = tokenizer.encode(_RESPONSE_TEMPLATE, add_special_tokens=False)
    logger.info("Response template %r → token IDs %s", _RESPONSE_TEMPLATE, response_template_ids)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Loading model with 4-bit quantization: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )

    collator = _CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template_ids=response_template_ids,
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        dataset_text_field="text",
        max_seq_length=640,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )

    logger.info("Training...")
    trainer.train()

    logger.info("Saving LoRA adapter → %s", adapter_dir)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Done.")


if __name__ == "__main__":
    app()
