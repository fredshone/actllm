# actllm

LLM-based 24-hour activity schedule generation conditioned on person-level attributes. Produces [caveat](https://github.com/big-ucl/caveat) compatible output for evaluation against the ActVAE model using [acteval](https://github.com/fredshone/acteval).

## Latest results

| Domain         | gemma4-2B-it (1) | qwen2.5-0.5B-it (1) | qwen2.5-0.5B-it (2) |
|----------------|------------------|----------------------|----------------------|
| creativity     | **0.0625**       | 0.490                | 0.210                |
| feasibility    | 0.00294          | 0.0710               | **0.000**            |
| participations | 1.29             | 0.282                | **0.184**            |
| timing         | 0.228            | **0.148**            | 0.203                |
| transitions    | 0.0373           | 0.0212               | **0.0160**           |

(1) 3-shot nearest-neighbour  
(2) 0-shot fine-tuned

## Setup

### Anthropic (default)

```bash
uv sync --extra dev
cp .env.example .env  # add your ANTHROPIC_API_KEY
```

```
ANTHROPIC_API_KEY=sk-ant-...
```

`actllm` always loads `.env` automatically.

### Local model (GPU)

Requires a CUDA-capable GPU. Installs `transformers` and `accelerate` alongside the base dependencies:

```bash
uv sync --extra local
```

Supported models and their configs:

| Model | Config | Size | Notes |
|-------|--------|------|-------|
| `google/gemma-4-E2B-it` | `configs/gemma.yaml` | ~4 GB | Multimodal; zero-shot default |
| `Qwen/Qwen2.5-0.5B-Instruct` | `configs/qwen.yaml` | ~1 GB | Lightweight; 3-shot nearest default |

Models are downloaded from HuggingFace on first use. No API key needed.

## Generate schedules

```bash
# Preview prompt without API calls
uv run python scripts/generate.py \
  --input data/nts_attributes_2023.csv \
  --output outputs/run.jsonl \
  --dry-run

# Generate 100 schedules
uv run python scripts/generate.py \
  --input data/nts_attributes_2023.csv \
  --output outputs/run.jsonl \
  --n-samples 100
```

`--output` always produces two files: the JSONL at the given path and an NTS-compatible CSV at the same path with a `.csv` extension (e.g. `outputs/run.csv`). Output is streamed as it is generated — safe to interrupt.

### Running with a local model

```bash
# Gemma 4 E2B
uv run python scripts/generate.py \
  --config configs/gemma.yaml \
  --input data/nts_attributes_2023.csv \
  --output outputs/gemma_run.jsonl \
  --n-samples 100

# Qwen 2.5 0.5B
uv run python scripts/generate.py \
  --config configs/qwen.yaml \
  --input data/nts_attributes_2023.csv \
  --output outputs/qwen_run.jsonl \
  --n-samples 100
```

The first run downloads the model weights. Subsequent runs load from the HuggingFace cache. Inference is sequential (`concurrency: 1`) — expect slower throughput than the Anthropic API but no per-token cost.

### Inspecting prompts

```bash
# Print the prompt for the first record without calling the API
uv run python scripts/generate.py \
  --input data/nts_attributes_2023.csv \
  --output outputs/run.jsonl \
  --dry-run

# Log full prompts during a live run
uv run python scripts/generate.py \
  --input data/nts_attributes_2023.csv \
  --output outputs/run.jsonl \
  --n-samples 5 \
  --debug
```

`--debug` prints every system and user prompt to stderr before each API call, including retry attempts.

## Prompt modes

| Mode | Description |
|------|-------------|
| `zero_shot` | No examples; JSON-only output |
| `few_shot` | N real NTS examples prepended; selection strategy configurable |
| `cot_zero_shot` | Model reasons in `<reasoning>` block then outputs JSON in `<schedule>` block |
| `cot_few_shot` | CoT + few-shot examples |

Configure via `configs/default.yaml` or any ablation config:

```yaml
prompt:
  mode: few_shot            # zero_shot | few_shot | cot_zero_shot | cot_few_shot
  n_examples: 3
  example_selection: nearest_neighbour   # random | stratified | nearest_neighbour
  few_shot_pool: data/few_shot/pool.jsonl
```

## Fine-tuning

QLoRA fine-tuning on the curated schedule pool. Supported models:

| Model | VRAM | Training time |
|-------|------|---------------|
| `google/gemma-4-E2B-it` | ≥24 GB | ~3 h (RTX A5000) |
| `Qwen/Qwen2.5-0.5B-Instruct` | ≥8 GB | ~30 min (RTX A5000) |

```bash
uv sync --extra finetune
```

**1. Prepare training data**

Joins `activities.csv` and `attributes_binned.csv` from the NTS 2023 output, converts minute-offset start times to HH:MM, and writes `train.jsonl` and `val.jsonl` using the model's chat template (stratified 90/10 split, ~56K persons).

```bash
# Gemma
uv run python scripts/prepare_finetune_data.py \
  --model google/gemma-4-E2B-it \
  --output-dir data/finetune/gemma

# Qwen
uv run python scripts/prepare_finetune_data.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir data/finetune/qwen
```

Pass `--verify` to print token count statistics before committing to training. Override the default NTS paths if your data lives elsewhere:

```bash
uv run python scripts/prepare_finetune_data.py \
  --activities ~/Data/foundata/out/nts/2023/activities.csv \
  --attributes ~/Data/foundata/out/nts/2023/attributes_binned.csv \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir data/finetune/qwen
```

**2. Train**

Runs QLoRA (4-bit NF4 base + LoRA rank-16 adapters on attention layers) for 3 epochs with gradient checkpointing. Adapters are saved to `outputs/finetune/<model>/adapter/`.

```bash
# Gemma
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/finetune.py \
  --model google/gemma-4-E2B-it \
  --data-dir data/finetune/gemma

# Qwen
uv run python scripts/finetune.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir data/finetune/qwen
```

Key options: `--epochs` (default 3), `--lr` (default 2e-4), `--rank` (default 16).

**3. Generate with the fine-tuned model**

```bash
# Gemma fine-tuned
uv run python scripts/generate.py \
  --config configs/gemma_finetuned.yaml \
  --input data/nts_attributes_2023.csv \
  --output outputs/gemma_finetuned_run.jsonl \
  --n-samples 100

# Qwen fine-tuned
uv run python scripts/generate.py \
  --config configs/qwen_finetuned.yaml \
  --input data/nts_attributes_2023.csv \
  --output outputs/qwen_finetuned_run.jsonl \
  --n-samples 100
```

Both configs use zero-shot prompting — fine-tuning subsumes few-shot examples.

## Few-shot pool

The pool (`data/few_shot/pool.jsonl`) contains 58 real NTS schedules sampled across 15 strata (5 age bands × 3 work statuses). To regenerate:

```bash
# Preview strata analysis without writing
uv run python scripts/curate_few_shot.py --dry-run

# Write the pool
uv run python scripts/curate_few_shot.py
```

Key options: `--per-stratum` (default 4), `--seed`, `--output`.

## Ablation study

```bash
# Print rendered prompts for each config (no API calls)
uv run python scripts/run_ablations.py \
  --input data/nts_attributes_2023.csv \
  --n-samples 5 \
  --dry-run

# Run all 8 configs on 500 schedules each
uv run python scripts/run_ablations.py \
  --input data/nts_attributes_2023.csv \
  --n-samples 500
```

Results are written to `outputs/ablations/<config>/schedules.jsonl` and `schedules.csv` per config, plus a summary at `outputs/ablations/summary.csv`.

| Config | Description |
|--------|-------------|
| `zero_shot` | Baseline — no examples, temperature 0.8 |
| `few_shot_random` | N randomly selected examples |
| `few_shot_stratified` | N examples matched on attributes |
| `few_shot_nearest` | N examples based on closeness to attributes |
| `cot_zero_shot` | Chain-of-thought reasoning, no examples |
| `cot_few_shot` | CoT + stratified examples |
| `high_temp` | temperature=1.1 — diversity test |
| `low_temp` | temperature=0.3 — fidelity test |

## Configuration

```yaml
model:
  provider: anthropic       # anthropic | local
  name: claude-sonnet-4-6   # model name or HuggingFace model ID
  temperature: 0.7
  max_tokens: 1024
  max_retries: 2

concurrency: 5  # parallel API calls; set to 1 for local inference
batch_size: 128
```

The `local` provider loads any HuggingFace text-generation model onto the GPU. Set `adapter_path` to load a LoRA adapter on top of the base model (see Fine-tuning above).

## Tests

```bash
uv run python -m pytest tests/ -v
```
