# actllm

LLM-based 24-hour activity schedule generation conditioned on person-level attributes. Produces NTS-compatible output for evaluation against ActVAE.

## Setup

### Anthropic (default)

```bash
uv sync --extra dev
cp .env.example .env  # add your ANTHROPIC_API_KEY
```

```
ANTHROPIC_API_KEY=sk-ant-...
```

Note that `actllm` will always check .env for the key.

### Local model (GPU)

Requires a CUDA-capable GPU. Installs `transformers` and `accelerate` alongside the base dependencies:

```bash
uv sync --extra local
```

The model is downloaded from HuggingFace on first use (~8 GB for Gemma 3 4B). No API key needed.

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

`--output` always produces two files: the JSONL at the given path and an NTS-compatible CSV at the same path with a `.csv` extension (e.g. `outputs/run.csv`).

Output is streamed to JSONL as it's generated — safe to interrupt.

Pass `--config configs/ablations/few_shot_stratified.yaml` (or any other config) to use a different prompt mode.

### Running with a local Gemma model

```bash
# Generate 100 schedules using Gemma 3 4B on GPU
uv run python scripts/generate.py \
  --config configs/gemma.yaml \
  --input data/nts_attributes_2023.csv \
  --output outputs/gemma_run.jsonl \
  --n-samples 100
```

The first run downloads the model weights. Subsequent runs load from the HuggingFace cache. Inference is sequential (`concurrency: 1`) — expect slower throughput than the Anthropic API but no per-token cost.

### Inspecting prompts

```bash
# Print the prompt for the first record without calling the API
uv run python scripts/generate.py \
  --input data/nts_attributes_2023.csv \
  --output outputs/run.jsonl \
  --config configs/ablations/few_shot_stratified.yaml \
  --dry-run

# Log full prompts (including rendered few-shot examples) during a live run
uv run python scripts/generate.py \
  --input data/nts_attributes_2023.csv \
  --output outputs/run.jsonl \
  --n-samples 5 \
  --debug-prompts
```

`--debug-prompts` prints every system and user prompt to stderr before each API call, including retry attempts. Output is clean (no timestamp noise) and independent of `--verbose`.

## Prompt modes

| Mode | Description |
|------|-------------|
| `zero_shot` | No examples; JSON-only output |
| `few_shot` | 3 real NTS examples prepended; selection strategy configurable |
| `cot_zero_shot` | Model reasons in `<reasoning>` block then outputs JSON in `<schedule>` block |
| `cot_few_shot` | CoT + few-shot examples |

Configure via `configs/default.yaml` or any ablation config:

```yaml
prompt:
  mode: few_shot            # zero_shot | few_shot | cot_zero_shot | cot_few_shot
  n_examples: 3
  example_selection: stratified   # random | stratified | nearest_neighbour
  few_shot_pool: data/few_shot/pool.jsonl
```

## Few-shot pool

The pool (`data/few_shot/pool.jsonl`) contains 58 real NTS schedules sampled across 15 strata (5 age bands × 3 work statuses). To regenerate it:

```bash
# Preview strata analysis without writing
uv run python scripts/curate_few_shot.py --dry-run

# Write the pool
uv run python scripts/curate_few_shot.py
```

Key options: `--per-stratum` (default 4), `--seed`, `--output`.

## Ablation study

```bash
# Print rendered prompts for each config (same person, no API calls)
uv run python scripts/run_ablations.py \
  --input data/nts_attributes_2023.csv \
  --n-samples 5 \
  --dry-run

# Run all 7 configs on 500 schedules each
uv run python scripts/run_ablations.py \
  --input data/nts_attributes_2023.csv \
  --n-samples 500
```

Results are written to `outputs/ablations/<config>/schedules.jsonl` and `schedules.csv` per config, plus a summary table at `outputs/ablations/summary.csv`. Add `--debug-prompts` to log prompts during the run.

| Config | Description |
|--------|-------------|
| `zero_shot` | Baseline — no examples, temperature 0.8 |
| `few_shot_random` | 3 randomly selected examples |
| `few_shot_stratified` | 3 examples matched on age band and work status |
| `cot_zero_shot` | Chain-of-thought reasoning, no examples |
| `cot_few_shot` | CoT + stratified examples |
| `high_temp` | temperature=1.1 — diversity test |
| `low_temp` | temperature=0.3 — fidelity test |

## Configuration

```yaml
model:
  provider: anthropic       # anthropic | local
  name: claude-sonnet-4-6   # model name or HuggingFace model ID
  temperature: 0.8
  max_retries: 3

concurrency: 5  # parallel API calls; set to 1 for local inference
```

The `local` provider loads any HuggingFace text-generation model onto the GPU. The bundled `configs/gemma.yaml` targets `google/gemma-3-4b-it`; swap `name` for any other Gemma variant (e.g. `google/gemma-2-9b-it`).

## Tests

```bash
uv run python -m pytest tests/ -v
```
