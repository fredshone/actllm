"""Run ablation study across all prompt config variants.

For each config in configs/ablations/, generates N schedules from a shared
sample of NTS attributes, then prints a summary comparison table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="Run prompt ablation study.")

ABLATION_CONFIGS = [
    "zero_shot.yaml",
    "few_shot_random.yaml",
    "few_shot_stratified.yaml",
    "few_shot_nearest.yaml",
    # "cot_zero_shot.yaml",
    # "cot_few_shot.yaml",
    "high_temp.yaml",
    "low_temp.yaml",
]


def _setup_logging(verbose: bool, debug_prompts: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    if debug_prompts:
        pl = logging.getLogger("actllm.prompts")
        pl.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pl.addHandler(handler)
        pl.propagate = False


@app.command()
def main(
    input: Path = typer.Option(..., "--input", "-i", help="Path to nts_attributes CSV"),
    output_dir: Path = typer.Option(
        Path("outputs/ablations"),
        "--output-dir",
        "-o",
        help="Root directory for all outputs",
    ),
    configs_dir: Path = typer.Option(
        Path("configs/ablations"),
        "--configs-dir",
        help="Directory containing ablation YAMLs",
    ),
    n_samples: int = typer.Option(
        500, "--n-samples", "-n", help="Schedules per config"
    ),
    configs: list[str] = typer.Option(
        [], "--config", "-c", help="Run only these config filenames (default: all)"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print rendered prompts for each config without calling the API",
    ),
    debug_prompts: bool = typer.Option(
        False, "--debug-prompts", help="Log full prompts (including few-shot) to stderr"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _setup_logging(verbose, debug_prompts)

    if not input.exists():
        typer.echo(f"Error: input file not found: {input}", err=True)
        raise typer.Exit(1)

    selected = list(configs) if configs else ABLATION_CONFIGS

    df = pd.read_csv(input)
    sample_size = min(n_samples, len(df))
    typer.echo(f"Sample size: {sample_size} persons")

    records_df = df.sample(n=sample_size, random_state=42)
    records = records_df.to_dict(orient="records")

    if dry_run:
        import yaml
        from actllm.prompt.builder import PromptBuilder
        from actllm.prompt.few_shot import FewShotSelector

        first = records[0]
        for config_name in selected:
            config_path = configs_dir / config_name
            if not config_path.exists():
                typer.echo(f"[skip] config not found: {config_path}", err=True)
                continue
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            prompt_cfg = cfg.get("prompt", {})
            mode = prompt_cfg.get("mode", "zero_shot")
            n_examples = prompt_cfg.get("n_examples", 0)
            few_shot_selector = None
            if mode in ("few_shot", "cot_few_shot") and n_examples > 0:
                pool_path = Path(
                    prompt_cfg.get("few_shot_pool", "data/few_shot/pool.jsonl")
                )
                few_shot_selector = FewShotSelector(pool_path)
            builder = PromptBuilder(
                cfg["attributes"]["fields"],
                mode=mode,
                few_shot_selector=few_shot_selector,
                n_examples=n_examples,
                example_selection=prompt_cfg.get("example_selection", "stratified"),
            )
            attrs = {field: first.get(field) for field in cfg["attributes"]["fields"]}
            sep = "=" * 72
            typer.echo(f"\n{sep}")
            typer.echo(f"CONFIG: {config_name}")
            typer.echo(sep)
            typer.echo("--- SYSTEM ---")
            typer.echo(builder.system_prompt)
            typer.echo("--- USER ---")
            typer.echo(builder.build(attrs))
        return
    results: list[dict] = []

    from actllm.pipeline import ScheduleGenerator

    for config_name in selected:
        config_path = configs_dir / config_name
        if not config_path.exists():
            typer.echo(f"  [skip] config not found: {config_path}", err=True)
            continue

        run_dir = output_dir / config_name.replace(".yaml", "")
        run_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = run_dir / "schedules.jsonl"
        csv_path = jsonl_path.with_suffix(".csv")

        # save records for reproducibility
        records_path = run_dir / "attributes.csv"
        records_df.to_csv(records_path, index=False)

        typer.echo(f"\n→ {config_name}")
        try:
            generator = ScheduleGenerator(config_path)
            stats = generator.generate_batch(records, jsonl_path, csv_path=csv_path)
        except Exception as exc:
            typer.echo(f"  ERROR: {exc}", err=True)
            results.append({"config": config_name, "error": str(exc)})
            continue

        row = {
            "config": config_name.replace(".yaml", ""),
            "n_valid": stats["n_valid"],
            "validity_rate": f"{stats['validity_rate']:.1%}",
            "parse_success": f"{stats['parse_success_rate']:.1%}",
            "n_invalid": stats["n_invalid"],
            "parse_failures": stats["parse_failures"],
        }
        results.append(row)
        typer.echo(
            f"  valid={stats['n_valid']}/{stats['total']} "
            f"({stats['validity_rate']:.1%}), "
            f"parse_ok={stats['parse_success_rate']:.1%}"
        )

    typer.echo("\n" + "=" * 72)
    typer.echo("ABLATION SUMMARY")
    typer.echo("=" * 72)
    if results:
        result_df = pd.DataFrame(results)
        typer.echo(result_df.to_string(index=False))

    summary_path = output_dir / "summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(summary_path, index=False)
    typer.echo(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    app()
