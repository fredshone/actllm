"""CLI for batch schedule generation from NTS attribute data."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="Generate activity schedules using an LLM.")


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
    output: Path = typer.Option(..., "--output", "-o", help="Output JSONL path (CSV written alongside)"),
    config: Path = typer.Option(
        Path("configs/default.yaml"), "--config", "-c", help="Pipeline config YAML"
    ),
    n_samples: int = typer.Option(10, "--n-samples", "-n", help="Number of schedules to generate"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print prompt only, no API call"),
    debug_prompts: bool = typer.Option(
        False, "--debug-prompts", help="Log full prompts (including few-shot) to stderr"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _setup_logging(verbose, debug_prompts)
    logger = logging.getLogger(__name__)

    if not input.exists():
        typer.echo(f"Error: input file not found: {input}", err=True)
        raise typer.Exit(1)
    if not config.exists():
        typer.echo(f"Error: config file not found: {config}", err=True)
        raise typer.Exit(1)

    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(input)
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    records = sample.to_dict(orient="records")

    if dry_run:
        from actllm.prompt.builder import PromptBuilder
        from actllm.prompt.few_shot import FewShotSelector
        prompt_cfg = cfg.get("prompt", {})
        mode = prompt_cfg.get("mode", "zero_shot")
        n_examples = prompt_cfg.get("n_examples", 0)
        few_shot_selector = None
        if mode in ("few_shot", "cot_few_shot") and n_examples > 0:
            pool_path = Path(prompt_cfg.get("few_shot_pool", "data/few_shot/pool.jsonl"))
            few_shot_selector = FewShotSelector(pool_path)
        builder = PromptBuilder(
            cfg["attributes"]["fields"],
            mode=mode,
            few_shot_selector=few_shot_selector,
            n_examples=n_examples,
            example_selection=prompt_cfg.get("example_selection", "stratified"),
        )
        attrs = {field: records[0].get(field) for field in cfg["attributes"]["fields"]}
        typer.echo("=== SYSTEM PROMPT ===")
        typer.echo(builder.system_prompt)
        typer.echo("\n=== USER PROMPT ===")
        typer.echo(builder.build(attrs))
        return

    csv_path = output.with_suffix(".csv")

    from actllm.pipeline import ScheduleGenerator
    generator = ScheduleGenerator(config)

    logger.info("Generating %d schedules → %s", len(records), output)
    stats = generator.generate_batch(records, output, csv_path=csv_path)

    typer.echo("\n=== Generation Summary ===")
    typer.echo(f"Total:              {stats['total']}")
    typer.echo(f"Valid:              {stats['n_valid']} ({stats['validity_rate']:.1%})")
    typer.echo(f"Invalid:            {stats['n_invalid']}")
    typer.echo(f"Parse failures:     {stats['parse_failures']}")
    typer.echo(f"Parse success rate: {stats['parse_success_rate']:.1%}")
    if stats["violation_counts"]:
        typer.echo("Violation breakdown:")
        for k, v in sorted(stats["violation_counts"].items()):
            typer.echo(f"  {k}: {v}")
    typer.echo(f"\nOutput written to: {output}")
    typer.echo(f"CSV written to:    {csv_path}")


if __name__ == "__main__":
    app()
