"""Command-line interface for inference and reproducible model development."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pdp_uq.config import load_config
from pdp_uq.data import fetch_dataset, validate_dataset_artifact
from pdp_uq.evaluation import evaluate_models
from pdp_uq.exceptions import PdpUqError
from pdp_uq.inference import PredictionConfig, predict_file
from pdp_uq.io import write_json
from pdp_uq.plotting import plot_profile_file
from pdp_uq.training import (
    create_split,
    prepare_training_data,
    record_existing_models,
    train_release_models,
    tune_models,
)

LOGGER = logging.getLogger("pdp_uq")
DEFAULT_CONFIG = Path("configs/model.toml")
DEFAULT_MANIFEST = Path("artifacts/manifest.json")


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Reproducibility configuration (default: configs/model.toml).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""
    parser = argparse.ArgumentParser(
        prog="pdp-uq",
        description="Bias correction and uncertainty quantification for phase-detection probes.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging.")
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict", help="Correct measurements in a CSV file.")
    predict.add_argument("input", type=Path)
    predict.add_argument("--output", type=Path)
    predict.add_argument("--dx", type=float, required=True, help="Streamwise tip separation [m].")
    predict.add_argument("--dy", type=float, required=True, help="Lateral tip separation [m].")
    predict.add_argument(
        "--particles-per-window",
        "--Np",
        dest="particles_per_window",
        type=int,
        required=True,
    )
    predict.add_argument("--model-dir", type=Path, default=Path("data"))
    predict.add_argument("--metadata", type=Path)
    _common_config(predict)

    data = commands.add_parser("data", help="Acquire and prepare published data.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch = data_commands.add_parser("fetch", help="Fetch the immutable DOI dataset.")
    fetch.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fetch.add_argument("--output", type=Path)

    validate = data_commands.add_parser("validate", help="Validate the published artifact.")
    validate.add_argument("input", type=Path)
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    validate.add_argument("--report", type=Path)

    prepare = data_commands.add_parser("prepare", help="Normalize model-development data.")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--report", type=Path, required=True)
    _common_config(prepare)

    split = commands.add_parser("split", help="Persist the untouched train/test split.")
    split.add_argument("input", type=Path)
    split.add_argument("--output", type=Path, required=True)
    _common_config(split)

    tune = commands.add_parser("tune", help="Run randomized five-fold hyperparameter search.")
    tune.add_argument("input", type=Path)
    tune.add_argument("--split", type=Path, required=True)
    tune.add_argument("--output", type=Path, required=True)
    _common_config(tune)

    evaluate = commands.add_parser("evaluate", help="Evaluate on the untouched holdout.")
    evaluate.add_argument("input", type=Path)
    evaluate.add_argument("--split", type=Path, required=True)
    evaluate.add_argument("--tuning", type=Path)
    evaluate.add_argument("--metrics", type=Path, required=True)
    evaluate.add_argument("--figure", type=Path, required=True)
    _common_config(evaluate)

    train = commands.add_parser("train", help="Train final released models on all eligible rows.")
    train.add_argument("input", type=Path)
    train.add_argument("--tuning", type=Path)
    train.add_argument("--model-dir", type=Path, required=True)
    train.add_argument("--metadata", type=Path, required=True)
    _common_config(train)

    plot = commands.add_parser("plot", help="Plot a corrected measurement profile.")
    plot.add_argument("input", type=Path)
    plot.add_argument("--output", type=Path)

    record = commands.add_parser(
        "record-models", help="Checksum existing trusted release model artifacts."
    )
    record.add_argument("--model-dir", type=Path, default=Path("data"))
    record.add_argument("--training-data", type=Path, default=Path("data/simulation_results.csv"))
    record.add_argument("--output", type=Path, default=Path("data/model_metadata.json"))

    return parser


def _render(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def run(args: argparse.Namespace) -> None:
    """Execute a parsed CLI command."""
    if args.command == "predict":
        project = load_config(args.config)
        destination = predict_file(
            args.input,
            PredictionConfig(
                delta_x=args.dx,
                delta_y=args.dy,
                particles_per_window=args.particles_per_window,
                quantiles=project.quantiles,
            ),
            model_directory=args.model_dir,
            output_path=args.output,
            metadata_path=args.metadata,
        )
        _render({"output": str(destination)})
        return

    if args.command == "data" and args.data_command == "fetch":
        destination = fetch_dataset(args.manifest, args.output)
        _render({"output": str(destination)})
        return

    if args.command == "data" and args.data_command == "validate":
        report = validate_dataset_artifact(args.input, args.manifest)
        if args.report:
            write_json(args.report, report)
        _render(report)
        return

    if args.command == "data" and args.data_command == "prepare":
        _render(
            prepare_training_data(
                args.input,
                args.output,
                args.report,
                load_config(args.config),
            )
        )
        return

    if args.command == "split":
        split = create_split(args.input, args.output, load_config(args.config))
        _render(
            {
                "output": str(args.output),
                "training_rows": len(split["train_ids"]),
                "test_rows": len(split["test_ids"]),
                "random_seed": split["random_seed"],
            }
        )
        return

    if args.command == "tune":
        _render(
            tune_models(
                args.input,
                args.split,
                args.output,
                load_config(args.config),
            )
        )
        return

    if args.command == "evaluate":
        _render(
            evaluate_models(
                args.input,
                args.split,
                args.metrics,
                args.figure,
                load_config(args.config),
                tuning_path=args.tuning,
            )
        )
        return

    if args.command == "train":
        _render(
            train_release_models(
                args.input,
                args.model_dir,
                args.metadata,
                load_config(args.config),
                tuning_path=args.tuning,
            )
        )
        return

    if args.command == "plot":
        _render({"output": str(plot_profile_file(args.input, args.output))})
        return

    if args.command == "record-models":
        _render(
            record_existing_models(
                args.model_dir,
                args.output,
                training_data_path=args.training_data,
            )
        )
        return

    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        run(args)
        return 0
    except (PdpUqError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
