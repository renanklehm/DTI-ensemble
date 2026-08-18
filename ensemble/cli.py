from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config
from .logging_utils import configure_logging
from .pipeline import predict_pipeline, prepare_frame, train_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ChemBERTa + ESM-2 multimodal DTI ensemble"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Align data, embed entities, and train the ensemble")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--resume", help="Path to a last.pt checkpoint")

    inspect = commands.add_parser("inspect-data", help="Validate pairing and print split statistics")
    inspect.add_argument("--config", default="configs/default.yaml")
    inspect.add_argument("--output", help="Optional path for the paired parquet")

    predict = commands.add_parser("predict", help="Run a trained ensemble on long or wide input")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", help="Override auto/cpu/cuda from the saved config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_dir = train_pipeline(load_config(args.config), args.resume)
        print(run_dir)
    elif args.command == "inspect-data":
        config = load_config(args.config)
        configure_logging(Path(config.runtime.output_dir) / "inspection", config.runtime.log_level)
        frame = prepare_frame(config)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(args.output, index=False)
        summary = frame.groupby("split")["target"].agg(["count", "mean"]).to_dict("index")
        print(json.dumps(summary, indent=2))
    else:
        predict_pipeline(args.checkpoint, args.input, args.output, args.device)
