from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dl_for_cta.config.loader import load_config
from dl_for_cta.data.minute_loader import inspect_minute_data
from dl_for_cta.experiments import run_backtest, run_cpd, run_features, run_pipeline, run_train
from dl_for_cta.utils.logging import configure_logging


logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dl-for-cta")
    parser.add_argument(
        "command",
        choices=["inspect-data", "build-features", "build-cpd", "train", "backtest", "run-pipeline"],
    )
    parser.add_argument("--config", required=True, help="Path to TOML experiment config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parser().parse_args(argv)
    logger.info("Starting command=%s config=%s", args.command, args.config)
    config = load_config(args.config)
    Path(config.outputs.root).mkdir(parents=True, exist_ok=True)
    Path("cache").mkdir(parents=True, exist_ok=True)

    if args.command == "inspect-data":
        print(inspect_minute_data(config.data).to_string(index=False))
    elif args.command == "build-features":
        print(run_features.run(config))
    elif args.command == "build-cpd":
        print(json.dumps(run_cpd.run(config), indent=2))
    elif args.command == "train":
        print(run_train.run(config))
    elif args.command == "backtest":
        print(run_backtest.run(config))
    elif args.command == "run-pipeline":
        print(json.dumps(run_pipeline.run(config), indent=2))
    else:
        raise AssertionError(args.command)
    logger.info("Finished command=%s", args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
