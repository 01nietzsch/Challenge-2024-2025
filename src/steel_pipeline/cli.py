from __future__ import annotations

import argparse

from .optimize import run_optimization
from .preprocess import run_preprocessing
from .train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Steel design pipeline entry point")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess", help="Build normalized, imputed, and final datasets")
    sub.add_parser("train", help="Train regressors/classifier and save artifacts")

    opt = sub.add_parser("optimize", help="Generate optimized candidate alloys")
    opt.add_argument("--samples", type=int, default=1000, help="Number of random compositions")

    sub.add_parser("all", help="Run preprocessing, training, and optimization")

    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocessing()
    elif args.command == "train":
        run_training()
    elif args.command == "optimize":
        run_optimization(n_samples=args.samples)
    elif args.command == "all":
        run_preprocessing()
        run_training()
        run_optimization()


if __name__ == "__main__":
    main()
