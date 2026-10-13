"""Validate torch/TensorBoard-aware training protocol settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import TrainingProtocol, get_torch_device, set_global_seed, validate_training_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate training protocol settings.")
    parser.add_argument("--scheme", default="expanding", choices=["expanding", "rolling"])
    parser.add_argument("--initial-train-months", default=60, type=int)
    parser.add_argument("--validation-months", default=12, type=int)
    parser.add_argument("--refit-frequency-months", default=1, type=int)
    parser.add_argument("--seed", default=20260408, type=int)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = TrainingProtocol(
        scheme=args.scheme,
        initial_train_months=args.initial_train_months,
        validation_months=args.validation_months,
        refit_frequency_months=args.refit_frequency_months,
        seed=args.seed,
    )
    validate_training_protocol(protocol)
    set_global_seed(protocol.seed)
    device = get_torch_device(prefer_cuda=not args.cpu)
    print(f"Training protocol is valid: {protocol}")
    print(f"Torch device selected: {device}")
    print("Actual training is reserved for the modeling stage.")


if __name__ == "__main__":
    main()
