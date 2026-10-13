# experiment_registry.py
"""
Registry of experiment configurations for FiNE-style experiments.

This is a small convenience layer used by run_experiment.py.
It maps an integer exp_code to an ExperimentConfig object.

Currently we define:

  100: E1 - Overlap Dataset (build Set-Poly / Set-Mono)
  210: E2a - Locality on Set-Poly
  300: E3a - Locality on Set-Mono
  410: A1  - Ablation (sensitivity to top-k / Jaccard), skeleton only
  500: A2  - Global locality, skeleton only
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExperimentConfig:
    # Basic meta-info
    code: int
    name: str
    description: str

    # Which driver in run_experiment.py to call
    #   "overlap_dataset" | "locality" | "ablation_sensitivity" | "global_locality"
    run_type: str

    # ---- Fields for overlap_dataset (E1) ----
    topk_neurons: int = 64          # k in Algorithm 1 (per-token top-k neurons)
    jaccard_high: float = 0.1       # threshold for Set-Poly
    jaccard_low: float = 0.01        # threshold for Set-Mono

    # ---- Fields for locality (E2/E3) ----
    dataset_set: str = "poly"       # "poly" | "mono" | "both"
    methods: List[str] = field(default_factory=lambda: ["fine", "rome"])

    # ---- Extra kwargs for ablations, etc. ----
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


# ===================== Define configurations =====================

_EXPERIMENTS: Dict[int, ExperimentConfig] = {}


# E1: Overlap dataset (build Set-Poly / Set-Mono)
_EXPERIMENTS[100] = ExperimentConfig(
    code=100,
    name="E1 - Overlap Dataset (Set-Poly / Set-Mono)",
    description=(
        "Construct Set-Poly and Set-Mono based on neuron-level Jaccard "
        "overlap using FiNE's Eq.(3) neuron localization."
    ),
    run_type="overlap_dataset",
    topk_neurons=64,        # you can tune this, e.g. 32 / 64 / 128
    jaccard_high=0.1,       # high-overlap threshold for Set-Poly
    jaccard_low=0.01,        # low-overlap threshold for Set-Mono
)


# E2a: Locality on Set-Poly (e.g., compare FiNE vs ROME)
_EXPERIMENTS[210] = ExperimentConfig(
    code=210,
    name="E2a - Locality on Set-Poly",
    description=(
        "Evaluate editing locality on the Set-Poly pairs: "
        "edit fact A, measure success on A and locality on B."
    ),
    run_type="locality",
    dataset_set="poly",          # use only Set-Poly
    methods=["fine", "rome"],    # edit methods to compare
)


# E3a: Locality on Set-Mono
_EXPERIMENTS[300] = ExperimentConfig(
    code=300,
    name="E3a - Locality on Set-Mono",
    description=(
        "Evaluate editing locality on the Set-Mono pairs: "
        "edit fact A, measure success on A and locality on B."
    ),
    run_type="locality",
    dataset_set="mono",          # use only Set-Mono
    methods=["fine", "rome"],
)


# A1: Ablation - sensitivity to top-k and Jaccard thresholds (skeleton)
_EXPERIMENTS[410] = ExperimentConfig(
    code=410,
    name="A1 - Ablation: Sensitivity to top-k / Jaccard",
    description=(
        "Ablation experiment skeleton: sweep over different top-k values "
        "and Jaccard high thresholds to study overlap sensitivity."
    ),
    run_type="ablation_sensitivity",
    topk_neurons=64,
    jaccard_high=0.1,
    jaccard_low=0.01,
    extra_kwargs={
        # example sweeps; you can change these
        "topk_list": [32, 64, 128],
        "jaccard_high_list": [0.3, 0.5, 0.7],
    },
)


# A2: Global locality (skeleton)
_EXPERIMENTS[500] = ExperimentConfig(
    code=500,
    name="A2 - Global Locality (zsRE / LM perplexity)",
    description=(
        "Global locality experiment skeleton: evaluate edited models on "
        "a broader QA / LM corpus (e.g., zsRE or general text) to measure "
        "global accuracy and perplexity changes."
    ),
    run_type="global_locality",
)


# ===================== Public API =====================

def get_experiment_config(code: int) -> ExperimentConfig:
    """
    Return the ExperimentConfig for a given exp_code.

    Example:
        cfg = get_experiment_config(100)
        print(cfg.run_type)  # "overlap_dataset"
    """
    if code not in _EXPERIMENTS:
        raise ValueError(
            f"Unknown exp_code={code}. "
            f"Available codes: {sorted(_EXPERIMENTS.keys())}"
        )
    return _EXPERIMENTS[code]
