"""Label-generation helpers."""

from funding_arb.labels.generator import assign_time_series_split, build_label_table, describe_labeling_assumption, forward_window_sum
from funding_arb.labels.pipeline import LabelPipelineArtifacts, build_supervised_dataset, describe_supervised_dataset_job, run_label_pipeline

__all__ = [
    "LabelPipelineArtifacts",
    "assign_time_series_split",
    "build_label_table",
    "build_supervised_dataset",
    "describe_labeling_assumption",
    "describe_supervised_dataset_job",
    "forward_window_sum",
    "run_label_pipeline",
]