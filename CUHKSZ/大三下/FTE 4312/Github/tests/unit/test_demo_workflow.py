from __future__ import annotations

from funding_arb.config.loader import load_command_settings
from funding_arb.demo.workflow import build_stage_plan, describe_demo_workflow_job


def test_build_stage_plan_contains_core_demo_stages_in_order() -> None:
    config = load_command_settings("run-demo")
    stage_plan = build_stage_plan(config)
    stage_keys = [stage.key for stage in stage_plan]
    assert stage_keys[:5] == [
        "fetch_data",
        "report_data_quality",
        "build_features",
        "build_labels",
        "train_baseline",
    ]
    assert "compare_deep_learning" in stage_keys
    assert stage_keys.index("compare_deep_learning") > stage_keys.index("train_deep_learning")
    assert stage_keys.index("compare_deep_learning") < stage_keys.index("generate_deep_learning_signals")
    assert stage_keys[-2:] == ["sync_vault", "export_demo_snapshot"]


def test_build_stage_plan_marks_deep_learning_stages_optional() -> None:
    config = load_command_settings("run-demo")
    stage_plan = build_stage_plan(config)
    stage_lookup = {stage.key: stage for stage in stage_plan}
    assert stage_lookup["train_deep_learning"].optional is True
    assert stage_lookup["compare_deep_learning"].optional is True
    assert stage_lookup["generate_deep_learning_signals"].optional is True
    assert "--source" in stage_lookup["generate_baseline_signals"].command
    assert "baseline" in stage_lookup["generate_baseline_signals"].command
    assert "dl" in stage_lookup["generate_deep_learning_signals"].command


def test_describe_demo_workflow_mentions_run_name_and_stage_count() -> None:
    config = load_command_settings("run-demo")
    description = describe_demo_workflow_job(config)
    assert "full_demo_default" in description
    assert "12 stages" in description


def test_exploratory_demo_workflow_focuses_on_showcase_stages() -> None:
    config = load_command_settings("run-exploratory-dl-demo")
    stage_plan = build_stage_plan(config)
    stage_keys = [stage.key for stage in stage_plan]

    assert stage_keys == [
        "build_exploratory_dataset",
        "compare_exploratory_deep_learning",
        "compare_exploratory_direction",
        "generate_exploratory_signals",
        "backtest_exploratory",
        "report_exploratory",
        "export_demo_snapshot",
    ]
