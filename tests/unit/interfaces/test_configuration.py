"""Kiểm thử parser TOML và guardrail benchmark mà không gọi provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_rag_router.domain.errors import ConfigurationError
from adaptive_rag_router.interfaces.configuration import load_benchmark_execution_plan


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _formal_project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    rag_path = tmp_path / "data" / "rag.jsonl"
    crag_path = tmp_path / "data" / "crag.jsonl"
    mock_path = tmp_path / "data" / "mock.json"
    _write_jsonl(
        rag_path,
        [
            {
                "query_id": f"rag-{index}",
                "query": f"RAG query {index}",
                "group_id": f"rag-group-{index}",
                "domain": "general",
                "query_type": "fact",
                "answer": f"answer {index}",
            }
            for index in range(700)
        ],
    )
    _write_jsonl(
        crag_path,
        [
            {
                "interaction_id": f"crag-{index}",
                "query": f"CRAG query {index}",
                "group_id": f"crag-group-{index}",
                "domain": "general",
                "category": "fact",
                "popularity": "head",
                "temporal_dynamism": "static",
                "answer": f"answer {index}",
            }
            for index in range(300)
        ],
    )
    mock_path.write_text("{}", encoding="utf-8")
    configs = tmp_path / "configs"
    configs.mkdir()
    formal = configs / "benchmark.toml"
    formal.write_text(
        """
[run]
name = "formal-test"
seed = 7
dev_size = 200
calibration_size = 200
test_size = 600
bootstrap_iterations = 100

[datasets]
ragrouter_path = "data/rag.jsonl"
ragrouter_sample_size = 700
crag_path = "data/crag.jsonl"
crag_sample_size = 300
mock_web_path = "data/mock.json"

[retrieval]
vector_top_k = 12
evidence_limit = 8
max_repair_rounds = 2
web_provider = "mock"

[generation]
max_output_tokens = 128
""".strip(),
        encoding="utf-8",
    )
    return formal


def test_load_formal_plan_builds_held_out_split(tmp_path: Path) -> None:
    formal = _formal_project(tmp_path)

    plan = load_benchmark_execution_plan(formal)

    assert plan.benchmark.run_id == "formal-test"
    assert len(plan.samples) == 600
    assert len(plan.calibration_samples) == 200
    assert plan.web_provider == "mock"
    assert plan.mock_web_path == (tmp_path / "data" / "mock.json").resolve()


def test_load_formal_plan_accepts_max_samples_for_smoke_run(tmp_path: Path) -> None:
    formal = _formal_project(tmp_path)
    content = formal.read_text(encoding="utf-8")
    formal.write_text(content.replace("seed = 7", "seed = 7\nmax_samples = 3"), encoding="utf-8")

    plan = load_benchmark_execution_plan(formal)

    assert plan.benchmark.max_samples == 3


def test_load_formal_plan_parses_typed_ablation_policy(tmp_path: Path) -> None:
    """Ablation switches phải được parse thành policy typed, mặc định còn lại bật."""

    formal = _formal_project(tmp_path)
    formal.write_text(
        formal.read_text(encoding="utf-8")
        + "\n[ablation]\nrepair_enabled = false\nstrong_fallback_enabled = false\n",
        encoding="utf-8",
    )

    plan = load_benchmark_execution_plan(formal)

    assert plan.pipeline_policy.context_gate_enabled is True
    assert plan.pipeline_policy.repair_enabled is False
    assert plan.pipeline_policy.strong_fallback_enabled is False
    assert plan.pipeline_policy.ablation_id == "no-repair-no-strong-fallback"


def test_load_formal_plan_reuses_frozen_gold_and_calibration_artifacts(tmp_path: Path) -> None:
    """Replay config resolve baseline artifacts mà không tạo gold/calibration mới."""

    formal = _formal_project(tmp_path)
    source = tmp_path / "artifacts" / "benchmarks" / "baseline-run"
    source.mkdir(parents=True)
    (source / "counterfactual.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "calibration-models.json").write_text("{}\n", encoding="utf-8")
    formal.write_text(
        formal.read_text(encoding="utf-8") + "\n[replay]\nsource_run_id = 'baseline-run'\n",
        encoding="utf-8",
    )

    plan = load_benchmark_execution_plan(formal)

    assert plan.replay_source_run_id == "baseline-run"
    assert plan.replay_counterfactual_path == source / "counterfactual.jsonl"
    assert plan.replay_calibration_path == source / "calibration-models.json"


def test_load_formal_plan_rejects_missing_replay_artifacts(tmp_path: Path) -> None:
    """Replay không được âm thầm sinh lại gold khi baseline artifact bị thiếu."""

    formal = _formal_project(tmp_path)
    formal.write_text(
        formal.read_text(encoding="utf-8") + "\n[replay]\nsource_run_id = 'missing-run'\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="thiếu frozen artifacts"):
        load_benchmark_execution_plan(formal)


def test_load_confirmatory_plan_excludes_pilot_queries_and_locks_looks(
    tmp_path: Path,
) -> None:
    formal = _formal_project(tmp_path)
    initial_plan = load_benchmark_execution_plan(formal)
    excluded_ids = [sample.query_id for sample in initial_plan.samples[:3]]
    source = tmp_path / "artifacts" / "benchmarks" / "pilot"
    source.mkdir(parents=True)
    _write_jsonl(source / "records.jsonl", [{"query_id": value} for value in excluded_ids])
    (source / "calibration-models.json").write_text("{}\n", encoding="utf-8")
    content = formal.read_text(encoding="utf-8").replace(
        "bootstrap_iterations = 100",
        "bootstrap_iterations = 100\nmax_samples = 10\nstratified_subset = true\n"
        "publish_acceptance = false\nmax_concurrency = 1\nrouters = ['jev', 'llm']",
    )
    content += """

[confirmatory]
enabled = true
looks = [4, 7, 10]
familywise_alpha = 0.05
calibration_source_run_id = "pilot"
excluded_run_ids = ["pilot"]
"""
    formal.write_text(content, encoding="utf-8")

    plan = load_benchmark_execution_plan(formal)

    assert plan.confirmatory_enabled is True
    assert plan.confirmatory_looks == (4, 7, 10)
    assert plan.benchmark.alternate_router_order is True
    assert len(plan.samples) == 10
    assert not (set(excluded_ids) & {sample.query_id for sample in plan.samples})


def test_load_budget_pilot_uses_stratified_subsets_and_disables_acceptance(
    tmp_path: Path,
) -> None:
    """Pilot giới hạn cả calibration/test và bắt buộc chạy tuần tự."""

    formal = _formal_project(tmp_path)
    content = formal.read_text(encoding="utf-8")
    content = content.replace(
        "seed = 7",
        "seed = 7\nmax_samples = 20\ncalibration_max_samples = 10\n"
        "stratified_subset = true\npublish_acceptance = false\nmax_concurrency = 1",
    )
    content += """

[cost_budget]
limit_usd = 2.79
provider_ledger_path = "artifacts/benchmarks/provider-cost-budget.json"
deepseek_limit_usd = 2.79
jev_limit_usd = 3.00
counterfactual_reserve_usd = 0.25
adaptive_reserve_usd = 0.50
"""
    formal.write_text(content, encoding="utf-8")

    plan = load_benchmark_execution_plan(formal)

    assert len(plan.samples) == 20
    assert len(plan.calibration_samples) == 10
    assert plan.cost_budget_usd == pytest.approx(2.79)
    assert plan.provider_budget_path == (
        tmp_path / "artifacts" / "benchmarks" / "provider-cost-budget.json"
    )
    assert plan.deepseek_limit_usd == pytest.approx(2.79)
    assert plan.jev_limit_usd == pytest.approx(3.0)
    assert plan.benchmark.publish_acceptance is False
    assert plan.benchmark.max_concurrency == 1


def test_cost_budget_rejects_concurrent_dispatch(tmp_path: Path) -> None:
    """Hard USD ledger không cho chạy nhiều invocation đồng thời."""

    formal = _formal_project(tmp_path)
    formal.write_text(
        formal.read_text(encoding="utf-8") + "\n[cost_budget]\nlimit_usd = 2.79\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="max_concurrency=1"):
        load_benchmark_execution_plan(formal)


def test_provider_budget_requires_path_and_both_limits(tmp_path: Path) -> None:
    """Không chấp nhận project hard cap chỉ cấu hình một phần."""

    formal = _formal_project(tmp_path)
    content = formal.read_text(encoding="utf-8").replace(
        "max_concurrency = 2",
        "max_concurrency = 1",
    )
    content += "\n[cost_budget]\nprovider_ledger_path = 'artifacts/providers.json'\n"
    formal.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="phải đi cùng nhau"):
        load_benchmark_execution_plan(formal)


def test_load_formal_plan_applies_preregistered_acceptance_thresholds(tmp_path: Path) -> None:
    formal = _formal_project(tmp_path)
    content = formal.read_text(encoding="utf-8")
    formal.write_text(
        content
        + """

[acceptance]
quality_non_inferiority_margin = 0.03
minimum_cost_reduction = 0.25
maximum_ece = 0.06
""",
        encoding="utf-8",
    )

    thresholds = load_benchmark_execution_plan(formal).benchmark.success_thresholds

    assert thresholds.quality_non_inferiority_margin == pytest.approx(0.03)
    assert thresholds.minimum_cost_reduction == pytest.approx(0.25)
    assert thresholds.maximum_ece == pytest.approx(0.06)
    assert thresholds.critical_recall == pytest.approx(0.90)


def test_load_live_plan_samples_formal_test_and_applies_budget(tmp_path: Path) -> None:
    _formal_project(tmp_path)
    live = tmp_path / "configs" / "live.toml"
    live.write_text(
        """
[run]
name = "live-test"
seed = 9
sample_size = 25

[retrieval]
web_provider = "tavily"
search_depth = "basic"
max_results = 5
max_repair_rounds = 2

[tavily]
requests_per_minute = 90
max_concurrency = 4
credit_budget = 350
max_retries = 4
""".strip(),
        encoding="utf-8",
    )

    plan = load_benchmark_execution_plan(live, live=True, tavily_credit_budget=17)

    assert len(plan.samples) == 25
    assert plan.benchmark.live is True
    assert plan.tavily_credit_budget == 17
    assert plan.web_provider == "tavily"
    assert (
        plan.calibration_artifact_path
        == (
            tmp_path / "artifacts" / "benchmarks" / "formal-test" / "calibration-models.json"
        ).resolve()
    )


def test_load_live_plan_applies_usd_budget_when_sequential(tmp_path: Path) -> None:
    """Live pilot nạp hard USD cap bên cạnh Tavily credit cap."""

    _formal_project(tmp_path)
    live = tmp_path / "configs" / "live.toml"
    live.write_text(
        """
[run]
name = "live-budget"
sample_size = 20
[retrieval]
web_provider = "tavily"
[tavily]
max_concurrency = 1
credit_budget = 180
[cost_budget]
limit_usd = 1.50
counterfactual_reserve_usd = 0.25
adaptive_reserve_usd = 0.50
""".strip(),
        encoding="utf-8",
    )

    plan = load_benchmark_execution_plan(live, live=True)

    assert plan.cost_budget_usd == pytest.approx(1.50)
    assert plan.counterfactual_reserve_usd == pytest.approx(0.25)
    assert plan.adaptive_reserve_usd == pytest.approx(0.50)


def test_load_live_plan_rejects_credit_above_hard_cap(tmp_path: Path) -> None:
    live = tmp_path / "live.toml"
    live.write_text(
        """
[run]
name = "live-test"
[retrieval]
web_provider = "tavily"
[tavily]
credit_budget = 350
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match=r"1\.\.350"):
        load_benchmark_execution_plan(live, live=True, tavily_credit_budget=351)


def test_formal_plan_reports_missing_dataset_clearly(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    config = tmp_path / "benchmark.toml"
    config.write_text(
        """
[run]
[datasets]
ragrouter_path = "missing/rag.jsonl"
crag_path = "missing/crag.jsonl"
mock_web_path = "missing/mock.jsonl"
[retrieval]
web_provider = "mock"
[generation]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Không tìm thấy dataset RAGRouter-Bench"):
        load_benchmark_execution_plan(config)
