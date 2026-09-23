"""Kiểm thử temperature scaling và hàng rào chống test leakage."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.evaluation.calibration import (
    MulticlassTemperatureCalibrator,
    fit_router_calibrators,
    load_calibration_models,
    write_calibration_models,
)
from adaptive_rag_router.evaluation.models import BenchmarkRecord, DatasetSplit


def _overconfident_probabilities() -> list[dict[str, float]]:
    return [
        {"none": 0.99, "web": 0.01},
        {"none": 0.99, "web": 0.01},
        {"none": 0.90, "web": 0.10},
        {"none": 0.90, "web": 0.10},
    ]


def test_temperature_scaler_fits_only_calibration_and_versions_artifact() -> None:
    calibrator = MulticlassTemperatureCalibrator()
    model = calibrator.fit(
        _overconfident_probabilities(),
        ["none", "web", "none", "web"],
        split=DatasetSplit.CALIBRATION,
    )

    assert model.calibrator_version == "temperature-scaling-v1"
    assert model.fitted_split == DatasetSplit.CALIBRATION
    assert model.temperature > 1.0
    assert model.negative_log_likelihood_after < model.negative_log_likelihood_before
    transformed = calibrator.transform([{"none": 0.9, "web": 0.1}])[0]
    assert sum(transformed.values()) == pytest.approx(1.0)
    assert transformed["web"] > 0.1


@pytest.mark.parametrize("split", [DatasetSplit.DEV, DatasetSplit.TEST])
def test_temperature_scaler_rejects_non_calibration_split(split: DatasetSplit) -> None:
    calibrator = MulticlassTemperatureCalibrator()

    with pytest.raises(ValueError, match="calibration split"):
        calibrator.fit(
            _overconfident_probabilities(),
            ["none", "web", "none", "web"],
            split=split,
        )


def test_temperature_scaler_requires_fit_before_transform() -> None:
    calibrator = MulticlassTemperatureCalibrator()

    with pytest.raises(RuntimeError, match="chưa được fit"):
        calibrator.transform([{"none": 1.0}])


def test_router_calibration_artifacts_round_trip(tmp_path: Path) -> None:
    records = [
        BenchmarkRecord(
            run_id="calibration",
            query_id=f"{router.value}-{index}",
            dataset="fixture",
            stratum="general",
            group_id=f"g-{router.value}-{index}",
            router=router,
            expected_route=expected,
            predicted_route="none:economy",
            route_probabilities={"none:economy": 0.9, "web:strong": 0.1},
            status=CompletionStatus.COMPLETED,
        )
        for router in (RouterKind.JEV, RouterKind.LLM)
        for index, expected in enumerate(
            ("none:economy", "web:strong", "none:economy", "web:strong")
        )
    ]

    models = fit_router_calibrators(records, (RouterKind.JEV, RouterKind.LLM))
    artifact = tmp_path / "calibration-models.json"
    write_calibration_models(artifact, models)
    loaded = load_calibration_models(artifact, (RouterKind.JEV, RouterKind.LLM))

    assert set(loaded) == {RouterKind.JEV, RouterKind.LLM}
    assert loaded[RouterKind.JEV].fitted_split is DatasetSplit.CALIBRATION
    assert loaded[RouterKind.LLM].temperature > 1.0
