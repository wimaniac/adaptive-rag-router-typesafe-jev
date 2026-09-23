"""Fit và áp dụng temperature scaling mà không làm rò rỉ held-out test."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from adaptive_rag_router.domain.enums import CompletionStatus, RouterKind
from adaptive_rag_router.evaluation.models import (
    BenchmarkRecord,
    DatasetSplit,
    TemperatureCalibrationModel,
)


def _validate_distributions(
    probabilities: Sequence[Mapping[str, float]],
    *,
    labels: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Kiểm tra distributions và trả label space ổn định."""

    if not probabilities:
        raise ValueError("probabilities không được rỗng")
    observed_labels = {label for distribution in probabilities for label in distribution}
    ordered_labels = tuple(labels) if labels is not None else tuple(sorted(observed_labels))
    if not ordered_labels or len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("labels phải không rỗng và không trùng lặp")
    if not observed_labels.issubset(ordered_labels):
        raise ValueError("distribution chứa label ngoài model")
    for distribution in probabilities:
        if not distribution:
            raise ValueError("probability distribution không được rỗng")
        values = tuple(float(value) for value in distribution.values())
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError("probability phải là số hữu hạn trong [0, 1]")
        if abs(sum(values) - 1.0) > 1e-3:
            raise ValueError("tổng probability phải bằng 1")
    return ordered_labels


def _scale_distribution(
    distribution: Mapping[str, float],
    labels: Sequence[str],
    temperature: float,
    epsilon: float,
) -> dict[str, float]:
    """Áp dụng softmax(log(p) / temperature) ổn định số."""

    scaled_logits = [
        math.log(max(float(distribution.get(label, 0.0)), epsilon)) / temperature
        for label in labels
    ]
    maximum = max(scaled_logits)
    exponentials = [math.exp(value - maximum) for value in scaled_logits]
    denominator = sum(exponentials)
    return {label: value / denominator for label, value in zip(labels, exponentials, strict=True)}


def _negative_log_likelihood(
    probabilities: Sequence[Mapping[str, float]],
    expected: Sequence[str],
    labels: Sequence[str],
    temperature: float,
    epsilon: float,
) -> float:
    """Tính mean NLL sau khi scale bằng một temperature ứng viên."""

    losses = []
    for distribution, actual in zip(probabilities, expected, strict=True):
        calibrated = _scale_distribution(distribution, labels, temperature, epsilon)
        losses.append(-math.log(max(calibrated[actual], epsilon)))
    return sum(losses) / len(losses)


class MulticlassTemperatureCalibrator:
    """Multiclass temperature scaler bắt buộc fit trên calibration split.

    Args:
        model: Artifact đã fit để khôi phục calibrator, nếu có.
        epsilon: Sàn xác suất dùng khi lấy logarithm.
        minimum_temperature: Cận dưới của lưới tìm kiếm log-scale.
        maximum_temperature: Cận trên của lưới tìm kiếm log-scale.
        search_steps: Số khoảng trong lưới tìm kiếm deterministic.
    """

    def __init__(
        self,
        model: TemperatureCalibrationModel | None = None,
        *,
        epsilon: float = 1e-12,
        minimum_temperature: float = 0.05,
        maximum_temperature: float = 20.0,
        search_steps: int = 400,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon phải dương")
        if minimum_temperature <= 0 or maximum_temperature <= minimum_temperature:
            raise ValueError("temperature bounds không hợp lệ")
        if search_steps < 10:
            raise ValueError("search_steps phải từ 10 trở lên")
        self._model = model
        self._epsilon = epsilon
        self._minimum_temperature = minimum_temperature
        self._maximum_temperature = maximum_temperature
        self._search_steps = search_steps

    @property
    def model(self) -> TemperatureCalibrationModel:
        """Trả artifact đã fit hoặc báo lỗi nếu calibrator chưa sẵn sàng."""

        if self._model is None:
            raise RuntimeError("calibrator chưa được fit")
        return self._model

    def fit(
        self,
        probabilities: Sequence[Mapping[str, float]],
        expected: Sequence[str],
        *,
        split: DatasetSplit,
    ) -> TemperatureCalibrationModel:
        """Fit scalar temperature, chỉ chấp nhận calibration split.

        Args:
            probabilities: Raw multiclass distributions.
            expected: Nhãn chuẩn theo cùng thứ tự.
            split: Split sinh ra các quan sát; bắt buộc là ``calibration``.

        Returns:
            TemperatureCalibrationModel chứa temperature và NLL trước/sau.

        Raises:
            ValueError: Khi split không phải calibration hoặc dữ liệu không hợp lệ.
        """

        if split != DatasetSplit.CALIBRATION:
            raise ValueError("temperature scaling chỉ được fit trên calibration split")
        if len(probabilities) != len(expected):
            raise ValueError("probabilities và expected phải có cùng độ dài")
        labels = _validate_distributions(probabilities)
        if not set(expected).issubset(labels):
            raise ValueError("distribution phải chứa mọi expected label")

        log_minimum = math.log(self._minimum_temperature)
        log_span = math.log(self._maximum_temperature) - log_minimum
        grid_candidates = (
            math.exp(log_minimum + log_span * index / self._search_steps)
            for index in range(self._search_steps + 1)
        )
        candidates = (*grid_candidates, 1.0)
        temperature, nll_after = min(
            (
                (
                    candidate,
                    _negative_log_likelihood(
                        probabilities,
                        expected,
                        labels,
                        candidate,
                        self._epsilon,
                    ),
                )
                for candidate in candidates
            ),
            key=lambda item: (item[1], abs(item[0] - 1.0)),
        )
        nll_before = _negative_log_likelihood(
            probabilities,
            expected,
            labels,
            1.0,
            self._epsilon,
        )
        self._model = TemperatureCalibrationModel(
            labels=labels,
            temperature=temperature,
            fitted_split=split,
            sample_count=len(expected),
            negative_log_likelihood_before=nll_before,
            negative_log_likelihood_after=nll_after,
        )
        return self._model

    def transform(
        self,
        probabilities: Sequence[Mapping[str, float]],
    ) -> tuple[dict[str, float], ...]:
        """Áp dụng temperature đã fit lên raw distributions mới.

        Args:
            probabilities: Distributions cần calibration, không dùng nhãn thật.

        Returns:
            Tuple distributions trên đúng label space của artifact.

        Raises:
            RuntimeError: Khi gọi trước ``fit`` hoặc chưa inject artifact.
            ValueError: Khi distribution không hợp lệ hay chứa label lạ.
        """

        model = self.model
        _validate_distributions(probabilities, labels=model.labels)
        return tuple(
            _scale_distribution(
                distribution,
                model.labels,
                model.temperature,
                self._epsilon,
            )
            for distribution in probabilities
        )


def fit_router_calibrators(
    records: Sequence[BenchmarkRecord],
    routers: Sequence[RouterKind],
) -> dict[RouterKind, TemperatureCalibrationModel]:
    """Fit một temperature model độc lập cho từng router.

    Args:
        records: Records sinh ra duy nhất từ calibration split.
        routers: Các router bắt buộc phải có model.

    Returns:
        Mapping router sang artifact calibration có version.

    Raises:
        ValueError: Khi một router thiếu record hợp lệ hoặc thiếu gold route.
    """

    models: dict[RouterKind, TemperatureCalibrationModel] = {}
    for router in routers:
        eligible = [
            record
            for record in records
            if record.router == router
            and record.status is CompletionStatus.COMPLETED
            and record.expected_route is not None
            and bool(record.route_probabilities)
        ]
        if not eligible:
            raise ValueError(f"router {router.value!r} không có calibration record hợp lệ")
        models[router] = MulticlassTemperatureCalibrator().fit(
            [record.route_probabilities for record in eligible],
            [record.expected_route for record in eligible if record.expected_route is not None],
            split=DatasetSplit.CALIBRATION,
        )
    return models


def write_calibration_models(
    path: Path,
    models: Mapping[RouterKind, TemperatureCalibrationModel],
) -> None:
    """Ghi atomic các calibration artifact, không chứa dữ liệu provider nhạy cảm.

    Args:
        path: File JSON đích.
        models: Mapping model theo router.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        router.value: model.model_dump(mode="json")
        for router, model in sorted(models.items(), key=lambda item: item[0].value)
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_calibration_models(
    path: Path,
    routers: Sequence[RouterKind],
) -> dict[RouterKind, TemperatureCalibrationModel]:
    """Đọc và validate artifact calibration đã fit trên calibration split.

    Args:
        path: File JSON do formal benchmark tạo.
        routers: Các router bắt buộc phải có trong artifact.

    Returns:
        Mapping model đã được Pydantic kiểm tra version và fitted split.

    Raises:
        FileNotFoundError: Khi formal artifact chưa tồn tại.
        ValueError: Khi JSON hoặc model không hợp lệ hay thiếu router.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy calibration artifact: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Không thể đọc calibration artifact {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("calibration artifact phải là JSON object")
    models: dict[RouterKind, TemperatureCalibrationModel] = {}
    for router in routers:
        payload = raw.get(router.value)
        if payload is None:
            raise ValueError(f"calibration artifact thiếu router {router.value!r}")
        models[router] = TemperatureCalibrationModel.model_validate(payload)
    return models
