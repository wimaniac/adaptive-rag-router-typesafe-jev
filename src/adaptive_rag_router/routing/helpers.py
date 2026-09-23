"""Chuẩn hóa probability và dựng typed evidence dùng chung giữa các router."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum

from adaptive_rag_router.domain import DecisionEvidence, ProbabilityProvenance, RouterKind
from adaptive_rag_router.domain.errors import DecisionSchemaError


def normalize_probabilities(
    values: Mapping[str, float],
    labels: Sequence[str],
    *,
    selected: str | None = None,
) -> dict[str, float]:
    """Chuẩn hóa score thành distribution đầy đủ theo đúng tập nhãn.

    Args:
        values: Score hoặc probability do decision engine trả về.
        labels: Tập nhãn hợp lệ theo thứ tự ổn định.
        selected: Nhãn đã chọn, dùng làm one-hot fallback khi tổng score bằng 0.

    Returns:
        Distribution có đủ nhãn, tổng bằng một.

    Raises:
        DecisionSchemaError: Khi nhãn, score hoặc tập labels không hợp lệ.
    """

    if not labels or len(set(labels)) != len(labels):
        raise DecisionSchemaError("labels phải không rỗng và không trùng lặp")

    canonical = {label.lower(): label for label in labels}
    normalized_input: dict[str, float] = {}
    for raw_label, raw_value in values.items():
        key = str(raw_label).strip().lower()
        if key not in canonical:
            raise DecisionSchemaError(f"nhãn probability không hợp lệ: {raw_label!r}")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise DecisionSchemaError(f"probability không phải số: {raw_value!r}") from error
        if not math.isfinite(value) or value < 0:
            raise DecisionSchemaError("probability phải hữu hạn và không âm")
        normalized_input[canonical[key]] = value

    selected_label: str | None = None
    if selected is not None:
        selected_key = selected.strip().lower()
        if selected_key not in canonical:
            raise DecisionSchemaError(f"quyết định không hợp lệ: {selected!r}")
        selected_label = canonical[selected_key]

    complete = {label: normalized_input.get(label, 0.0) for label in labels}
    total = math.fsum(complete.values())
    if total <= 0:
        if selected_label is None:
            uniform = 1.0 / len(labels)
            return {label: uniform for label in labels}
        return {label: float(label == selected_label) for label in labels}

    distribution = {label: score / total for label, score in complete.items()}
    # Hiệu chỉnh sai số dấu phẩy động để Pydantic luôn thấy tổng chính xác.
    largest = max(distribution, key=distribution.__getitem__)
    distribution[largest] += 1.0 - math.fsum(distribution.values())
    return distribution


def build_evidence[EnumT: StrEnum](
    enum_type: type[EnumT],
    selected: str | EnumT,
    probabilities: Mapping[str, float],
    *,
    provenance: ProbabilityProvenance,
    engine: RouterKind,
    model_id: str,
    prompt_version: str,
    confidence: float | None = None,
    reasons: Sequence[str] = (),
) -> DecisionEvidence[EnumT]:
    """Dựng `DecisionEvidence` đã kiểm tra enum và chuẩn hóa distribution.

    Args:
        enum_type: Enum đích của quyết định.
        selected: Giá trị đã được engine chọn.
        probabilities: Distribution hoặc score theo nhãn enum.
        provenance: Nguồn gốc probability.
        engine: Loại router tạo quyết định.
        model_id: Phiên bản model hoặc policy.
        prompt_version: Phiên bản prompt/rule.
        confidence: Confidence riêng do provider trả về, nếu có.
        reasons: Reason codes ngắn, không chứa chain-of-thought.

    Returns:
        Evidence bất biến, sẵn sàng gắn vào domain decision.
    """

    selected_value = selected.value if isinstance(selected, StrEnum) else str(selected)
    try:
        selected_enum = enum_type(selected_value.strip().lower())
    except ValueError as error:
        raise DecisionSchemaError(f"{selected_value!r} không thuộc {enum_type.__name__}") from error

    labels = [member.value for member in enum_type]
    distribution = normalize_probabilities(
        probabilities,
        labels,
        selected=selected_enum.value,
    )
    if confidence is not None and (
        not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1
    ):
        raise DecisionSchemaError("confidence phải nằm trong [0, 1]")

    return DecisionEvidence[EnumT](
        selected=selected_enum,
        raw_probabilities=distribution,
        confidence=None if confidence is None else float(confidence),
        provenance=provenance,
        engine=engine,
        model_id=model_id,
        prompt_version=prompt_version,
        reasons=tuple(str(reason) for reason in reasons),
    )


def peaked_distribution[EnumT: StrEnum](
    selected: EnumT,
    enum_type: type[EnumT],
    *,
    peak: float = 0.8,
) -> dict[str, float]:
    """Tạo heuristic distribution có một đỉnh và phần dư chia đều.

    Args:
        selected: Nhãn được rule chọn.
        enum_type: Enum chứa toàn bộ nhãn.
        peak: Probability dành cho nhãn được chọn.

    Returns:
        Distribution đã chuẩn hóa.
    """

    labels = [member.value for member in enum_type]
    if selected.value not in labels:
        raise DecisionSchemaError("selected không thuộc enum_type")
    if not 0 <= peak <= 1:
        raise DecisionSchemaError("peak phải nằm trong [0, 1]")
    if len(labels) == 1:
        return {labels[0]: 1.0}
    remainder = (1.0 - peak) / (len(labels) - 1)
    return {label: peak if label == selected.value else remainder for label in labels}
