"""Cung cấp paired stratified bootstrap cho so sánh các router."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean

from adaptive_rag_router.evaluation.models import BootstrapInterval


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Nội suy tuyến tính một quantile từ dãy đã sắp xếp."""

    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_stratified_bootstrap_ci(
    candidate_values: Sequence[float],
    baseline_values: Sequence[float],
    strata: Sequence[str],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    """Ước lượng CI của mean(candidate - baseline) bằng paired bootstrap.

    Mỗi lần lặp resample có hoàn lại bên trong từng stratum, nhờ đó vừa giữ
    pairing theo query vừa bảo toàn kích thước từng tầng dữ liệu.

    Args:
        candidate_values: Metric của candidate theo query.
        baseline_values: Metric baseline cùng thứ tự query.
        strata: Stratum của từng cặp.
        resamples: Số lần bootstrap.
        confidence_level: Mức confidence hai phía.
        seed: Seed tái lập.

    Returns:
        Estimate gốc và percentile confidence interval.

    Raises:
        ValueError: Khi đầu vào rỗng, khác độ dài hoặc tham số không hợp lệ.
    """

    if not candidate_values:
        raise ValueError("cần ít nhất một paired observation")
    if not (len(candidate_values) == len(baseline_values) == len(strata)):
        raise ValueError("candidate, baseline và strata phải có cùng độ dài")
    if resamples < 1:
        raise ValueError("resamples phải dương")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level phải nằm trong (0, 1)")

    differences = [
        float(candidate) - float(baseline)
        for candidate, baseline in zip(candidate_values, baseline_values, strict=True)
    ]
    indices_by_stratum: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        indices_by_stratum[stratum].append(index)

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled_differences: list[float] = []
        for indices in indices_by_stratum.values():
            sampled_differences.extend(
                differences[rng.choice(indices)] for _ in range(len(indices))
            )
        estimates.append(fmean(sampled_differences))

    estimates.sort()
    alpha = 1.0 - confidence_level
    return BootstrapInterval(
        estimate=fmean(differences),
        lower=_quantile(estimates, alpha / 2.0),
        upper=_quantile(estimates, 1.0 - alpha / 2.0),
        confidence_level=confidence_level,
        resamples=resamples,
        pair_count=len(differences),
    )
