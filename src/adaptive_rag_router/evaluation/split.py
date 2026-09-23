"""Tạo split dev/calibration/test có stratification và chống group leakage."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import partial

from adaptive_rag_router.evaluation.models import BenchmarkSample, BenchmarkSplits


def _allocation_score(
    candidate: str,
    *,
    target_sizes: Mapping[str, int],
    assignments: Mapping[str, Sequence[BenchmarkSample]],
    assigned_strata: Mapping[str, Counter[str]],
    expected_strata: Mapping[str, Mapping[str, float]],
    global_strata: Mapping[str, int],
    group_strata: Mapping[str, int],
    group_size: int,
) -> tuple[float, int, str]:
    """Chấm độ lệch stratification toàn cục sau một phân bổ giả định."""

    squared_error = 0.0
    for split_name in target_sizes:
        for stratum in global_strata:
            observed = assigned_strata[split_name][stratum]
            if split_name == candidate:
                observed += group_strata[stratum]
            target = expected_strata[split_name][stratum]
            squared_error += (observed - target) ** 2 / max(target, 1.0)
    remaining = target_sizes[candidate] - len(assignments[candidate]) - group_size
    return squared_error, remaining, candidate


def create_grouped_benchmark_splits(
    samples: Sequence[BenchmarkSample],
    *,
    dev_size: int = 200,
    calibration_size: int = 200,
    test_size: int = 600,
    seed: int = 42,
) -> BenchmarkSplits:
    """Chia dữ liệu theo group với mục tiêu stratification gần tỷ lệ toàn tập.

    Hàm mặc định khóa đúng split 200/200/600 cho benchmark 1.000 mẫu. Toàn bộ
    sample cùng ``group_id`` luôn đi vào cùng một split. Nếu kích thước group
    khiến không thể lấp chính xác capacity bằng thuật toán xác định, hàm báo lỗi
    thay vì âm thầm phá group boundary.

    Args:
        samples: Tập sample đầu vào.
        dev_size: Số sample cho development split.
        calibration_size: Số sample cho calibration split.
        test_size: Số sample cho held-out test split.
        seed: Seed dùng để phá hòa giữa các group tương đương.

    Returns:
        Ba split không giao nhau theo query ID và group ID.

    Raises:
        ValueError: Khi tổng size không khớp, query ID trùng hoặc không thể chia.
    """

    target_sizes = {
        "dev": dev_size,
        "calibration": calibration_size,
        "test": test_size,
    }
    if any(size < 0 for size in target_sizes.values()):
        raise ValueError("kích thước split không được âm")
    if sum(target_sizes.values()) != len(samples):
        raise ValueError("tổng kích thước dev/calibration/test phải bằng số sample đầu vào")
    query_ids = [sample.query_id for sample in samples]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_id phải duy nhất trước khi split")

    groups: dict[str, list[BenchmarkSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    global_strata = Counter(sample.stratum for sample in samples)
    total_count = len(samples)
    expected_strata = {
        split_name: {
            stratum: count * split_size / total_count for stratum, count in global_strata.items()
        }
        for split_name, split_size in target_sizes.items()
    }
    assignments: dict[str, list[BenchmarkSample]] = {split_name: [] for split_name in target_sizes}
    assigned_strata: dict[str, Counter[str]] = {
        split_name: Counter() for split_name in target_sizes
    }

    for group_id, group_samples in group_items:
        group_size = len(group_samples)
        group_strata = Counter(sample.stratum for sample in group_samples)
        candidates = [
            split_name
            for split_name, target_size in target_sizes.items()
            if len(assignments[split_name]) + group_size <= target_size
        ]
        if not candidates:
            raise ValueError(
                f"không thể phân bổ group {group_id!r} mà vẫn giữ đúng kích thước split"
            )

        score = partial(
            _allocation_score,
            target_sizes=target_sizes,
            assignments=assignments,
            assigned_strata=assigned_strata,
            expected_strata=expected_strata,
            global_strata=global_strata,
            group_strata=group_strata,
            group_size=group_size,
        )
        selected_split = min(candidates, key=score)
        assignments[selected_split].extend(group_samples)
        assigned_strata[selected_split].update(group_strata)

    actual_sizes = {name: len(items) for name, items in assignments.items()}
    if actual_sizes != target_sizes:
        raise ValueError(
            "group boundaries không cho phép đạt chính xác kích thước split: "
            f"nhận {actual_sizes}, cần {target_sizes}"
        )

    return BenchmarkSplits(
        dev=tuple(sorted(assignments["dev"], key=lambda sample: sample.query_id)),
        calibration=tuple(sorted(assignments["calibration"], key=lambda sample: sample.query_id)),
        test=tuple(sorted(assignments["test"], key=lambda sample: sample.query_id)),
    )
