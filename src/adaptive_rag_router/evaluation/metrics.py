"""Tính các metric routing, calibration, chất lượng, chi phí và latency."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from statistics import fmean

from adaptive_rag_router.domain.enums import (
    CompletionStatus,
    ContextQuality,
    ModelTier,
    RetrievalSource,
    RouterKind,
)
from adaptive_rag_router.evaluation.models import (
    AnswerQualityMetrics,
    BenchmarkRecord,
    BinaryRankingMetrics,
    CalibrationMetrics,
    CitationMetrics,
    CriterionComparator,
    CriterionStatus,
    NumericSummary,
    OrdinalMetrics,
    RetrievalMetrics,
    RiskCoverageMetrics,
    RiskCoveragePoint,
    RouterMetrics,
    RoutingMetrics,
    SuccessCriteriaEvaluation,
    SuccessCriteriaInputs,
    SuccessCriteriaThresholds,
    SuccessCriterion,
)

_ENGLISH_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Trả tỷ lệ, dùng 0 khi mẫu số bằng 0."""

    return numerator / denominator if denominator else 0.0


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Nội suy tuyến tính percentile trên dãy đã sắp xếp."""

    if not sorted_values:
        raise ValueError("không thể tính percentile trên dãy rỗng")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile phải nằm trong [0, 1]")
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def summarize_numeric(values: Iterable[float]) -> NumericSummary:
    """Tạo mean, min, max và p50/p95/p99 cho một chuỗi số.

    Args:
        values: Chuỗi giá trị cần tổng hợp.

    Returns:
        NumericSummary có các percentile nội suy; dãy rỗng trả các giá trị None.
    """

    materialized = sorted(float(value) for value in values)
    if not materialized:
        return NumericSummary(count=0)
    return NumericSummary(
        count=len(materialized),
        mean=fmean(materialized),
        minimum=materialized[0],
        maximum=materialized[-1],
        p50=_percentile(materialized, 0.50),
        p95=_percentile(materialized, 0.95),
        p99=_percentile(materialized, 0.99),
    )


def calculate_routing_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> RoutingMetrics:
    """Tính accuracy, macro-F1, balanced accuracy và confusion matrix.

    Args:
        expected: Nhãn chuẩn theo thứ tự sample.
        predicted: Nhãn dự đoán cùng thứ tự.
        labels: Thứ tự label tùy chọn; mặc định lấy hợp của hai dãy.

    Returns:
        Bộ metric routing multiclass.

    Raises:
        ValueError: Khi hai dãy rỗng, khác độ dài hoặc labels thiếu nhãn xuất hiện.
    """

    if len(expected) != len(predicted):
        raise ValueError("expected và predicted phải có cùng độ dài")
    if not expected:
        raise ValueError("cần ít nhất một sample để tính routing metrics")

    observed = set(expected) | set(predicted)
    ordered_labels = tuple(labels) if labels is not None else tuple(sorted(observed))
    if not ordered_labels or not observed.issubset(ordered_labels):
        raise ValueError("labels phải chứa mọi nhãn đã quan sát")

    confusion = {
        actual: {prediction: 0 for prediction in ordered_labels} for actual in ordered_labels
    }
    correct = 0
    for actual, prediction in zip(expected, predicted, strict=True):
        confusion[actual][prediction] += 1
        correct += int(actual == prediction)

    f1_scores: list[float] = []
    recalls: list[float] = []
    for label in ordered_labels:
        true_positive = confusion[label][label]
        false_negative = sum(confusion[label].values()) - true_positive
        false_positive = sum(confusion[actual][label] for actual in ordered_labels) - true_positive
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1_scores.append(
            2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        if true_positive + false_negative:
            recalls.append(recall)

    return RoutingMetrics(
        sample_count=len(expected),
        accuracy=correct / len(expected),
        macro_f1=fmean(f1_scores),
        balanced_accuracy=fmean(recalls) if recalls else 0.0,
        labels=ordered_labels,
        confusion_matrix=confusion,
    )


def calculate_calibration_metrics(
    expected: Sequence[str],
    probabilities: Sequence[Mapping[str, float]],
    *,
    bin_count: int = 10,
    epsilon: float = 1e-15,
) -> CalibrationMetrics:
    """Tính multiclass Brier, NLL và top-label ECE.

    Args:
        expected: Nhãn chuẩn theo thứ tự sample.
        probabilities: Distribution dự đoán tương ứng từng sample.
        bin_count: Số bin confidence bằng nhau cho ECE.
        epsilon: Sàn xác suất khi tính log loss.

    Returns:
        Metric calibration được chuẩn hóa theo số sample.

    Raises:
        ValueError: Khi đầu vào không hợp lệ hoặc distribution không có tổng bằng 1.
    """

    if len(expected) != len(probabilities):
        raise ValueError("expected và probabilities phải có cùng độ dài")
    if not expected:
        raise ValueError("cần ít nhất một sample để tính calibration")
    if bin_count < 2:
        raise ValueError("bin_count phải từ 2 trở lên")
    if epsilon <= 0:
        raise ValueError("epsilon phải dương")

    label_space = set(expected)
    for distribution in probabilities:
        if not distribution:
            raise ValueError("probability distribution không được rỗng")
        if any(value < 0 or value > 1 for value in distribution.values()):
            raise ValueError("probability phải nằm trong [0, 1]")
        if abs(sum(distribution.values()) - 1.0) > 1e-3:
            raise ValueError("tổng probability phải bằng 1")
        label_space.update(distribution)

    brier_values: list[float] = []
    nll_values: list[float] = []
    bins: list[list[tuple[float, float]]] = [[] for _ in range(bin_count)]

    for actual, distribution in zip(expected, probabilities, strict=True):
        brier_values.append(
            sum(
                (distribution.get(label, 0.0) - float(label == actual)) ** 2
                for label in label_space
            )
        )
        true_probability = max(distribution.get(actual, 0.0), epsilon)
        nll_values.append(-math.log(true_probability))

        predicted_label, confidence = max(distribution.items(), key=lambda item: item[1])
        bin_index = min(int(confidence * bin_count), bin_count - 1)
        bins[bin_index].append((confidence, float(predicted_label == actual)))

    sample_count = len(expected)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        average_confidence = fmean(item[0] for item in bucket)
        average_accuracy = fmean(item[1] for item in bucket)
        ece += len(bucket) / sample_count * abs(average_accuracy - average_confidence)

    return CalibrationMetrics(
        sample_count=sample_count,
        brier_score=fmean(brier_values),
        negative_log_likelihood=fmean(nll_values),
        expected_calibration_error=ece,
        bin_count=bin_count,
    )


def calculate_ordinal_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    ordered_labels: Sequence[str],
) -> OrdinalMetrics:
    """Tính ordinal MAE và quadratic weighted kappa.

    Args:
        expected: Nhãn chuẩn theo thứ tự sample.
        predicted: Nhãn dự đoán theo cùng thứ tự.
        ordered_labels: Toàn bộ nhãn từ mức thấp đến cao.

    Returns:
        OrdinalMetrics với MAE tính theo khoảng cách chỉ số nhãn.

    Raises:
        ValueError: Khi dữ liệu rỗng, lệch độ dài, nhãn trùng hoặc nhãn lạ.
    """

    if len(expected) != len(predicted):
        raise ValueError("expected và predicted phải có cùng độ dài")
    if not expected:
        raise ValueError("cần ít nhất một sample để tính ordinal metrics")
    labels = tuple(ordered_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("ordered_labels phải không rỗng và không trùng lặp")
    index_by_label = {label: index for index, label in enumerate(labels)}
    observed_labels = set(expected) | set(predicted)
    if not observed_labels.issubset(index_by_label):
        raise ValueError("ordered_labels phải chứa mọi nhãn đã quan sát")

    expected_indices = [index_by_label[label] for label in expected]
    predicted_indices = [index_by_label[label] for label in predicted]
    mean_absolute_error = fmean(
        abs(actual - prediction)
        for actual, prediction in zip(expected_indices, predicted_indices, strict=True)
    )

    label_count = len(labels)
    if label_count == 1:
        weighted_kappa = 1.0
    else:
        observed = [[0.0 for _ in labels] for _ in labels]
        actual_histogram = [0 for _ in labels]
        predicted_histogram = [0 for _ in labels]
        for actual, prediction in zip(expected_indices, predicted_indices, strict=True):
            observed[actual][prediction] += 1.0
            actual_histogram[actual] += 1
            predicted_histogram[prediction] += 1

        weighted_observed = 0.0
        weighted_expected = 0.0
        denominator_scale = float((label_count - 1) ** 2)
        sample_count = len(expected)
        for actual_index in range(label_count):
            for predicted_index in range(label_count):
                weight = ((actual_index - predicted_index) ** 2) / denominator_scale
                expected_count = (
                    actual_histogram[actual_index]
                    * predicted_histogram[predicted_index]
                    / sample_count
                )
                weighted_observed += weight * observed[actual_index][predicted_index]
                weighted_expected += weight * expected_count
        weighted_kappa = (
            1.0
            if weighted_expected == 0 and weighted_observed == 0
            else 0.0
            if weighted_expected == 0
            else 1.0 - weighted_observed / weighted_expected
        )
        weighted_kappa = min(1.0, weighted_kappa)

    return OrdinalMetrics(
        sample_count=len(expected),
        labels=labels,
        mean_absolute_error=mean_absolute_error,
        quadratic_weighted_kappa=weighted_kappa,
    )


def calculate_one_vs_rest_metrics(
    expected: Sequence[str],
    positive_scores: Sequence[float],
    *,
    positive_label: str,
    threshold: float = 0.5,
) -> BinaryRankingMetrics:
    """Tính AUROC, AUPRC và recall tại threshold cho một lớp quan trọng.

    Args:
        expected: Nhãn multiclass chuẩn.
        positive_scores: Xác suất/score [0, 1] của ``positive_label``.
        positive_label: Lớp được xem là positive trong phép one-vs-rest.
        threshold: Ngưỡng biến score thành dự đoán positive để tính recall.

    Returns:
        BinaryRankingMetrics; metric không xác định được trả về ``None``.

    Raises:
        ValueError: Khi input rỗng, lệch độ dài hoặc score/threshold ngoài [0, 1].
    """

    if len(expected) != len(positive_scores):
        raise ValueError("expected và positive_scores phải có cùng độ dài")
    if not expected:
        raise ValueError("cần ít nhất một sample để tính one-vs-rest metrics")
    if not positive_label.strip():
        raise ValueError("positive_label không được rỗng")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold phải nằm trong [0, 1]")
    scores = [float(score) for score in positive_scores]
    if any(not math.isfinite(score) or score < 0 or score > 1 for score in scores):
        raise ValueError("positive_scores phải là số hữu hạn trong [0, 1]")

    targets = [label == positive_label for label in expected]
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    recall = None
    if positive_count:
        recall = (
            sum(
                target and score >= threshold for target, score in zip(targets, scores, strict=True)
            )
            / positive_count
        )

    auroc = None
    if positive_count and negative_count:
        ranked = sorted(enumerate(scores), key=lambda item: item[1])
        rank_sum_positive = 0.0
        cursor = 0
        while cursor < len(ranked):
            group_end = cursor + 1
            while group_end < len(ranked) and ranked[group_end][1] == ranked[cursor][1]:
                group_end += 1
            average_rank = (cursor + 1 + group_end) / 2.0
            rank_sum_positive += sum(
                average_rank for index, _ in ranked[cursor:group_end] if targets[index]
            )
            cursor = group_end
        auroc = (rank_sum_positive - positive_count * (positive_count + 1) / 2.0) / (
            positive_count * negative_count
        )

    auprc = None
    if positive_count:
        ranked_descending = sorted(
            zip(scores, targets, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
        true_positives = 0
        accepted = 0
        previous_recall = 0.0
        area = 0.0
        cursor = 0
        while cursor < len(ranked_descending):
            group_end = cursor + 1
            while (
                group_end < len(ranked_descending)
                and ranked_descending[group_end][0] == ranked_descending[cursor][0]
            ):
                group_end += 1
            group = ranked_descending[cursor:group_end]
            true_positives += sum(target for _, target in group)
            accepted += len(group)
            current_recall = true_positives / positive_count
            precision = true_positives / accepted
            area += (current_recall - previous_recall) * precision
            previous_recall = current_recall
            cursor = group_end
        auprc = area

    return BinaryRankingMetrics(
        sample_count=len(expected),
        positive_label=positive_label,
        positive_count=positive_count,
        negative_count=negative_count,
        threshold=threshold,
        auroc=auroc,
        auprc=auprc,
        critical_class_recall=recall,
    )


def _normalize_answer(answer: str) -> str:
    """Chuẩn hóa câu trả lời kiểu SQuAD trước khi so khớp token."""

    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in answer.casefold()
    )
    without_articles = _ENGLISH_ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def _token_f1(prediction: str, reference: str) -> float:
    """Tính token-F1 có xét số lần token xuất hiện."""

    predicted_tokens = _normalize_answer(prediction).split()
    reference_tokens = _normalize_answer(reference).split()
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0
    common_count = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if common_count == 0:
        return 0.0
    precision = common_count / len(predicted_tokens)
    recall = common_count / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def calculate_answer_quality_metrics(
    predicted: Sequence[str],
    references: Sequence[str | Sequence[str]],
) -> AnswerQualityMetrics:
    """Tính exact match và token-F1, hỗ trợ nhiều reference mỗi sample.

    Args:
        predicted: Các câu trả lời do pipeline sinh ra.
        references: Một reference hoặc dãy reference hợp lệ cho từng sample.

    Returns:
        AnswerQualityMetrics lấy điểm tốt nhất trong các reference của sample.

    Raises:
        ValueError: Khi input rỗng, lệch độ dài hoặc một sample không có reference.
    """

    if len(predicted) != len(references):
        raise ValueError("predicted và references phải có cùng độ dài")
    if not predicted:
        raise ValueError("cần ít nhất một sample để tính answer metrics")

    exact_matches: list[float] = []
    token_scores: list[float] = []
    for prediction, sample_references in zip(predicted, references, strict=True):
        candidates = (
            (sample_references,) if isinstance(sample_references, str) else tuple(sample_references)
        )
        if not candidates:
            raise ValueError("mỗi sample phải có ít nhất một reference")
        normalized_prediction = _normalize_answer(prediction)
        exact_matches.append(
            max(
                float(normalized_prediction == _normalize_answer(reference))
                for reference in candidates
            )
        )
        token_scores.append(max(_token_f1(prediction, reference) for reference in candidates))

    return AnswerQualityMetrics(
        sample_count=len(predicted),
        exact_match=fmean(exact_matches),
        token_f1=fmean(token_scores),
    )


def calculate_citation_metrics(
    predicted: Sequence[Collection[str]],
    references: Sequence[Collection[str]],
) -> CitationMetrics:
    """Tính citation precision/recall dạng micro trên ID evidence.

    Args:
        predicted: Citation ID được câu trả lời sử dụng theo từng sample.
        references: Citation ID chuẩn/hỗ trợ được theo từng sample.

    Returns:
        CitationMetrics với duplicate citation trong cùng sample chỉ tính một lần.

    Raises:
        ValueError: Khi input rỗng hoặc lệch độ dài.
    """

    if len(predicted) != len(references):
        raise ValueError("predicted và references phải có cùng độ dài")
    if not predicted:
        raise ValueError("cần ít nhất một sample để tính citation metrics")

    predicted_count = 0
    reference_count = 0
    supported_count = 0
    for predicted_ids, reference_ids in zip(predicted, references, strict=True):
        predicted_set = {str(item).strip() for item in predicted_ids if str(item).strip()}
        reference_set = {str(item).strip() for item in reference_ids if str(item).strip()}
        predicted_count += len(predicted_set)
        reference_count += len(reference_set)
        supported_count += len(predicted_set & reference_set)

    precision = (
        supported_count / predicted_count if predicted_count else float(reference_count == 0)
    )
    recall = supported_count / reference_count if reference_count else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CitationMetrics(
        sample_count=len(predicted),
        predicted_citation_count=predicted_count,
        reference_citation_count=reference_count,
        supported_citation_count=supported_count,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def calculate_risk_coverage_curve(
    confidences: Sequence[float],
    correctness: Sequence[float],
) -> RiskCoverageMetrics:
    """Dựng đường risk-coverage khi abstain dưới confidence threshold.

    Args:
        confidences: Confidence [0, 1] của từng dự đoán.
        correctness: Điểm đúng [0, 1]; risk của sample bằng ``1 - correctness``.

    Returns:
        Các điểm tại từng confidence duy nhất và diện tích trapezoid từ coverage 0.

    Raises:
        ValueError: Khi input rỗng, lệch độ dài hoặc có giá trị ngoài [0, 1].
    """

    if len(confidences) != len(correctness):
        raise ValueError("confidences và correctness phải có cùng độ dài")
    if not confidences:
        raise ValueError("cần ít nhất một sample để tính risk-coverage")
    confidence_values = [float(value) for value in confidences]
    correctness_values = [float(value) for value in correctness]
    if any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in confidence_values + correctness_values
    ):
        raise ValueError("confidence và correctness phải là số hữu hạn trong [0, 1]")

    ranked = sorted(
        zip(confidence_values, correctness_values, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    points: list[RiskCoveragePoint] = []
    accepted_count = 0
    cumulative_risk = 0.0
    cursor = 0
    while cursor < len(ranked):
        group_end = cursor + 1
        while group_end < len(ranked) and ranked[group_end][0] == ranked[cursor][0]:
            group_end += 1
        group = ranked[cursor:group_end]
        accepted_count += len(group)
        cumulative_risk += sum(1.0 - score for _, score in group)
        points.append(
            RiskCoveragePoint(
                threshold=ranked[cursor][0],
                coverage=accepted_count / len(ranked),
                risk=cumulative_risk / accepted_count,
                accepted_count=accepted_count,
            )
        )
        cursor = group_end

    area = 0.0
    previous_coverage = 0.0
    previous_risk = 0.0
    for point in points:
        area += (point.coverage - previous_coverage) * (point.risk + previous_risk) / 2.0
        previous_coverage = point.coverage
        previous_risk = point.risk
    return RiskCoverageMetrics(
        sample_count=len(ranked),
        points=tuple(points),
        area_under_curve=area,
    )


def calculate_retrieval_metrics(
    expected_supporting_facts: Sequence[Collection[str]],
    retrieved_supporting_facts: Sequence[Collection[str]],
    expected_context_quality: Sequence[str],
    predicted_context_quality: Sequence[str],
) -> RetrievalMetrics:
    """Tính supporting-fact recall và metric phân loại context quality.

    Args:
        expected_supporting_facts: ID supporting fact chuẩn theo sample.
        retrieved_supporting_facts: ID evidence đã retrieve theo sample.
        expected_context_quality: Nhãn context quality chuẩn.
        predicted_context_quality: Nhãn context quality dự đoán.

    Returns:
        RetrievalMetrics với supporting-fact recall dạng micro và macro-F1 ba lớp.

    Raises:
        ValueError: Khi bốn dãy không cùng độ dài hoặc rỗng.
    """

    lengths = {
        len(expected_supporting_facts),
        len(retrieved_supporting_facts),
        len(expected_context_quality),
        len(predicted_context_quality),
    }
    if len(lengths) != 1:
        raise ValueError("các input retrieval metrics phải có cùng độ dài")
    if not expected_context_quality:
        raise ValueError("cần ít nhất một sample để tính retrieval metrics")

    fact_hits = 0
    fact_count = 0
    for expected_ids, retrieved_ids in zip(
        expected_supporting_facts,
        retrieved_supporting_facts,
        strict=True,
    ):
        expected_set = {str(item).strip() for item in expected_ids if str(item).strip()}
        retrieved_set = {str(item).strip() for item in retrieved_ids if str(item).strip()}
        fact_hits += len(expected_set & retrieved_set)
        fact_count += len(expected_set)
    supporting_fact_recall = fact_hits / fact_count if fact_count else 1.0

    quality_labels = tuple(quality.value for quality in ContextQuality)
    quality_metrics = calculate_routing_metrics(
        expected_context_quality,
        predicted_context_quality,
        labels=quality_labels,
    )
    insufficient = ContextQuality.INSUFFICIENT.value
    insufficient_total = sum(label == insufficient for label in expected_context_quality)
    insufficient_hits = sum(
        actual == insufficient and prediction == insufficient
        for actual, prediction in zip(
            expected_context_quality,
            predicted_context_quality,
            strict=True,
        )
    )
    insufficient_recall = insufficient_hits / insufficient_total if insufficient_total else None
    return RetrievalMetrics(
        sample_count=len(expected_context_quality),
        supporting_fact_hits=fact_hits,
        supporting_fact_count=fact_count,
        supporting_fact_recall=supporting_fact_recall,
        context_quality_macro_f1=quality_metrics.macro_f1,
        insufficient_recall=insufficient_recall,
    )


def _criterion(
    name: str,
    observed: float | None,
    threshold: float,
    comparator: CriterionComparator,
    description: str,
) -> SuccessCriterion:
    """Tạo kết quả tiêu chí số, giữ trạng thái chưa đánh giá khi thiếu input."""

    if observed is None:
        status = CriterionStatus.NOT_EVALUATED
    elif comparator == CriterionComparator.GREATER_OR_EQUAL:
        status = CriterionStatus.PASSED if observed >= threshold else CriterionStatus.FAILED
    elif comparator == CriterionComparator.LESS_OR_EQUAL:
        status = CriterionStatus.PASSED if observed <= threshold else CriterionStatus.FAILED
    elif comparator == CriterionComparator.LESS_THAN:
        status = CriterionStatus.PASSED if observed < threshold else CriterionStatus.FAILED
    else:
        status = CriterionStatus.PASSED if observed >= threshold else CriterionStatus.FAILED
    return SuccessCriterion(
        name=name,
        status=status,
        comparator=comparator,
        threshold=threshold,
        observed=observed,
        description=description,
    )


def _relative_reduction(candidate: float | None, baseline: float | None) -> float | None:
    """Tính mức giảm tương đối; trả None khi baseline không dương."""

    if candidate is None or baseline is None or baseline <= 0:
        return None
    return (baseline - candidate) / baseline


def evaluate_success_criteria(
    inputs: SuccessCriteriaInputs,
    thresholds: SuccessCriteriaThresholds | None = None,
) -> SuccessCriteriaEvaluation:
    """Đánh giá các ngưỡng thành công đã khóa trong kế hoạch MVP.

    Args:
        inputs: Aggregate/CI quan sát được từ benchmark held-out.
        thresholds: Ngưỡng preregistered; bỏ qua để dùng giá trị mặc định.

    Returns:
        Kết quả từng tiêu chí và trạng thái tổng. Nếu thiếu bất kỳ quan sát nào
        và chưa có tiêu chí thất bại, trạng thái tổng là ``not_evaluated``.
    """

    active = thresholds or SuccessCriteriaThresholds()
    cost_reduction = _relative_reduction(inputs.jev_mean_cost_usd, inputs.llm_mean_cost_usd)
    p50_reduction = _relative_reduction(
        inputs.jev_p50_latency_ms,
        inputs.llm_p50_latency_ms,
    )
    p95_increase = None
    if (
        inputs.jev_p95_latency_ms is not None
        and inputs.llm_p95_latency_ms is not None
        and inputs.llm_p95_latency_ms > 0
    ):
        p95_increase = (
            inputs.jev_p95_latency_ms - inputs.llm_p95_latency_ms
        ) / inputs.llm_p95_latency_ms

    pareto = None
    if (
        inputs.jev_quality is not None
        and inputs.rule_quality is not None
        and inputs.jev_mean_cost_usd is not None
        and inputs.rule_mean_cost_usd is not None
    ):
        weakly_better = (
            inputs.jev_quality >= inputs.rule_quality
            and inputs.jev_mean_cost_usd <= inputs.rule_mean_cost_usd
        )
        strictly_better = (
            inputs.jev_quality > inputs.rule_quality
            or inputs.jev_mean_cost_usd < inputs.rule_mean_cost_usd
        )
        pareto = float(weakly_better and strictly_better)

    criteria = (
        _criterion(
            "quality_non_inferiority",
            inputs.quality_difference_ci_lower,
            -active.quality_non_inferiority_margin,
            CriterionComparator.GREATER_OR_EQUAL,
            "Cận dưới CI một phía của quality Jev - LLM không thấp hơn "
            f"-{active.quality_non_inferiority_margin:.4f}.",
        ),
        _criterion(
            "mean_cost_reduction",
            cost_reduction,
            active.minimum_cost_reduction,
            CriterionComparator.GREATER_OR_EQUAL,
            "Jev giảm mean variable cost ít nhất "
            f"{active.minimum_cost_reduction:.1%} so với LLM router.",
        ),
        _criterion(
            "p50_latency_reduction",
            p50_reduction,
            active.minimum_p50_latency_reduction,
            CriterionComparator.GREATER_OR_EQUAL,
            "Jev giảm p50 latency ít nhất "
            f"{active.minimum_p50_latency_reduction:.1%} so với LLM router.",
        ),
        _criterion(
            "p95_latency_increase",
            p95_increase,
            active.maximum_p95_latency_regression,
            CriterionComparator.LESS_OR_EQUAL,
            "p95 latency của Jev không tăng quá "
            f"{active.maximum_p95_latency_regression:.1%} so với LLM router.",
        ),
        _criterion(
            "critical_class_recall",
            inputs.critical_class_recall,
            active.critical_recall,
            CriterionComparator.GREATER_OR_EQUAL,
            f"Recall của lớp routing quan trọng đạt ít nhất {active.critical_recall:.4f}.",
        ),
        _criterion(
            "expected_calibration_error",
            inputs.expected_calibration_error,
            active.maximum_ece,
            CriterionComparator.LESS_OR_EQUAL,
            f"ECE sau calibration không vượt quá {active.maximum_ece:.4f}.",
        ),
        _criterion(
            "unhandled_router_schema_error_rate",
            inputs.unhandled_router_schema_error_rate,
            active.maximum_unhandled_error_rate,
            CriterionComparator.LESS_THAN,
            "Lỗi router/schema không được xử lý thấp hơn "
            f"{active.maximum_unhandled_error_rate:.2%}.",
        ),
        _criterion(
            "quality_cost_pareto_vs_rule",
            pareto,
            1.0,
            CriterionComparator.PARETO,
            "Jev không kém rule-based ở quality và cost, đồng thời tốt hơn ít nhất một chiều.",
        ),
    )
    statuses = {criterion.status for criterion in criteria}
    overall_status = (
        CriterionStatus.FAILED
        if CriterionStatus.FAILED in statuses
        else CriterionStatus.NOT_EVALUATED
        if CriterionStatus.NOT_EVALUATED in statuses
        else CriterionStatus.PASSED
    )
    return SuccessCriteriaEvaluation(
        overall_status=overall_status,
        criteria=criteria,
    )


def aggregate_router_metrics(
    router: RouterKind,
    records: Sequence[BenchmarkRecord],
    *,
    ece_bins: int = 10,
) -> RouterMetrics:
    """Tổng hợp toàn bộ metric cho một router.

    Args:
        router: Router cần tổng hợp.
        records: Record của benchmark run; record router khác sẽ bị bỏ qua.
        ece_bins: Số bin cho ECE.

    Returns:
        RouterMetrics kể cả khi router chưa có record.
    """

    selected = [record for record in records if record.router == router]
    labeled = [
        record
        for record in selected
        if record.expected_route is not None and record.predicted_route is not None
    ]
    routing = None
    if labeled:
        routing = calculate_routing_metrics(
            [record.expected_route for record in labeled if record.expected_route is not None],
            [record.predicted_route for record in labeled if record.predicted_route is not None],
        )

    calibrated = [record for record in labeled if record.route_probabilities]
    calibration = None
    if calibrated:
        calibration = calculate_calibration_metrics(
            [record.expected_route for record in calibrated if record.expected_route is not None],
            [record.route_probabilities for record in calibrated],
            bin_count=ece_bins,
        )

    ordinal_model_tier = None
    if labeled:
        expected_tiers = [(record.expected_route or "").rsplit(":", 1)[-1] for record in labeled]
        predicted_tiers = [(record.predicted_route or "").rsplit(":", 1)[-1] for record in labeled]
        if set((*expected_tiers, *predicted_tiers)).issubset({tier.value for tier in ModelTier}):
            ordinal_model_tier = calculate_ordinal_metrics(
                expected_tiers,
                predicted_tiers,
                ordered_labels=tuple(tier.value for tier in ModelTier),
            )

    critical_route = None
    if calibrated:
        critical_route = calculate_one_vs_rest_metrics(
            [
                "critical" if _is_critical_route(record.expected_route or "") else "standard"
                for record in calibrated
            ],
            [
                sum(
                    probability
                    for label, probability in record.route_probabilities.items()
                    if _is_critical_route(label)
                )
                for record in calibrated
            ],
            positive_label="critical",
        )

    risk_coverage = None
    if calibrated:
        risk_coverage = calculate_risk_coverage_curve(
            [max(record.route_probabilities.values()) for record in calibrated],
            [float(record.expected_route == record.predicted_route) for record in calibrated],
        )

    answer_records = [record for record in selected if record.reference_answer is not None]
    answer_quality = None
    if answer_records:
        answer_quality = calculate_answer_quality_metrics(
            [record.answer for record in answer_records],
            [
                record.reference_answer
                for record in answer_records
                if record.reference_answer is not None
            ],
        )

    citation_records = [
        record
        for record in selected
        if isinstance(record.metadata.get("supporting_document_ids"), list)
        and bool(record.metadata.get("supporting_document_ids"))
    ]
    citations = None
    if citation_records:
        citations = calculate_citation_metrics(
            [
                _as_string_collection(record.metadata.get("citations"))
                for record in citation_records
            ],
            [
                _as_string_collection(record.metadata.get("supporting_document_ids"))
                for record in citation_records
            ],
        )

    retrieval_records = [
        record
        for record in selected
        if isinstance(record.metadata.get("expected_context_quality"), str)
        and isinstance(record.metadata.get("predicted_context_quality"), str)
    ]
    retrieval = None
    if retrieval_records:
        retrieval = calculate_retrieval_metrics(
            [
                _as_string_collection(record.metadata.get("supporting_document_ids"))
                for record in retrieval_records
            ],
            [
                _as_string_collection(record.metadata.get("retrieved_document_ids"))
                for record in retrieval_records
            ],
            [str(record.metadata["expected_context_quality"]) for record in retrieval_records],
            [str(record.metadata["predicted_context_quality"]) for record in retrieval_records],
        )

    record_count = len(selected)
    error_count = sum(
        record.error is not None or record.status == CompletionStatus.FAILED for record in selected
    )
    abstention_count = sum(record.status == CompletionStatus.ABSTAINED for record in selected)
    return RouterMetrics(
        router=router,
        record_count=record_count,
        routing=routing,
        calibration=calibration,
        ordinal_model_tier=ordinal_model_tier,
        critical_route=critical_route,
        risk_coverage=risk_coverage,
        answer_quality=answer_quality,
        citations=citations,
        retrieval=retrieval,
        quality=summarize_numeric(
            record.quality_score for record in selected if record.quality_score is not None
        ),
        total_tokens=summarize_numeric(
            float(record.metadata.get("total_tokens", 0)) for record in selected
        ),
        cost_usd=summarize_numeric(record.cost_usd for record in selected),
        latency_ms=summarize_numeric(record.latency_ms for record in selected),
        error_rate=_safe_ratio(error_count, record_count),
        abstention_rate=_safe_ratio(abstention_count, record_count),
        retrieval_rate=_safe_ratio(
            sum(bool(record.metadata.get("used_retrieval")) for record in selected),
            record_count,
        ),
        fallback_rate=_safe_ratio(
            sum(bool(record.metadata.get("used_strong_fallback")) for record in selected),
            record_count,
        ),
        total_external_credits=sum(record.external_credits for record in selected),
    )


def _is_critical_route(label: str) -> bool:
    """Xem route dùng web hoặc strong model là lớp quan trọng cần recall cao."""

    try:
        source, tier = label.split(":", 1)
    except ValueError:
        return False
    return source == RetrievalSource.WEB.value or tier == ModelTier.STRONG.value


def _as_string_collection(value: object) -> Collection[str]:
    """Chuẩn hóa metadata list-like thành collection chuỗi an toàn cho metric."""

    if not isinstance(value, list | tuple | set):
        return ()
    return tuple(str(item) for item in value)
