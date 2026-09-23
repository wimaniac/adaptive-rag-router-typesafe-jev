"""Ghi benchmark records ra JSONL/Parquet và báo cáo tổng hợp ra Markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptive_rag_router.evaluation.models import ArtifactPaths, BenchmarkReport
from adaptive_rag_router.evaluation.reporting import build_markdown_report
from adaptive_rag_router.evaluation.visualization import (
    build_pareto_svg,
    build_reliability_svg,
    build_risk_coverage_svg,
)


class BenchmarkArtifactWriter:
    """Writer artifact dùng file tạm và atomic replace trong cùng thư mục.

    Args:
        output_dir: Thư mục đích; được tạo khi gọi phương thức ghi.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def _prepare_path(self, run_id: str, suffix: str) -> Path:
        """Tạo thư mục run và trả đường dẫn artifact ổn định."""

        run_directory = self._output_dir / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory / suffix

    @staticmethod
    def _replace_text(path: Path, content: str) -> None:
        """Ghi text UTF-8 qua file tạm rồi thay thế atomically."""

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8", newline="\n")
        temporary_path.replace(path)

    def write_records_jsonl(self, report: BenchmarkReport) -> Path:
        """Ghi mỗi BenchmarkRecord thành một dòng JSON.

        Args:
            report: Báo cáo chứa record cần ghi.

        Returns:
            Đường dẫn file JSONL hoàn tất.
        """

        path = self._prepare_path(report.run_id, "records.jsonl")
        lines = [
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for record in report.records
        ]
        payload = "\n".join(lines) + ("\n" if lines else "")
        self._replace_text(path, payload)
        return path

    def write_records_parquet(self, report: BenchmarkReport) -> Path:
        """Ghi record phẳng ra Parquet, encode trường lồng nhau thành JSON.

        Args:
            report: Báo cáo chứa record cần ghi.

        Returns:
            Đường dẫn file Parquet hoàn tất.
        """

        import pandas as pd  # type: ignore[import-untyped]

        path = self._prepare_path(report.run_id, "records.parquet")
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        rows: list[dict[str, Any]] = []
        for record in report.records:
            rows.append(
                {
                    "run_id": record.run_id,
                    "query_id": record.query_id,
                    "dataset": record.dataset,
                    "stratum": record.stratum,
                    "group_id": record.group_id,
                    "router": record.router.value,
                    "expected_route": record.expected_route,
                    "predicted_route": record.predicted_route,
                    "route_probabilities_json": json.dumps(
                        record.route_probabilities,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "answer": record.answer,
                    "quality_score": record.quality_score,
                    "cost_usd": record.cost_usd,
                    "latency_ms": record.latency_ms,
                    "status": record.status.value,
                    "external_credits": record.external_credits,
                    "error": record.error,
                    "metadata_json": json.dumps(
                        record.metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                }
            )
        columns = [
            "run_id",
            "query_id",
            "dataset",
            "stratum",
            "group_id",
            "router",
            "expected_route",
            "predicted_route",
            "route_probabilities_json",
            "answer",
            "quality_score",
            "cost_usd",
            "latency_ms",
            "status",
            "external_credits",
            "error",
            "metadata_json",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        frame.to_parquet(temporary_path, index=False)
        temporary_path.replace(path)
        return path

    def write_report_markdown(self, report: BenchmarkReport) -> Path:
        """Ghi báo cáo Markdown tiếng Việt.

        Args:
            report: Báo cáo benchmark cần trình bày.

        Returns:
            Đường dẫn file Markdown hoàn tất.
        """

        path = self._prepare_path(report.run_id, "report.md")
        self._replace_text(path, build_markdown_report(report))
        return path

    def write_visualizations(self, report: BenchmarkReport) -> tuple[Path, Path, Path]:
        """Ghi ba SVG không cần dependency đồ họa bên ngoài.

        Args:
            report: Báo cáo chứa records và aggregate metrics.

        Returns:
            Đường dẫn Pareto, reliability và risk-coverage SVG.
        """

        pareto = self._prepare_path(report.run_id, "pareto.svg")
        reliability = self._prepare_path(report.run_id, "reliability.svg")
        risk_coverage = self._prepare_path(report.run_id, "risk-coverage.svg")
        self._replace_text(pareto, build_pareto_svg(report))
        self._replace_text(reliability, build_reliability_svg(report))
        self._replace_text(risk_coverage, build_risk_coverage_svg(report))
        return pareto, reliability, risk_coverage

    def write_all(self, report: BenchmarkReport) -> ArtifactPaths:
        """Ghi đồng thời JSONL, Parquet và Markdown cho một run.

        Args:
            report: Báo cáo benchmark cần lưu.

        Returns:
            Typed paths của ba artifact.
        """

        pareto, reliability, risk_coverage = self.write_visualizations(report)
        return ArtifactPaths(
            records_jsonl=self.write_records_jsonl(report),
            records_parquet=self.write_records_parquet(report),
            report_markdown=self.write_report_markdown(report),
            pareto_svg=pareto,
            reliability_svg=reliability,
            risk_coverage_svg=risk_coverage,
        )
