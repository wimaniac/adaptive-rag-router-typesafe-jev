"""Kiểm thử chuẩn hóa RAGRouter-Bench và CRAG hoàn toàn offline."""

from __future__ import annotations

import bz2
import json
from pathlib import Path

from adaptive_rag_router.evaluation import load_local_records
from adaptive_rag_router.interfaces.data_preparation import prepare_benchmark_sources


def test_prepare_benchmark_sources_writes_default_layout(tmp_path: Path) -> None:
    rag_dir = tmp_path / "official-ragrouter" / "graphragBench_medical"
    rag_dir.mkdir(parents=True)
    (rag_dir / "Question.json").write_text(
        json.dumps(
            [
                {
                    "id": "medical-1",
                    "question": "What is BCC?",
                    "answer": "A skin cancer.",
                    "supporting_facts": [{"doc_id": "doc-1", "text": "BCC is a skin cancer."}],
                    "type": "single_hop",
                }
            ]
        ),
        encoding="utf-8",
    )
    (rag_dir / "Corpus.json").write_text(
        json.dumps([{"id": "doc-1", "title": "BCC", "context": "BCC is a skin cancer."}]),
        encoding="utf-8",
    )
    second_domain = rag_dir.parent / "graphragBench_musique"
    second_domain.mkdir()
    (second_domain / "Corpus.json").write_text(
        json.dumps([{"id": "doc-1", "title": "Team", "context": "The blue team won."}]),
        encoding="utf-8",
    )
    crag_path = tmp_path / "crag_task_1_and_2_dev_v4.jsonl.bz2"
    crag_record = {
        "interaction_id": "crag-1",
        "query": "Who won?",
        "answer": "The blue team.",
        "domain": "sports",
        "question_type": "simple",
        "static_or_dynamic": "fast-changing",
        "popularity": "head",
        "search_results": [
            {
                "page_name": "Result",
                "page_url": "https://example.test/result",
                "page_result": "<p>The blue team won.</p>",
            }
        ],
    }
    with bz2.open(crag_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(crag_record) + "\n")

    counts = prepare_benchmark_sources(rag_dir.parent, crag_path, tmp_path / "data")

    assert counts.ragrouter_questions == 1
    assert counts.crag_questions == 1
    assert counts.mock_web_queries == 1
    assert counts.corpus_documents == 2
    rag_questions = load_local_records(tmp_path / "data" / "raw" / "ragrouter" / "questions.jsonl")
    assert rag_questions[0]["domain"] == "medical"
    assert rag_questions[0]["query_type"] == "factual"
    assert rag_questions[0]["supporting_document_ids"] == ["ragrouter:medical:doc-1"]
    crag_questions = load_local_records(tmp_path / "data" / "raw" / "crag" / "questions.jsonl")
    assert crag_questions[0]["temporal_dynamism"] == "fast-changing"
    mock_web = load_local_records(tmp_path / "data" / "raw" / "crag" / "mock_web.jsonl")
    assert mock_web[0]["results"][0]["text"] == "The blue team won."
    corpus = load_local_records(tmp_path / "data" / "processed" / "corpus.jsonl")
    assert {record["document_id"] for record in corpus} == {
        "ragrouter:medical:doc-1",
        "ragrouter:wikipedia:doc-1",
    }
