# Adaptive RAG Router với TypeSafe Jev

Một hệ thống **Retrieval-Augmented Generation (RAG) thích ứng** có khả năng tự động quyết định:

* có cần truy xuất thông tin hay không,
* nên dùng nguồn truy xuất nào,
* nên chọn model tier nào,
* context hiện tại có đủ tốt hay không,
* và khi nào cần repair, fallback hoặc abstain.

Dự án so sánh ba chiến lược routing trên cùng một hạ tầng RAG:

* **TypeSafe Jev Router**
* **Rule-based Router**
* **LLM Router**

Mục tiêu chính là đánh giá sự đánh đổi giữa **chất lượng câu trả lời, chi phí, độ trễ và độ tin cậy**.

---

## Điểm nổi bật

* Adaptive routing giữa **không retrieval, vector retrieval và web search**
* Tự động chọn **economy / strong LLM**
* Đánh giá chất lượng context trước khi sinh câu trả lời
* Tự động repair retrieval khi context chưa đủ tốt
* Fallback sang model mạnh hơn khi cần
* Controlled abstention khi thiếu bằng chứng hoặc provider lỗi
* Typed decision trace để audit toàn bộ quá trình xử lý
* Retrieval với **BGE-M3 + Qdrant**
* Benchmark offline, live, repeatability, ablation và safety
* Theo dõi chi phí API và hỗ trợ resume benchmark bằng checkpoint
* Evaluation có calibration và held-out test

### Kết quả thực nghiệm nổi bật

Trong một confirmatory evaluation giới hạn ngân sách trên **70 fresh held-out paired queries**, TypeSafe Jev đạt quality non-inferiority so với LLM router, đồng thời:

* giảm **66.79% mean variable cost**
* giảm **53.95% p50 latency**
* giảm **8.82% p95 latency**
* đạt **0% unhandled errors**

Kết quả này chỉ áp dụng cho cấu hình dataset, prompt, provider và benchmark cố định trong repository.

---

# Động lực xây dựng dự án

Một pipeline RAG truyền thống thường:

1. luôn retrieval cho mọi query,
2. dùng cùng một retrieval strategy,
3. gửi mọi câu hỏi đến cùng một model,
4. ít kiểm tra xem context có thật sự hữu ích hay không.

Điều này có thể gây ra:

* tăng latency,
* tăng chi phí API,
* thêm retrieval noise,
* và làm hệ thống khó xử lý các failure case.

Dự án này tiếp cận RAG theo hướng khác:

> Xem RAG như một chuỗi quyết định thay vì một pipeline cố định.

Với mỗi query, hệ thống quyết định cần bao nhiêu retrieval và computation trước khi tạo final answer.

---

# Kiến trúc hệ thống

```text
User Query
    │
    ▼
┌─────────────────────┐
│     Pre-Router      │
│                     │
│ none / vector / web │
│ complexity estimate │
│ model tier          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      Retrieval      │
│                     │
│ Vector Search       │
│ Web Search          │
│ No Retrieval        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Context Assessment  │
└──────────┬──────────┘
           │
      insufficient?
       ┌───┴───┐
       │       │
      yes      no
       │       │
       ▼       │
 Retrieval     │
 Repair        │
(max 2 rounds) │
       │       │
       └───┬───┘
           ▼
┌─────────────────────┐
│   Answer Generation │
│                     │
│ Economy / Strong    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Post-generation Gate│
│                     │
│ Accept              │
│ Strong fallback     │
│ Abstain             │
└──────────┬──────────┘
           │
           ▼
      Final Answer
      + Typed Trace
```

---

# Các chiến lược Routing

Ba router dùng cùng một pipeline RAG để đảm bảo phép so sánh công bằng.

## TypeSafe Jev Router

Dùng TypeSafe Jev làm decision engine có cấu trúc.

Router chịu trách nhiệm dự đoán các quyết định như:

* có cần retrieval hay không,
* dùng vector hay web,
* độ phức tạp của query,
* model tier phù hợp.

Typed output giúp downstream pipeline không cần parse free-form text từ LLM.

---

## Rule-Based Router

Một baseline deterministic dựa trên các heuristic được định nghĩa thủ công.

Mục tiêu là tạo một baseline chi phí thấp để so sánh với các router phức tạp hơn.

---

## LLM Router

Dùng DeepSeek làm LLM-based router.

Đây là baseline linh hoạt hơn nhưng thường có chi phí và latency cao hơn.

---

# Adaptive Pipeline

Pipeline không chỉ chọn retrieval strategy.

Mỗi query có thể đi qua nhiều decision stage:

```text
Pre-routing
    ↓
Retrieval
    ↓
Context evaluation
    ↓
Retrieval repair
    ↓
Generation
    ↓
Post-generation decision
```

Nhờ đó hệ thống có thể xử lý khác nhau với:

* câu hỏi đơn giản,
* câu hỏi cần nhiều kiến thức,
* câu hỏi cần thông tin ngoài corpus,
* retrieval kém,
* evidence xung đột,
* provider lỗi.

---

# Retrieval

## Vector Retrieval

Vector retrieval sử dụng:

* **BAAI/bge-m3**
* **Qdrant**
* chunk size 512 token
* overlap 64 token

Corpus hiện tại chứa khoảng:

```text
21,460 source documents
36,582 indexed passages
```

BGE-M3 tự động sử dụng CUDA FP16 khi GPU khả dụng và fallback sang CPU khi cần.

---

## Web Retrieval

Live web retrieval sử dụng Tavily với:

* Basic Search
* local cache
* rate limiter
* retry control
* credit tracking
* hard budget cap

Formal offline benchmark dùng frozen web results để tăng khả năng tái lập.

---

# Evaluation Framework

Một trọng tâm quan trọng của project không chỉ là xây dựng RAG system mà còn là kiểm tra xem adaptive routing có thật sự tạo ra lợi ích đo được hay không.

Evaluation framework hỗ trợ:

* routing metrics
* answer-quality evaluation
* calibration
* counterfactual comparison
* latency tracking
* provider cost tracking
* safety evaluation
* repeatability analysis
* ablation studies

---

# Dataset

Benchmark kết hợp:

* **RAGRouter-Bench**
* **CRAG**

Thiết kế benchmark đầy đủ gồm:

```text
1,000 English queries

200 development
200 calibration
600 held-out test
```

Dữ liệu được chia theo group-aware split để giảm nguy cơ leakage.

Do giới hạn ngân sách API, full formal benchmark trên 600 held-out queries hiện chưa được hoàn tất.

---

# Confirmatory Experiment

Một protocol riêng được thiết kế để kiểm tra câu hỏi chính:

> TypeSafe Jev có thể giữ chất lượng câu trả lời trong khi giảm chi phí và latency so với LLM router hay không?

Experiment dùng các query mới chưa xuất hiện trong các pilot trước đó.

Group-sequential design được kiểm tra tại:

```text
40 queries
70 queries
100 queries
```

Experiment đạt predefined stopping condition tại **70 paired queries**.

| Metric             | Jev vs LLM Router |
| ------------------ | ----------------: |
| Mean variable cost |       **-66.79%** |
| p50 latency        |       **-53.95%** |
| p95 latency        |        **-8.82%** |
| Unhandled errors   |            **0%** |
| Quality            |      Non-inferior |

Chi phí total của confirmatory run:

```text
$0.5225
```

Run gồm:

```text
70 paired queries
140 adaptive records
0 failed records
```

Các kết quả này không nên được hiểu là TypeSafe Jev luôn tốt hơn trong mọi dataset, provider hoặc workload.

---

# Ablation Studies

Benchmark framework hỗ trợ các ablation:

```text
no-context-gate
no-repair
no-strong-fallback
```

Mục tiêu là đo ảnh hưởng của từng component đến:

* quality,
* routing behavior,
* cost,
* reliability.

Frozen benchmark artifacts có thể được tái sử dụng để tránh chạy lại toàn bộ counterfactual baseline.

---

# Reliability và Safety

Pipeline xử lý rõ ràng các failure case như:

* thiếu evidence,
* context yếu,
* provider timeout,
* generator failure,
* evidence xung đột.

Thay vì luôn cố tạo ra câu trả lời, pipeline có thể:

```text
retrieval repair
        ↓
strong-model fallback
        ↓
controlled abstention
```

Điều này giúp lỗi provider không bị ẩn đi dưới dạng answer không đáng tin cậy.

---

# Typed Decision Trace

Mỗi adaptive execution lưu một typed trace chứa các thông tin như:

```text
routing decision
retrieval source
model tier
context quality
repair attempts
provider cost
fallback decision
final status
```

Điều này giúp:

* debug routing,
* audit benchmark,
* so sánh các router,
* phân tích failure cases.

---

# Tech Stack

## Core

* Python 3.12
* Pydantic v2
* TypeSafe SDK
* DeepSeek API

## Retrieval

* BGE-M3
* Sentence Transformers
* Qdrant
* Tavily

## Evaluation

* Pandas
* PyArrow
* custom benchmark utilities
* calibration utilities

## Engineering

* uv
* Typer
* pytest
* mypy
* Ruff

---

# Cấu trúc dự án

```text
adaptive-rag-router-typesafe-jev/
│
├── configs/
│   ├── benchmark configs
│   ├── pilot configs
│   ├── ablation configs
│   └── confirmatory configs
│
├── data/
│   └── local datasets
│
├── docs/
│   ├── benchmark protocol
│   ├── technical report
│   ├── provider snapshot
│   └── dataset snapshot
│
├── src/
│   └── adaptive_rag_router/
│       ├── application/
│       ├── domain/
│       ├── evaluation/
│       ├── infrastructure/
│       └── interfaces/
│
├── tests/
│
├── artifacts/
│   └── generated benchmark outputs
│
├── pyproject.toml
├── README.md
└── SUMMARY.md
```

Các dữ liệu lớn, model cache, Qdrant index, secret và benchmark artifacts được loại khỏi Git.

---

# Cài đặt

Dự án dùng `uv` để quản lý dependency.

## Yêu cầu

* Python 3.12
* uv

Clone repository:

```bash
git clone https://github.com/wimaniac/adaptive-rag-router-typesafe-jev.git
cd adaptive-rag-router-typesafe-jev
```

Cài dependency:

```bash
uv sync --extra dev
```

Tạo file environment:

```bash
cp .env.example .env
```

Trên Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Sau đó cấu hình các API key cần thiết trong `.env`.

---

# Quick Start

Chạy query với rule-based router:

```bash
uv run rag-router query \
    --engine rule \
    --query "What is retrieval augmented generation?"
```

Chuẩn bị benchmark data:

```bash
uv run rag-router prepare-data \
    --ragrouter-source <RAGROUTER_PATH> \
    --crag-source <CRAG_PATH>
```

Build vector index:

```bash
uv run rag-router ingest \
    --source data/processed/corpus.jsonl \
    --recreate
```

Chạy smoke benchmark:

```bash
uv run rag-router benchmark \
    --config configs/smoke.toml
```

---

# Development

Chạy Ruff:

```bash
uv run ruff check .
```

Chạy type checking:

```bash
uv run mypy src
```

Chạy test:

```bash
uv run pytest
```

Trạng thái hiện tại:

```text
165 tests passed
3 live integration tests skipped by default
```

Các live provider tests được tắt mặc định vì có thể tiêu tốn API credits.

---

# Reproducibility

Repository tách riêng:

* source code,
* benchmark config,
* provider snapshot,
* dataset snapshot,
* generated artifacts.

Benchmark hỗ trợ checkpoint và idempotent resume.

Những record đã hoàn tất sẽ không bị chạy lại khi resume.

Chi phí provider cũng được theo dõi giữa các run để tránh vô tình gọi lại API và phát sinh chi phí không cần thiết.

---

# Giới hạn hiện tại

Đây là một research MVP, chưa phải production RAG service.

Một số giới hạn hiện tại:

* full formal 600-query held-out benchmark chưa hoàn tất,
* confirmatory result chỉ dựa trên 70 paired queries,
* benchmark hiện chủ yếu bằng tiếng Anh,
* live Tavily validation còn giới hạn,
* human evaluation chưa hoàn tất,
* kết quả phụ thuộc vào provider và dataset snapshot hiện tại.

Do đó, kết quả hiện tại cho thấy tính khả thi và bằng chứng thực nghiệm, nhưng không nên được xem là kết luận tổng quát cho mọi Adaptive RAG system.

---

# Hướng phát triển

Một số hướng mở rộng:

* hoàn thành full 600-query held-out evaluation
* external-validation dataset lớn hơn
* human answer-quality evaluation
* multilingual benchmark
* learned routing model
* multi-turn RAG
* GraphRAG
* multimodal retrieval
* production API deployment

---

# Câu hỏi nghiên cứu chính

Câu hỏi trung tâm của project là:

> **Một lightweight structured decision engine có thể thay thế LLM router trong Adaptive RAG mà không làm giảm chất lượng câu trả lời hay không?**

Kết quả thực nghiệm ban đầu cho thấy rằng trong benchmark hiện tại, cách tiếp cận này có thể giảm đáng kể chi phí inference và latency trong khi vẫn giữ được chất lượng câu trả lời.

Tuy nhiên, cần thêm đánh g
