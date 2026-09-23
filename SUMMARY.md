# Tóm tắt trạng thái dự án

## 1. Mục tiêu hoặc giai đoạn hiện tại

- Research MVP Adaptive RAG Router đã hoàn thành implementation, dữ liệu,
  Qdrant index, smoke, safety, offline/live/repeatability, ablation pilot và
  budget-constrained confirmatory study.
- Ba decision engine dùng cùng typed adaptive pipeline: TypeSafe Jev,
  rule-based và DeepSeek LLM router.
- Full formal 200 calibration + 600 test chưa chạy vì hard cap DeepSeek $2,79
  thấp hơn ước lượng $14,33–$25,08; pilot N=20 không thay thế formal conclusion.
- Primary end-to-end Jev-vs-LLM hypothesis được hỗ trợ ở corrected Look 2 trên
  70 fresh paired queries; kết luận không mở rộng sang routing optimality/ECE.
- Giai đoạn hiện tại: source snapshot đã được push lên private GitHub repository
  `wimaniac/adaptive-rag-router-typesafe-jev` trên `main`.

## 2. Công việc đã hoàn thành

- Python 3.12/`uv`, source layout, Pydantic v2, Typer, strict mypy, Ruff/Pytest.
- `.env` bị ignore; `.env.example` chỉ placeholder; secret dùng `SecretStr`.
- Typed contracts cho pre-route, context, repair, fallback, probability
  provenance, answer và trace.
- `JevRouter`, versioned `RuleBasedRouter`, structured `LLMRouter`; DeepSeek
  economy/strong generator và shared query rewriter.
- Adaptive vector/web/none loop, tối đa hai repair rounds, evidence gate,
  fallback, controlled abstention và provider failure safety.
- BGE-M3 CUDA/CPU fallback, Qdrant 36.582 passages, frozen mock-web và Tavily
  Basic adapter có cache, rate limiter, retry, credit ledger/checkpoint.
- Formal harness có group-aware split 200/200/600, six-branch counterfactual,
  temperature scaling, metrics, bootstrap, SVG plots và typed acceptance.
- Shared project provider ledger khóa DeepSeek $2,79 và Jev $3, reserve trước
  từng workflow và commit idempotent theo query/router.
- Dữ liệu chuẩn hóa: 7.727 RAGRouter, 2.706 CRAG, 2.706 mock-web và 21.460
  corpus documents; 9.328 supporting-document references đều resolve.
- Safety track: 100 variants/300 results, 0 errors. Human-audit packet/forms và
  adjudication harness đã sẵn sàng; chưa có annotation người thật.
- Repeatability comparator khóa baseline gold bằng SHA-256 và báo fixed/native
  gold accuracy; pilot phát hiện 8/20 query có generative-gold drift.
- Typed `PipelinePolicy` hỗ trợ `no-context-gate`, `no-repair` và
  `no-strong-fallback`, có E2E unit tests và provenance trong artifact.
- Frozen replay đọc counterfactual/calibration artifact baseline, không charge
  lại provider, kiểm tra đủ query trong preflight và ghi `frozen_gold_run_id`.
- Ba ablation pilot hợp lệ đều đạt 60/60 records, 0 failed/budget denial:
  $0,073096, $0,085603 và $0,115322.
- Paired baseline-vs-ablation artifacts ghép đủ 20 query, 0 frozen-gold drift;
  predicted-route agreement Jev 0,95 và rule/LLM 1,00.
- Confirmatory harness khóa manifest, loại pilot leakage, xen kẽ router order,
  chạy 40/70/100 looks với Bonferroni one-sided alpha và tái tạo look reports.
- `confirmatory-jev-vs-llm-100` dừng sớm tại Look 2: 140 records, 0 failed,
  quality lower +0,001743, cost −66,79%, p50 −53,95%, p95 −8,82%, error 0%.

## 3. Công việc đang thực hiện hoặc còn lại

- Full formal 200 calibration + 600 held-out bị chặn bởi hard budget hiện tại.
- Tavily formal live 100 phụ thuộc full formal calibration; mới có live pilot 20.
- Human audit 100 với hai annotator/adjudication chưa được thực hiện.
- Routing/calibration, repeatability, ablation và Pareto formal vẫn thiếu
  statistical power; primary conclusion chỉ có phạm vi fixed benchmark snapshot.

## 4. Các file/module chính đã sửa

- Root: `pyproject.toml`, `uv.lock`, `.gitignore`, `.env.example`, `README.md`.
- README bổ sung kết luận confirmatory, phạm vi tái lập trên máy mới; `.gitignore`
  bổ sung `.uv-cache/` trước khi stage.
- Config: baseline/smoke/live/budget/repeat, ba `budget-ablation-*.toml` và
  `configs/confirmatory.toml`.
- Application: `application/policy.py`, adaptive pipeline và runtime wiring.
- Evaluation: runner, calibration, counterfactual, cost budget/reconciliation,
  repeatability, confirmatory, safety, human audit, metrics và visualization.
- Interface: confirmatory CLI, fresh-sample parser, preflight và runtime wiring.
- Docs: benchmark protocol, provider/dataset snapshots, technical report và
  completion audit.
- Runtime artifacts: baseline/live/repeat, ba valid ablation run directories và
  `confirmatory-jev-vs-llm-100` với manifest/report theo từng look.

## 5. Quyết định kỹ thuật và lý do

- Full config giữ nguyên; mọi pilot dùng run-id riêng và
  `publish_acceptance=false`.
- Frozen replay tái dùng gold/calibration baseline để so ablation trên cùng mục
  tiêu và tránh trả phí lại six-branch counterfactual.
- Confirmatory amendment loại counterfactual khỏi primary endpoint, dùng fresh
  paired quality/cost/latency và alpha correction để giữ statistical validity.
- `needs_retrieval = 1 - P(NONE)`; raw/calibrated/self-reported/heuristic
  probability có provenance riêng.
- Strong model không thay thế evidence; thiếu evidence/provider failure abstain.
- Frozen/mock web dùng tạo gold; Tavily thật chỉ dùng external-validity track.
- Artifact cũ thiếu provider breakdown được double-count bảo thủ; ledger có thể
  overestimate nhưng không thể làm lỏng hard cap.
- Một `no-context-gate` diagnostic run không có network access tạo 60 controlled
  abstentions cost 0; artifact được giữ để audit nhưng bị loại khỏi phân tích.
  Config hiện trỏ tới run `budget-ablation-no-context-gate-20-v2` hợp lệ.

## 6. Lệnh chạy và kết quả gần nhất

- `uv run --no-sync ruff format --check .`: pass, 103 files formatted.
- `uv run --no-sync ruff check .`: pass.
- `uv run --no-sync mypy src`: pass, 62 source files.
- `uv run --no-sync pytest -q`: 165 passed, 3 live provider tests skipped đúng
  theo opt-in flag.
- Git stage audit: 121 file được chọn; không có `.env`, dataset, Qdrant index,
  model cache, benchmark artifacts hoặc credential pattern trong staged snapshot.
- Initial commit `9b9051c` chứa đúng 121 file đã audit; local tree sạch.
- GitHub private repository đã tạo; `origin/main` khớp local `main` tại commit
  `60dbf38` sau lần push đầu tiên. Remote tree có 121 file.
- Baseline `budget-pilot-2-79`: 60/60 adaptive, cost tổng run $0,645933.
- Live `budget-live-20`: 60/60, 40 Tavily credits, cost $0,299152.
- Repeat `budget-repeat-20`: 60/60, cost $0,359934.
- Ablations: no-context-gate v2 $0,073096; no-repair $0,085603;
  no-strong-fallback $0,115322; mỗi run 60/60 và 0 failed.
- Jev quality baseline/context-gate/repair/fallback: 0,1163 / 0,1240 / 0,1046 /
  0,1038. Đây là descriptive N=20, không phải formal inference.
- Confirmatory run cost $0,522523: DeepSeek $0,490809, Jev $0,031713.
- Provider ledger: 1.024 idempotent units; DeepSeek $2,194741/$2,79, còn
  $0,595259; Jev $0,219577/$3, còn $2,780423.

## 7. Lỗi, rủi ro, giả định hoặc câu hỏi mở

- N=20 và calibration N=10 không đủ statistical power; ECE/CI biến động lớn.
- Full formal cost vượt ngân sách; đây là blocker của N=600, không phải lỗi code.
- Confirmatory N=70 đủ cho primary corrected stopping rule nhưng không cung cấp
  gold route, ECE, critical recall hoặc Tavily external validity.
- Offline/live baseline được tạo trước query-rewrite metering; historical cost
  thấp hơn actual một lượng nhỏ, nên không dùng làm exact pricing baseline.
- Qdrant local cảnh báo >20.000 points; đúng chức năng nhưng latency kém server.
- CRAG license CC BY-NC 4.0; mục đích sử dụng phải phù hợp.
- Tavily monthly usage ngoài repository không nằm trong ledger run.
- Dataset, Qdrant index và confirmatory artifacts không được push; người clone
  repository cần chuẩn bị dữ liệu và tạo lại artifacts theo README.

## 8. Bước tiếp theo cụ thể

1. Giữ manifest, frozen calibration và confirmatory artifacts bất biến để tái
   lập kết luận trong phạm vi benchmark snapshot hiện tại.
2. Nếu cần claim rộng hơn: bổ sung ngân sách cho formal N=600, Tavily live100
   và human audit100; primary confirmatory conclusion không phụ thuộc các bước đó.
