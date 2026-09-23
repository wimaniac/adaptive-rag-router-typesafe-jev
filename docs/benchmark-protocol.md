# Giao thức benchmark

## Dataset và split

Benchmark formal dùng đúng 1.000 English queries: 700 mẫu RAGRouter-Bench và
300 mẫu CRAG. Dữ liệu được nạp từ file local JSON/JSONL/Parquet; loader không tự
download và manifest lưu SHA-256 của từng file. Sampling được stratify theo
metadata nguồn, sau đó chia theo `entity/corpus/question-family` thành:

- dev: 200;
- calibration: 200;
- held-out test: 600.

Một `group_id` chỉ được xuất hiện trong một split. Prompt, rule, threshold và
calibrator phải khóa trước khi chạy held-out test.

CLI formal chạy calibration split trước, fit temperature scaling độc lập cho
từng router và lưu artifact có version. Distribution raw vẫn được giữ trong
record metadata. Tavily live chỉ áp artifact formal đã khóa, không refit.
Counterfactual gold của held-out test chỉ bắt đầu sau khi calibration artifact
đã hoàn tất. Các threshold `[acceptance]` được nạp vào typed run config; runner
chỉ đánh giá pass/fail trên formal held-out test, không đánh giá trên dev,
calibration, safety hoặc Tavily live.

## Tạo gold route

Nhãn route gốc của dataset không được dùng trực tiếp. Với mỗi query, runner
counterfactual đánh giá sáu nhánh:

```text
{none, vector, mock_web} × {economy, strong}
```

Gold action là nhánh rẻ nhất đạt quality floor. Nếu không nhánh nào đạt floor,
chọn nhánh có quality cao nhất rồi đánh dấu `no_good_route=true`. Tie-break phải
ổn định theo cost, latency, retrieval source và model tier.

### Checkpoint và resume

Formal run ghi counterfactual atomically sau từng query và adaptive records sau
mỗi dispatch batch. Các artifact dùng để resume trong thư mục của `run-id` là:

- `counterfactual.jsonl` và `calibration-counterfactual.jsonl` cho six-branch
  gold generation;
- `adaptive-records-checkpoint.jsonl` và
  `calibration-records-checkpoint.jsonl` cho ba router;
- `checkpoint.json` và `calibration-checkpoint.json` cho tiến độ tổng hợp.

Khi chạy lại cùng `run-id`, record hoàn tất/abstained/degraded được giữ nguyên;
chỉ record `failed` được gọi provider lại. Loader từ chối query/router ngoài job
hiện tại, ID trùng hoặc counterfactual query ngoài split. Checkpoint không thay
thế experiment fingerprint: prompt, model catalog, threshold hoặc config thay
đổi phải dùng `run-id` mới để tránh trộn hai phiên bản thí nghiệm.

`rag-router benchmark-preflight` phải chạy trước formal/live execution. Lệnh
validate schema cùng identity của checkpoint, chỉ đếm non-failed adaptive
records, kiểm tra prerequisite và nội suy cost từ smoke artifact mà không gọi
provider. Exit code 0 nghĩa là sẵn sàng; `ready=false` trả exit code 2. Cost
estimate không phải provider-side hard cap.

### Budgeted pilot

Khi operator đặt hard provider budget thấp hơn ước lượng formal, full config
không được sửa hoặc chạy một phần rồi diễn giải như formal. Config pilot riêng
phải dùng `stratified_subset=true`, giới hạn độc lập calibration/test,
`publish_acceptance=false`, `max_concurrency=1` và section `[cost_budget]`.
Ledger USD dùng khóa idempotent cho từng counterfactual query và adaptive
query-router record; reserve được kiểm tra trước provider calls, checkpoint được
ghi atomic sau mỗi đơn vị. Tavily live pilot ghép đồng thời USD ledger và Tavily
credit ledger; một hook chấp nhận không được bỏ qua hook còn lại.

Kết quả pilot chỉ dùng để kiểm tra workflow, phát hiện failure mode và ước lượng
effect direction. Nó không thay thế 200 calibration + 600 held-out formal,
10.000 bootstrap resamples hoặc human audit 100 mẫu.

## Ba track

1. **Formal offline** — frozen corpus và CRAG mock web; đây là track duy nhất
   dùng cho kết luận quality, routing, calibration, cost và ablation.
2. **Adaptive replay** — phát lại snapshot/cache nhưng chạy đủ context gate,
   tối đa hai repair rounds và strong fallback.
3. **Tavily live subset** — 100 query được interleave giữa ba router, chỉ đo
   network latency, retry/rate-limit và external validity; không tune threshold.

### Ablation bắt buộc

Ba component ablations được preregister và phải chạy trên đúng held-out IDs,
frozen counterfactual gold, calibration artifact, prompt và model snapshot của
formal baseline. Mỗi ablation dùng run-id riêng và không phát hành acceptance:

- `no-context-gate`: không gọi context assessment, chấp nhận toàn bộ evidence
  từ retrieval đầu; vì context được coi sufficient nên không chạy repair;
- `no-repair`: vẫn đánh giá context nhưng dừng sau retrieval đầu, không gọi
  repair decision hoặc query rewrite;
- `no-strong-fallback`: vẫn chạy fallback gate, nhưng khi gate yêu cầu strong
  regeneration thì giữ economy draft để đo đúng marginal benefit của strong
  fallback.

TOML dùng section typed sau; field không khai báo mặc định là `true`:

```toml
[ablation]
context_gate_enabled = true
repair_enabled = false
strong_fallback_enabled = true
```

Artifact record lưu `ablation_id` và toàn bộ `pipeline_policy`; report ghi policy
ở phần tổng quan. Kết luận ablation dùng paired quality/cost/latency delta, thay
đổi retrieval/fallback/abstention rate và không được suy từ observational subset.
Budget pilot dùng `[replay].source_run_id` để đọc frozen `counterfactual.jsonl`
và `calibration-models.json` của baseline. Replay từ chối source thiếu artifact,
không charge lại frozen phase và ghi `frozen_gold_run_id` vào từng record. Ba
ablation pilot N=20 đã chạy; chúng kiểm tra harness nhưng không thay thế formal
N=600.

## Protocol amendment: budget-constrained confirmatory track

Amendment ngày 2026-09-23 tách giả thuyết end-to-end Jev-vs-LLM khỏi formal
routing study 200+600. Amendment không thay đổi hay xóa formal protocol; nó tạo
một track xác nhận hẹp hơn khi hard provider budget không đủ chạy six-branch
gold trên 800 query.

- Sampling frame là held-out test split gốc, cùng seed `20260921`.
- 100 query được stratify sau khi loại mọi query từng xuất hiện trong baseline,
  live, repeatability và ablation pilot.
- Manifest được khóa trước provider call, có SHA-256
  `8f7b400b41b6ca5606ff8fe157d03ccbd2b1943951dd9f39af4eacf123695f40`.
- Chỉ chạy Jev và LLM router; không tạo calibration hoặc counterfactual mới.
- Frozen calibration của `budget-pilot-2-79` chỉ giữ threshold cố định; không
  dùng để kết luận ECE.
- Group-sequential looks là 40, 70 và 100 paired queries. One-sided family-wise
  alpha 0,05 được chia Bonferroni thành 0,016667 mỗi look.
- Thứ tự router đảo AB/BA theo query, concurrency một, prompt/model/retrieval và
  fallback policy giữ nguyên.
- Dừng sớm chỉ khi quality lower bound đạt margin -0,02, cost/p50/p95 đạt ngưỡng
  preregistered, đủ paired records và Jev unhandled error dưới 0,5%.
- Run cap là $0,95; shared DeepSeek $2,79/Jev $3 ledger vẫn là lớp chặn cuối.

Run đã dừng ở Look 2 (70 cặp) với decision `supported`; Look 1 và Look 2 được
giữ thành artifact riêng. Kết luận chỉ áp dụng cho primary end-to-end hypothesis,
không mở rộng sang routing calibration/optimality hoặc Tavily live.

## Metric và thống kê

- Routing: macro-F1, balanced accuracy, confusion matrix, ordinal MAE/kappa,
  one-vs-rest AUROC/AUPRC và critical-class recall.
- Calibration: Brier, NLL, ECE, reliability bins và risk–coverage.
- Retrieval/context: supporting-fact recall, context-quality macro-F1 và recall
  của lớp `insufficient`.
- Answer: native correctness, exact match, token-F1, grounded correctness và
  citation precision/recall.
- System: token/cost/query, retrieval/fallback/abstention/error rate và
  p50/p95/p99 latency.
- So sánh paired dùng stratified bootstrap 10.000 lần, 95% CI.

Lớp critical được preregister là action dùng `web` hoặc model tier `strong`.
Non-inferiority dùng cận dưới một phía 95%; report vẫn giữ CI hai phía 95% cho
phân tích mô tả. Supporting-fact/citation metrics chỉ được tính khi normalized
dataset cung cấp `supporting_document_ids`; thiếu annotation được hiển thị là
không đánh giá, không tự gán điểm hoàn hảo. Vector passage ID được quy về
`parent_document_id` trước khi đối chiếu với annotation; artifact vẫn giữ riêng
raw passage IDs để audit lineage. Gold context quality được suy ra từ coverage:
đủ mọi supporting document là `sufficient`, có một phần là `partial`, không có
support là `insufficient`.

Human audit dùng 100 held-out samples, hai annotator độc lập và adjudication.
Safety exploratory dùng 100 biến thể context có noise, conflict, stale facts và
prompt injection; kết quả không trộn vào primary quality metric.

### Human audit

`rag-router prepare-audit` stratify 100 query từ những held-out records có đủ ba
router. Mỗi query random hóa answer thành candidate A/B/C bằng seed khóa trước.
Packet của annotator không chứa router; mapping nằm riêng trong
`blind-key.jsonl`. Hai annotator chấm độc lập theo thang 1–5 cho correctness,
groundedness và citation support, đồng thời đánh dấu unsupported claim. Chỉ mở
blind key sau khi hai file annotation đã khóa; adjudicator điền kết luận cuối
trong `adjudication.jsonl`.

### Safety exploratory

`rag-router prepare-safety` tạo đúng 100 variants cân bằng 25 mẫu mỗi loại:

- noise: giữ support đúng và thêm passage không liên quan;
- conflict: thêm nguồn phủ định material claim;
- stale: chỉ cung cấp archived evidence có cảnh báo outdated;
- prompt injection: thêm chỉ dẫn độc hại bên cạnh support đúng.

`rag-router benchmark-safety` chỉ chạy `assess_context` của ba router, ghi pass
theo quality cùng required accepted/conflicting/rejected IDs. Track này có
provider calls nhưng không gọi answer generator hoặc Tavily, và không được trộn
vào primary quality/non-inferiority result.

## Tiêu chí thành công

- One-sided lower 95% CI của `quality_Jev - quality_LLM >= -0.02`.
- Mean variable cost của Jev giảm ít nhất 20% so với LLM router.
- p50 latency giảm ít nhất 15%, p95 không tăng quá 5%.
- Critical-class recall ít nhất 0.90; ECE sau calibration không quá 0.08.
- Unhandled router/schema error dưới 0.5%.
- Jev cải thiện quality–cost Pareto frontier so với rule-based router.

Live/mocked results phải được báo cáo tách biệt. Không diễn giải mock-web như
ước lượng trực tiếp cho độ mới, độ nhiễu hoặc latency của Tavily production.
