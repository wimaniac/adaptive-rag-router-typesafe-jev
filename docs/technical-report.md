# Báo cáo kỹ thuật Adaptive RAG Router

## Trạng thái báo cáo

Báo cáo này ghi nhận kiến trúc, dữ liệu, smoke validation, safety exploratory,
budgeted offline/live/repeatability/ablation pilot và budget-constrained
confirmatory study đã hoàn tất. Primary end-to-end Jev-vs-LLM hypothesis được
hỗ trợ trong phạm vi fixed benchmark snapshot. Full routing study 200
calibration + 600 held-out vẫn chưa chạy vì ngân sách DeepSeek $2,79.

## 1. Câu hỏi nghiên cứu

Hệ thống kiểm tra liệu một decision model nhỏ như TypeSafe Jev có thể giảm
variable cost và latency của RAG/LLM pipeline mà vẫn giữ answer quality tương
đương LLM router hay không. Jev được so sánh với:

- rule-based router có version và threshold cố định;
- DeepSeek Flash router trả structured JSON cùng self-reported distribution.

Ba router dùng chung retrieval, generator, prompts, query rewriter, budget và
evaluation harness. Khác biệt duy nhất cần đo là decision engine và các quyết
định mà engine tạo ra.

## 2. Kiến trúc hệ thống

Luồng xử lý:

```text
query
  → pre-route: none/vector/web + complexity + economy/strong
  → retrieval và tối đa hai repair rounds
  → context assessment theo passage
  → grounded draft
  → accept / strong regeneration / abstain
  → answer, citations và typed trace
```

`needs_retrieval` luôn được suy từ `1 - P(NONE)`. Raw distribution, calibrated
distribution, confidence và probability provenance được lưu riêng. Strong model
không được dùng để thay thế evidence còn thiếu.

## 3. Dữ liệu và retrieval

Snapshot ngày 2026-09-22 gồm:

- 7.727 câu hỏi RAGRouter-Bench;
- 2.706 câu hỏi CRAG và 2.706 frozen web records;
- 21.460 tài liệu vector corpus;
- 36.582 passages BGE-M3 sau chunk 512 token, overlap 64.

Trong RAGRouter-Bench, 3.414 câu có 9.328 supporting-document references và
không có reference nào thiếu parent document trong corpus. Fixed split gồm 58
calibration và 186 held-out queries có annotation để tính citation/supporting-
fact metrics. Chunk ID được canonicalize về parent document trước khi chấm;
raw passage lineage vẫn được giữ riêng trong record.

Formal sample dùng 700 RAGRouter-Bench và 300 CRAG queries, chia group-aware
thành 200 dev, 200 calibration và 600 held-out test. SHA-256 và license được ghi
trong `docs/dataset-snapshot.md`.

Vector retrieval dùng BAAI/bge-m3 1.024 chiều, cosine search và Qdrant local.
RTX 3050 chạy FP16; CPU fallback dùng float32. Frozen/mock web không tiêu Tavily
và giữ formal run tái lập được.

## 4. Gold route và calibration

Mỗi query được chạy sáu counterfactual branches:

```text
{NONE, VECTOR, MOCK_WEB} × {ECONOMY, STRONG}
```

Gold action là branch rẻ nhất đạt quality floor 0,70. Nếu không branch nào đạt,
chọn branch có quality cao nhất và đánh dấu `no_good_route`. Temperature scaling
được fit riêng theo router chỉ trên calibration split. Tavily live chỉ được đọc
calibration artifact formal đã khóa.

Orchestrator bắt buộc hoàn tất calibration trước khi tạo held-out
counterfactual artifacts. Các ngưỡng preregistered trong `[acceptance]` được
parse vào typed benchmark config; trạng thái pass/fail chỉ được xuất cho formal
held-out test.

## 5. Xác minh vận hành

Smoke benchmark `smoke-dev-3` đã hoàn tất:

- 3 queries, 9 adaptive records;
- 18/18 counterfactual branches thành công;
- 0 provider/schema error;
- counterfactual cost $0,026156;
- adaptive cost $0,027580;
- tổng cost $0,053735.

V4 Pro ban đầu trả content rỗng vì hidden reasoning dùng hết completion cap
512/1.024. Runtime hiện giữ cap cấu hình cho economy và dành cho strong
`max(4 × cap, 2048)`, tối đa 8.192 token. Smoke sạch sau thay đổi này.

Smoke report không đánh giá acceptance criteria vì đây là dev split. Các metric
mô tả với N=3 không được xem là kết quả nghiên cứu.

## 6. Safety exploratory

Track gồm 100 held-out queries, cân bằng 25 variants cho mỗi loại noise,
conflict, stale evidence và prompt injection. Mỗi router chỉ chạy context gate;
không gọi answer generator hoặc Tavily.

| Router | N | Pass rate | Cost (USD) | p50/p95 (ms) | Errors |
|---|---:|---:|---:|---:|---:|
| Jev | 100 | 0,6100 | 0,002814 | 408,64 / 1.523,38 | 0 |
| Rule | 100 | 0,1600 | 0,000000 | 0,09 / 0,15 | 0 |
| LLM | 100 | 0,5600 | 0,034743 | 1.245,37 / 1.513,78 | 0 |

Pass rate theo loại:

| Router | Noise | Conflict | Stale | Prompt injection |
|---|---:|---:|---:|---:|
| Jev | 0,72 | 0,88 | 0,36 | 0,48 |
| Rule | 0,00 | 0,00 | 0,64 | 0,00 |
| LLM | 0,80 | 0,32 | 0,52 | 0,60 |

Pass tổng yêu cầu đồng thời đúng context quality và mọi required ID. Khi tách
tín hiệu, cả ba router đạt 1,00 ở injection rejection. Rule pass tổng thấp chủ
yếu do quality và accepted-evidence matching, không phải do bỏ lọt injection.

Kết quả safety mang tính exploratory, không được trộn vào primary answer quality
hoặc non-inferiority test.

## 7. Human audit

Harness chọn 100 held-out queries có đủ ba router, random hóa answers thành
candidate A/B/C và tách `blind-key.jsonl`. Hai annotator chấm độc lập correctness,
groundedness, citation support và unsupported claim trên thang đã preregister.
Adjudication chỉ thực hiện sau khi hai annotation files được khóa.

Packet chính thức chưa thể tạo trước khi formal records tồn tại. Smoke packet ba
query đã xác minh schema, số dòng và không chứa router field.

## 8. Kết quả budgeted pilot và Tavily live

Full formal preflight vẫn `ready=true` nhưng ước tính $14,33–$25,08, vượt hard
budget DeepSeek $2,79. Vì vậy full config được giữ nguyên và không chạy. Thay vào
đó, config pilot riêng lấy stratified subset 10 calibration + 20 held-out,
chạy tuần tự, tắt acceptance và dùng ledger $2,79.

### 8.1 Offline pilot

Run `budget-pilot-2-79` hoàn tất 10 calibration, 20 held-out, 30/30 calibration
records và 60/60 test records; 0 failed, không dừng sớm. Recorded cost là
$0,645933. Human-audit packet mù 20 query đã được tạo nhưng chưa có hai annotator
thật.

| Router | Quality | Cost/query | p50 / p95 ms | Critical recall | ECE | Error |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 0,1163 | $0,002356 | 3.955 / 17.390 | 0,9231 | 0,1173 | 0,0000 |
| Rule | 0,1442 | $0,000413 | 2.031 / 2.739 | 1,0000 | 0,0659 | 0,0000 |
| LLM | 0,0924 | $0,005699 | 8.651 / 28.719 | 0,6923 | 0,0868 | 0,0000 |

Paired descriptive result Jev − LLM: quality +0,0239, cost −$0,0033/query và
latency −4.905 ms. Đây không phải non-inferiority conclusion: calibration model
chỉ fit trên 10 queries, bootstrap chỉ 1.000 lần và test N=20.

### 8.2 Tavily live pilot

Run `budget-live-20` dùng cùng 20 queries và calibration artifact pilot, hoàn
tất 60/60 records, 0 failed, 40 Basic Search credits, cost $0,299152 và không có
429/budget denial. USD cap riêng là $1,50; Tavily cap là 180 credits.

| Router | Quality | Cost/query | p50 / p95 ms | Critical recall | ECE | Credits |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 0,1522 | $0,002203 | 6.768 / 21.836 | 0,9167 | 0,0671 | 13 |
| Rule | 0,1457 | $0,000900 | 3.486 / 9.237 | 1,0000 | 0,0659 | 13 |
| LLM | 0,1131 | $0,005292 | 13.459 / 36.069 | 0,5833 | 0,2434 | 12 |

### 8.3 Repeatability pilot

Run `budget-repeat-20` dùng cùng seed, strata và 20 held-out queries của offline
pilot, nhưng dùng run-id/checkpoint mới. Run hoàn tất 10 calibration, 20 test,
60/60 adaptive records, 0 failed, không dừng sớm và ghi $0,359934. Run mới đã
meter cả query rewrite.

| Router | Quality | Cost/query | p50 / p95 ms | Critical recall | ECE |
|---|---:|---:|---:|---:|---:|
| Jev | 0,0884 | $0,000996 | 3.429 / 15.607 | 0,9231 | 0,1677 |
| Rule | 0,1515 | $0,000304 | 2.199 / 3.183 | 1,0000 | 0,0659 |
| LLM | 0,0956 | $0,003599 | 9.229 / 39.367 | 1,0000 | 0,1593 |

Comparator ghép đủ 20 queries/60 records, không có record chỉ xuất hiện ở một
run. Route agreement Jev/Rule/LLM là 0,90/1,00/0,95; status agreement là
0,90/1,00/0,95; answer token-F1 giữa hai run là 0,7869/0,6410/0,8607. Gold-route
agreement chỉ 0,60 cho cả ba vì gold được sinh lại bằng six-branch generative
counterfactual; đây là một nguồn variance quan trọng của protocol. Comparator
mới khóa baseline gold bằng SHA-256 và ghi nhận drift 8/20 query. Khi chấm cả
hai run trên fixed baseline gold, accuracy Jev/Rule/LLM giữ nguyên lần lượt
0,30/0,25/0,30; native repeat-gold accuracy là 0,25/0,25/0,35.

Paired descriptive Jev − LLM trong repeat run: quality −0,0071, cận dưới một
phía −0,0218, cost −$0,0026/query và latency −8.104 ms. Giá trị cận dưới thấp
hơn ngưỡng −0,02 một lượng nhỏ; với N=20 và 1.000 bootstrap samples, không được
coi đây là formal failure hay success.

Ledger project-wide đã reconcile 86 counterfactual units, 258 adaptive units và
300 safety units, gồm cả smoke thử nghiệm trước lần sửa strong-token budget.
Upper bound bảo thủ sau confirmatory run là DeepSeek $2,194741/$2,79 và Jev
$0,219577/$3; còn lại lần lượt $0,595259 và $2,780423. Artifact cũ chưa có stage breakdown được
charge toàn bộ cho DeepSeek và, với Jev workflow, đồng thời cho Jev. Vì vậy số
liệu có thể double-count nhưng không thể làm lỏng hard cap. Resume E2E từ toàn
bộ checkpoint trước ablation giữ nguyên 644 project ledger entries và không replay
provider. Sau bốn lần chạy ablation (gồm một diagnostic network failure cost 0),
ledger có 884 idempotent entries. Confirmatory run thêm 140 exact-breakdown
entries, đưa ledger lên 1.024 entries.

Mỗi benchmark report hiện tạo ba biểu đồ SVG tái lập: quality–cost Pareto,
reliability và risk–coverage. Các plot pilot phục vụ kiểm tra pipeline báo cáo,
không thay thế Pareto conclusion từ held-out formal.

Typed ablation policy đã triển khai và kiểm thử cho ba biến thể bắt buộc:
no-context-gate, no-repair và no-strong-fallback. Frozen replay dùng đúng 20
held-out IDs, counterfactual gold và calibration artifact của
`budget-pilot-2-79`; do đó mỗi run chỉ trả phí adaptive phase.

| Policy | Run cost | Jev quality/cost | Rule quality/cost | LLM quality/cost |
|---|---:|---:|---:|---:|
| baseline | — | 0,1163 / $0,002356 | 0,1442 / $0,000413 | 0,0924 / $0,005699 |
| no-context-gate | $0,073096 | 0,1240 / $0,000648 | 0,1381 / $0,000308 | 0,1278 / $0,002699 |
| no-repair | $0,085603 | 0,1046 / $0,001643 | 0,1414 / $0,000289 | 0,1040 / $0,002348 |
| no-strong-fallback | $0,115322 | 0,1038 / $0,001494 | 0,1471 / $0,000311 | 0,1136 / $0,003961 |

Mỗi ablation có 60/60 records, 0 failed và không budget denial. Với Jev, bỏ
repair hoặc strong fallback làm quality giảm khoảng 0,0117–0,0125 so với pilot
baseline; bỏ context gate tăng 0,0077 nhưng đây là variance mô tả ở N=20, không
phải bằng chứng rằng gate có hại. Provider-backed ablation formal N=600 vẫn chưa
chạy vì hard budget hiện tại.

Paired comparison artifacts xác nhận 20/20 query được ghép, 0 gold drift và
cùng baseline gold SHA-256 cho cả ba run. Predicted-route agreement Jev là 0,95;
rule/LLM là 1,00. Vì router call vẫn được thực hiện lại, khác biệt answer và
latency còn chứa provider variance ngoài tác động của component ablation.

### Confirmatory conclusion trong hard budget

Protocol amendment khóa 100 fresh held-out queries nhưng cho phép dừng ở ba look
40/70/100. Manifest 100 query không trùng pilot có SHA-256
`8f7b400b41b6ca5606ff8fe157d03ccbd2b1943951dd9f39af4eacf123695f40`.
Không có calibration/counterfactual provider call mới; chỉ Jev và LLM router
được chạy với thứ tự AB/BA.

Look 1 đạt quality/cost/p50/error nhưng p95 regression +71,01%, nên quyết định
đúng protocol là `continue`. Look 2 đạt toàn bộ tiêu chí và dừng sớm ở 70 cặp:

| Metric xác nhận | Kết quả | Ngưỡng | Trạng thái |
|---|---:|---:|---|
| Quality Jev − LLM | +0,026088 | — | mô tả |
| Corrected one-sided lower | +0,001743 | ≥ −0,02 | đạt |
| Mean cost reduction | 66,79% | ≥ 20% | đạt |
| p50 latency reduction | 53,95% | ≥ 15% | đạt |
| p95 latency regression | −8,82% | ≤ 5% | đạt |
| Jev unhandled error | 0% | < 0,5% | đạt |

Run có 140 records, 0 failed, tổng cost $0,522523: DeepSeek $0,490809 và Jev
$0,031713. Kết luận primary hypothesis là **supported** trên fixed corpus,
prompt, model alias và provider snapshot hiện tại.

Chưa được kết luận các claim rộng hơn sau đây nếu chưa có full formal study:

- calibrated ECE và critical-class recall trên 600 held-out queries;
- six-branch optimal routing và Pareto conclusion formal;
- external validity trên Tavily live100;
- human audit/adjudication.

## 9. Giới hạn hiện tại

- Qdrant local cảnh báo khi collection vượt 20.000 points; kết quả đúng nhưng
  latency có thể kém server deployment.
- Mock web bảo đảm tái lập nhưng không mô phỏng đầy đủ freshness/noise/network
  behavior của Tavily.
- CRAG dùng license CC BY-NC 4.0.
- Provider aliases, pricing và behavior có thể thay đổi; snapshot phải giữ cố
  định trong một benchmark run.
- Budgeted N=20 không đủ power; confirmatory N=70 chỉ xác nhận primary
  Jev-vs-LLM system hypothesis, không thay thế routing study N=600.
- Query-rewrite usage đã được đưa vào typed trace/cost cho các run tiếp theo.
  Offline/live pilot được tạo trước thay đổi này nên ledger lịch sử chưa gồm rewrite
  cost; chúng không nên
  dùng hai ledger cũ làm pricing baseline chính xác tuyệt đối.

## 10. Lệnh tái lập

```powershell
uv run --no-sync rag-router prepare-data --ragrouter-source <path> --crag-source <path>
uv run --no-sync rag-router ingest --source data/processed/corpus.jsonl --recreate --batch-size 128
uv run --no-sync rag-router benchmark-preflight --config configs/benchmark.toml
uv run --no-sync rag-router benchmark --config configs/smoke.toml
uv run --no-sync rag-router benchmark --config configs/budget-pilot.toml
uv run --no-sync rag-router benchmark --config configs/budget-repeat.toml
uv run --no-sync rag-router compare-runs --baseline-run budget-pilot-2-79 --repeat-run budget-repeat-20
uv run --no-sync rag-router prepare-confirmatory --config configs/confirmatory.toml
uv run --no-sync rag-router benchmark-preflight --config configs/confirmatory.toml
uv run --no-sync rag-router benchmark-confirmatory --config configs/confirmatory.toml
uv run --no-sync rag-router report-confirmatory --config configs/confirmatory.toml
uv run --no-sync rag-router benchmark --config configs/budget-ablation-no-context-gate.toml
uv run --no-sync rag-router benchmark --config configs/budget-ablation-no-repair.toml
uv run --no-sync rag-router benchmark --config configs/budget-ablation-no-strong-fallback.toml
uv run --no-sync rag-router reconcile-cost-budget --run-id smoke-dev-3-pre-strong-budget --run-id smoke-dev-3 --run-id budget-pilot-2-79 --run-id budget-live-20 --run-id budget-repeat-20 --safety-results artifacts/safety/safety-100/results.jsonl
uv run --no-sync rag-router benchmark-live --config configs/budget-live.toml
uv run --no-sync rag-router benchmark --config configs/benchmark.toml
uv run --no-sync rag-router prepare-safety --config configs/benchmark.toml
uv run --no-sync rag-router benchmark-safety --config configs/benchmark.toml
uv run --no-sync rag-router prepare-audit --config configs/benchmark.toml --run-id formal-1000
uv run --no-sync rag-router benchmark-live --config configs/live.toml --tavily-credit-budget 350
```
