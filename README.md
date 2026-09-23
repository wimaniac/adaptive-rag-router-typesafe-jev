# Adaptive RAG Router with TypeSafe Jev

Dự án nghiên cứu một pipeline RAG thích ứng, trong đó decision engine quyết định
có retrieval hay không, chọn vector/web, chọn model tier, đánh giá context và
fallback sau draft. Ba router được so sánh trên cùng hạ tầng: TypeSafe Jev,
rule-based và DeepSeek LLM router.

Benchmark xác nhận tiết kiệm chi phí đã dừng hợp lệ tại 70 cặp truy vấn mới:
Jev đạt quality non-inferiority so với LLM router, giảm 66,79% mean variable
cost và 53,95% p50 latency. Phạm vi kết luận, cách tính và giới hạn được ghi ở
[báo cáo kỹ thuật](docs/technical-report.md) và
[giao thức benchmark](docs/benchmark-protocol.md).

Repository chỉ chứa mã nguồn, cấu hình, test và tài liệu. `.env`, dataset gốc,
Qdrant index, model cache, benchmark records và cost ledger là dữ liệu local,
không được đẩy lên GitHub. Vì vậy các lệnh benchmark cần chạy `prepare-data`,
`ingest` và tạo lại artifacts liên quan trước khi sử dụng trên máy mới. Không
dùng `configs/confirmatory.toml` trực tiếp để tuyên bố tái lập kết quả nếu chưa
có frozen calibration và baseline artifacts tương ứng.

## Kiến trúc

```text
query
  → pre-route (none/vector/web, complexity, model tier)
  → retrieval + tối đa 2 repair rounds
  → context assessment
  → draft answer
  → accept / strong-model fallback / abstain
  → answer + typed decision trace
```

Các provider nằm sau Protocol/adapter nên benchmark thay router mà không thay
retrieval, generator hoặc prompt. Mọi distribution đều lưu provenance; score
heuristic hoặc do LLM tự khai báo không bị gắn nhãn nhầm là calibrated
probability.

## Yêu cầu

- Python 3.12.
- `uv`.
- GPU là tùy chọn; `BAAI/bge-m3` dùng CUDA FP16 khi khả dụng và tự fallback
  về CPU float32.
- Ba API key trong `.env`: TypeSafe, DeepSeek và Tavily.

```powershell
Copy-Item .env.example .env
uv sync --extra dev
```

Không commit `.env`. Ứng dụng dùng `SecretStr` và không đưa key vào log, trace
hoặc benchmark artifact.

### Thiết lập CUDA tùy chọn trên Windows

Lockfile mặc định giữ khả năng cài đa nền tảng. Trên máy NVIDIA, có thể thay
wheel CPU trong virtualenv bằng wheel CUDA chính thức; lệnh dưới đây là cấu hình
đã kiểm tra với Python 3.12, PyTorch 2.14 và CUDA 13.0:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv pip install --python .venv\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu130 `
  "torch==2.14.0+cu130"
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Sau override này, dùng `uv run --no-sync ...` cho ingestion/benchmark để `uv`
không đồng bộ lại wheel CPU từ lockfile. Chọn wheel phù hợp driver trên
[hướng dẫn cài PyTorch chính thức](https://docs.pytorch.org/get-started/locally/)
nếu môi trường khác cấu hình trên.

## Chạy ứng dụng

```powershell
uv run rag-router query --engine rule --query "What is retrieval augmented generation?"
uv run rag-router prepare-data --ragrouter-source <RAGROUTER_PATH> --crag-source <CRAG_JSONL_BZ2>
uv run rag-router ingest --source data/processed/corpus.jsonl --recreate
uv run rag-router benchmark-preflight --config configs/benchmark.toml
uv run rag-router benchmark --config configs/smoke.toml
uv run rag-router benchmark-preflight --config configs/budget-pilot.toml
uv run rag-router benchmark --config configs/budget-pilot.toml
uv run rag-router benchmark --config configs/budget-repeat.toml
uv run rag-router compare-runs --baseline-run budget-pilot-2-79 --repeat-run budget-repeat-20
uv run rag-router benchmark --config configs/budget-ablation-no-context-gate.toml
uv run rag-router benchmark --config configs/budget-ablation-no-repair.toml
uv run rag-router benchmark --config configs/budget-ablation-no-strong-fallback.toml
uv run rag-router prepare-confirmatory --config configs/confirmatory.toml
uv run rag-router benchmark-preflight --config configs/confirmatory.toml
uv run rag-router benchmark-confirmatory --config configs/confirmatory.toml
uv run rag-router report-confirmatory --config configs/confirmatory.toml
uv run rag-router reconcile-cost-budget `
  --run-id smoke-dev-3 --run-id budget-pilot-2-79 `
  --run-id budget-live-20 --run-id budget-repeat-20 `
  --safety-results artifacts/safety/safety-100/results.jsonl
uv run rag-router benchmark --config configs/benchmark.toml
uv run rag-router prepare-safety --config configs/benchmark.toml
uv run rag-router benchmark-safety --config configs/benchmark.toml
uv run rag-router prepare-audit --config configs/benchmark.toml --run-id formal-1000
uv run rag-router benchmark-live --config configs/live.toml --tavily-credit-budget 350
uv run rag-router benchmark-preflight --config configs/budget-live.toml --live
uv run rag-router benchmark-live --config configs/budget-live.toml
uv run rag-router report --run-id <run-id>
```

`query` có thể dùng Jev hoặc DeepSeek khi API key tương ứng hợp lệ. Benchmark
formal dùng frozen/mock web results; Tavily thật chỉ dành cho smoke test và live
subset.

## Chuẩn bị dữ liệu

Tải dữ liệu từ nguồn chính thức và tự kiểm tra license trước khi sử dụng:

- [RAGRouter-Bench trên Hugging Face](https://huggingface.co/datasets/Chaplain0908/RAGRouter)
  hoặc [repository tác giả](https://github.com/ziqiwang0908/RAGRouter-Bench).
- [CRAG của Facebook Research](https://github.com/facebookresearch/CRAG), ưu tiên
  Task 1/2 vì có tối đa 5 web results/query và phù hợp frozen-web track.

`prepare-data` không download và không sửa source. Lệnh này tạo:

```text
data/raw/ragrouter/questions.jsonl
data/raw/crag/questions.jsonl
data/raw/crag/mock_web.jsonl
data/processed/corpus.jsonl
```

Sau đó `ingest` chunk corpus bằng đúng tokenizer BGE-M3, cửa sổ 512 token,
overlap 64 và ghi Qdrant local. Trên RTX 3050 8 GB, batch 128 FP16 đã được kiểm
tra dưới giới hạn VRAM:

```powershell
$env:HF_HOME = ".cache\huggingface"
uv run --no-sync rag-router ingest `
  --source data/processed/corpus.jsonl --recreate --batch-size 128
```

Sau lần tải model đầu tiên, có thể đặt `HF_HUB_OFFLINE=1` và
`TRANSFORMERS_OFFLINE=1` để tránh kiểm tra mạng không cần thiết. Dữ liệu, model
và index lớn đều bị `.gitignore`.

## Tavily free tier

- Rolling-window limiter mặc định 90 requests/60 giây, thấp hơn giới hạn 100 RPM.
- Concurrency tối đa 4.
- Chỉ dùng Basic Search, tối đa 5 kết quả.
- Cache được kiểm tra trước khi gọi API.
- Live run có hard credit cap mặc định 350 và lưu checkpoint.
- Mỗi query-router workflow bị chặn ở tối đa 3 Tavily attempts, gồm retry.
- Không tự động chuyển sang gói trả phí.

## Benchmark

Thiết kế formal gồm 1.000 English queries: 700 RAGRouter-Bench và 300 CRAG,
chia 200 dev, 200 calibration và 600 held-out test theo group để tránh leakage.
Offline counterfactual chạy sáu nhánh:

```text
{none, vector, mock_web} × {economy, strong}
```

Các metrics chính gồm routing macro-F1, Brier/NLL/ECE, context sufficiency,
answer quality, citation support, variable cost và p50/p95/p99 latency. Mỗi run
tạo thêm `pareto.svg`, `reliability.svg` và `risk-coverage.svg` trực tiếp từ
report typed, không cần dependency đồ họa. Không dùng held-out test để chỉnh
prompt, rule hoặc threshold.

Với RAGRouter samples có `doc_id` chuẩn, pipeline giữ
`supporting_document_ids`, canonicalize vector chunk về parent document và suy
ra gold context quality từ supporting-document coverage. Các sample không có
annotation evidence bị loại khỏi citation/retrieval metric thay vì được gán
điểm mặc định.

Khi chạy split `test`, CLI tự động chạy split `calibration` bằng frozen
providers, fit temperature scaling riêng cho từng router và lưu
`calibration-models.json`. Held-out records lưu cả raw và calibrated
distribution để audit. `benchmark-live` chỉ đọc artifact formal đã khóa; nó từ
chối chạy nếu artifact chưa tồn tại nên 100 query Tavily không bao giờ được dùng
để tune calibration.

Các ngưỡng trong `[acceptance]` được parse thành typed config và đóng gói cùng
run. Chỉ formal held-out `test` ở chế độ offline được phép phát hành trạng thái
acceptance; dev/smoke, calibration, safety và Tavily live chỉ báo metric mô tả.

Lần chạy formal đầu tiên tốn nhiều provider calls vì phải tạo gold từ sáu nhánh
counterfactual cho cả calibration và test. Mỗi query counterfactual được ghi
atomic vào `counterfactual.jsonl` hoặc `calibration-counterfactual.jsonl`.
Adaptive records được ghi sau mỗi dispatch batch vào
`adaptive-records-checkpoint.jsonl` hoặc
`calibration-records-checkpoint.jsonl`. Chạy lại cùng `run-id` sẽ bỏ qua record
đã hoàn tất và chỉ retry record có trạng thái `failed`, nhờ đó crash không làm
phát sinh lại toàn bộ provider cost. Giữ nguyên prompt/config khi resume; nếu
thay đổi thiết kế thí nghiệm, dùng `run-id` mới thay vì tái sử dụng checkpoint.

Chạy `benchmark-preflight` trước benchmark có chi phí. Lệnh chỉ đọc config,
filesystem và artifacts; nó không khởi tạo model/SDK client hay gọi provider.
Preflight kiểm tra key readiness không lộ secret, Qdrant/mock-web/calibration,
validate checkpoint theo run/query/router và chỉ tính non-failed records là có
thể resume. Khi `ready=false`, CLI trả exit code 2.

Smoke 3 mẫu gần nhất tiêu **$0,053735** cho counterfactual và adaptive
workflows; preflight nội suy cho 800 mẫu calibration+test là khoảng
**$14,33–$25,08**, tùy độ dài reasoning/context. Đây chỉ là ước lượng, không
phải hard cap của provider. Hãy kiểm tra lại trước khi xóa cache hoặc đổi
`run-id`.

Khi ngân sách DeepSeek bị giới hạn ở **$2,79**, dùng
`configs/budget-pilot.toml`: 10 calibration + 20 held-out stratified queries,
chạy tuần tự và ghi `cost-budget-checkpoint.json` sau từng đơn vị. Pilot đã hoàn
tất 0 failed records với cost ghi nhận **$0,645933**; acceptance bị tắt vì N=20
không thay thế formal N=600. `configs/budget-live.toml` chạy cùng 20 query trên
Tavily thật, có đồng thời hard cap **$1,50** và 180 credits; run hoàn tất với
**$0,299152**, 40 Basic Search credits và 0 failed records.
`configs/budget-repeat.toml` lặp lại đúng sample set offline với run-id riêng;
run hoàn tất 0 failed records với **$0,359934**. Ledger project-wide tách hard
cap DeepSeek **$2,79** và Jev **$3**, dùng chung giữa mọi run-id. Sau khi
reconcile cả smoke thử nghiệm, các budgeted run và safety track, upper bound bảo
thủ là DeepSeek **$2,194741** và Jev **$0,219577**; phần còn lại lần lượt
**$0,595259** và **$2,780423**. Artifact cũ thiếu provider breakdown được charge
bảo thủ cho toàn bộ provider có thể liên quan, nên ledger không thể đánh giá
thiếu chi phí dù có thể double-count.

`compare-runs` ghép record theo query/router và ghi JSON/Markdown về route/gold/
status agreement, answer token-F1, quality delta, cost delta và latency delta.
Repeatability pilot N=20 đạt predicted-route agreement Jev/Rule/LLM lần lượt
0,90/1,00/0,95; đây vẫn chỉ là bằng chứng mô tả, không thay thế formal N=600.
Comparator còn khóa baseline gold bằng SHA-256 và báo riêng accuracy của repeat
run trên fixed baseline gold. Pilot phát hiện 8/20 query bị counterfactual-gold
drift; fixed-gold accuracy Jev/Rule/LLM đều giữ nguyên 0,30/0,25/0,30 giữa hai
run, tránh nhầm gold variance với router variance.

Mọi budget config trỏ tới cùng `provider-cost-budget.json`. Counterfactual được
charge cho DeepSeek; adaptive record mới lưu `provider_costs_usd` từ typed stage
trace. `benchmark-preflight` hiển thị consumed/remaining của cả ledger từng run
và ledger toàn dự án. Resume commit lại record theo khóa idempotent nhưng không
gọi provider hoặc cộng trùng cost.

Formal harness hỗ trợ ba typed ablation: `no-context-gate`, `no-repair` và
`no-strong-fallback`. Cấu hình qua `[ablation]`; record và report luôn ghi policy
để không trộn baseline với ablation. Định nghĩa chính xác nằm trong
[giao thức benchmark](docs/benchmark-protocol.md).

Ba budgeted ablation pilot dùng frozen counterfactual gold và calibration của
`budget-pilot-2-79`, nên không trả phí lại sáu nhánh gold. Các run hợp lệ đều có
20 query/60 records, 0 failed và lần lượt tiêu $0,073096 (`no-context-gate`),
$0,085603 (`no-repair`) và $0,115322 (`no-strong-fallback`). Đây chỉ là paired
pilot N=20; full formal ablation vẫn phụ thuộc ngân sách N=600. Một lần chạy
`no-context-gate` không có network access được giữ làm artifact chẩn đoán và bị
loại khỏi phân tích; config hiện trỏ tới run `-v2` hợp lệ.

### Budget-constrained confirmatory track

`configs/confirmatory.toml` là protocol amendment dành riêng cho giả thuyết
end-to-end Jev-vs-LLM khi không đủ ngân sách chạy formal 200+600. Track khóa 100
held-out query mới, loại mọi query pilot, không tạo counterfactual/calibration
mới và chạy group-sequential ở 40/70/100 cặp. Bonferroni chia one-sided alpha
0,05 cho ba look; thứ tự Jev/LLM được đảo theo query để giảm order bias.

Run `confirmatory-jev-vs-llm-100` dừng sớm hợp lệ ở Look 2 với 70/70 cặp,
140 records, 0 failed và decision **supported**. Corrected quality lower bound
là `0,001743` so với margin `-0,02`; mean cost giảm `66,79%`, p50 latency giảm
`53,95%`, p95 latency giảm `8,82%` và unhandled error `0%`. Run tiêu tổng
`$0,522523`, gồm DeepSeek `$0,490809` và Jev `$0,031713`; Look 3 không chạy.

Kết luận này chỉ áp dụng cho fixed corpus/prompt/provider snapshot và primary
end-to-end hypothesis. ECE, critical-route recall, six-branch optimal routing,
Tavily live100 và generalization rộng vẫn là exploratory/chưa được chứng minh.

Strong-generation provider failure được chuyển thành controlled abstention có
trace, không còn trở thành unhandled benchmark error. Retry cùng `run-id` thay
placeholder cost 0 một cách idempotent và không cộng trùng ledger.

V4 Pro dùng chung completion budget cho hidden reasoning và final answer. Runtime
vì vậy giữ cap cấu hình cho economy, còn strong tier dự phòng
`max(4 × cap, 2048)` và chặn ở 8192 token; hidden reasoning không bao giờ được
dùng thay cho answer text.

`prepare-safety` chỉ tạo 100 fixture offline, cân bằng bốn loại noise, conflict,
stale fact và prompt injection. `benchmark-safety` mới gọi Jev/DeepSeek để chạy
context gate; kết quả được tách khỏi primary benchmark. Sau formal run,
`prepare-audit` tạo packet mù, hai form annotator, form adjudication và một
`blind-key.jsonl`; không đưa blind key cho annotator trước khi khóa annotation.

Chi tiết tái lập nằm trong [giao thức benchmark](docs/benchmark-protocol.md).
Model alias, giá và giới hạn provider được khóa trong
[provider snapshot](docs/provider-snapshot.md). Hash, license và số lượng record
của dữ liệu thực nghiệm nằm trong [dataset snapshot](docs/dataset-snapshot.md).
Kết quả đã xác minh và các mục còn chờ formal run được tổng hợp trong
[báo cáo kỹ thuật](docs/technical-report.md) và
[completion audit](docs/completion-audit.md).

Raw datasets, Qdrant index và artifacts lớn không được commit. Cấu hình chỉ trỏ
tới dữ liệu local đã được tải hợp pháp.

## Kiểm thử và chất lượng

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
```

Integration tests gọi provider thật được đánh dấu `integration` và không chạy
mặc định trong workflow offline.

Smoke test chủ động (một request TypeSafe, một DeepSeek và tối đa một Tavily
Basic credit):

```powershell
$env:RUN_LIVE_INTEGRATION = "1"
uv run pytest tests/integration/test_provider_smoke.py -q
```

## Phạm vi MVP

MVP cung cấp Python API và CLI. FastAPI, production deployment, benchmark tiếng
Việt, multi-turn, multimodal, GraphRAG và fine-tune/distill Jev nằm ngoài phạm
vi hiện tại.
