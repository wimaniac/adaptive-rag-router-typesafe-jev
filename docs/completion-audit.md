# Completion audit theo kế hoạch gốc

Ngày audit: 2026-09-23. Trạng thái được đối chiếu trực tiếp với `PLAN.md`, source,
tests, configs và runtime artifacts; pilot không được dùng để thay thế formal.

| Hạng mục | Trạng thái | Bằng chứng chính |
|---|---|---|
| Nền tảng Python 3.12/uv/source layout | Hoàn tất | `pyproject.toml`, `uv.lock`, `src/`, `tests/` |
| Bảo mật `.env`/SecretStr | Hoàn tất | `.gitignore`, `.env.example`, settings tests |
| Typed decisions và ba routers | Hoàn tất | `domain/`, `routing/`, cross-router tests |
| Vector/mock/Tavily retrieval | Hoàn tất | `retrieval/`, 36.582 Qdrant points, adapter tests |
| Tavily free-tier guardrails | Hoàn tất | 90 RPM, concurrency cap, cache, retry, ledger tests |
| Adaptive repair/context/fallback | Hoàn tất | `application/pipeline.py`, E2E unit tests |
| Strong provider failure safety | Hoàn tất | controlled abstention; offline/live pilot 0 failed |
| Six-branch counterfactual | Hoàn tất | counterfactual runner/checkpoints và pilot artifacts |
| Calibration/metrics/bootstrap/report/plots | Hoàn tất về code | unit tests, reports và ba SVG/run |
| Safety exploratory 100 variants | Hoàn tất | 300 records, 0 errors |
| Formal dataset 1.000/split 200-200-600 | Hoàn tất chuẩn bị | normalized data, manifests, preflight ready |
| Full formal 200 calibration + 600 test | Chưa chạy | ước lượng $14,33–$25,08 > DeepSeek cap $2,79 |
| Formal routing acceptance | Chưa thể kết luận | cần full held-out; pilot chủ động tắt acceptance |
| Budget-constrained confirmatory | Hoàn tất, supported | 70 paired fresh queries; Bonferroni-corrected Look 2 |
| Primary Jev-vs-LLM hypothesis | Được hỗ trợ trong phạm vi hẹp | quality lower +0,001743; cost −66,79%; p50 −53,95% |
| Tavily live 100 queries | Chưa chạy | phụ thuộc full formal artifact và provider budget |
| Tavily live pilot 20 queries | Hoàn tất | 60/60 records, 40 credits, 0 failed |
| Human audit 100, 2 annotators | Chưa thực hiện | harness hoàn tất; mới tạo packet pilot 20 |
| Repeatability pilot N=20 | Hoàn tất | 60/60 records, 0 failed, typed comparator artifact |
| Repeatability formal | Chưa chạy | phụ thuộc full formal sample/power |
| Project-wide provider hard caps | Hoàn tất | shared ledger, 1.024 idempotent units |
| Required ablation harness | Hoàn tất về code | 3 typed policies, E2E unit tests, report provenance |
| Ablation pilot N=20 | Hoàn tất | 3 frozen-replay runs, mỗi run 60/60 records và 0 failed |
| Formal ablation results | Chưa chạy | phụ thuộc full formal budget và frozen gold |
| Báo cáo kỹ thuật/tái lập/SUMMARY | Hoàn tất cho trạng thái hiện tại | docs và README đã cập nhật |

## Budget evidence

- Offline pilot: recorded $0,645933, 0 failed.
- Live pilot: recorded $0,299152, 40/180 Tavily credits, 0 failed.
- Repeatability pilot: recorded $0,359934, 0 failed.
- Ba ablation pilot: $0,073096 + $0,085603 + $0,115322, 0 failed.
- Confirmatory run: $0,522523; DeepSeek $0,490809; Jev $0,031713; 0 failed.
- Project ledger bảo thủ: DeepSeek $2,194741/$2,79; Jev $0,219577/$3.
- Còn lại theo ledger: DeepSeek $0,595259; Jev $2,780423.
- Hai pilot chạy trước khi query-rewrite metering được bổ sung, nên recorded cost
  không phải exact provider invoice. Dù quy toàn bộ recorded cost và vùng thiếu
  cho DeepSeek, khoảng cách hơn $1,8 tới cap $2,79 vẫn đủ lớn cho các rewrite
  calls tối đa 128 output tokens đã quan sát.
- Jev-router totals còn bao gồm DeepSeek generation, vì vậy là upper bound bảo
  thủ cho Jev và thấp hơn cap Jev $3.
- Artifact cũ thiếu provider breakdown được double-count có chủ đích; resume
  commit idempotent và không replay provider.

## Kết luận audit

Implementation MVP và budget-constrained primary validation đã hoàn tất. Trên
70 fresh paired queries, Jev đạt corrected non-inferiority bound, cost, p50,
p95 và error criteria; primary end-to-end hypothesis được đánh dấu `supported`
trong fixed benchmark scope. Kế hoạch nghiên cứu rộng vẫn chưa hoàn tất theo
formal protocol: thiếu routing/calibration N=600, Tavily live100 và human audit
thật. Không dùng kết luận hẹp này để tuyên bố routing optimality hoặc production
generalization.
