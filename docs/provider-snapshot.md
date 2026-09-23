# Snapshot provider và cost model

Snapshot này được khóa ngày **2026-09-22** để một benchmark không thay đổi cách
tính cost khi provider cập nhật model hoặc giá giữa các lần chạy. Runtime vẫn
ghi `model` thực tế từ response để phát hiện alias bị đổi tuyến.

## TypeSafe Jev

- Model benchmark được pin: `jev-1.13.0`; không dùng alias `jev-latest`.
- SDK đã khóa trong `uv.lock`: `typesafe-sdk==0.7.0`.
- Endpoint mặc định: `https://api.typesafe.ai`.
- Giá snapshot: $0.042/một triệu input token; output token miễn phí.
- Giới hạn công bố tại thời điểm snapshot: 1.200 requests/phút và 250.000
  tokens/giây; benchmark vẫn dùng concurrency nhỏ hơn nhiều.
- Một lời gọi `system_one` batch nhiều câu hỏi `Choice`/`Noul` trên cùng state.
- Distribution do Jev trả về được lưu với provenance `native`; chỉ distribution
  qua calibrator mới được ghi là `calibrated`.

TypeSafe mô tả Jev là System One model nhận state và trả quyết định typed thay vì
sinh prose: [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
Model, giá và rate limit được lấy từ
[TypeSafe Models](https://docs.typesafe.ai/models).

## DeepSeek

| Tier logic | Model request | Thinking | Peak cache-miss input | Peak cache-hit input | Peak output |
|---|---|---:|---:|---:|---:|
| Economy | `deepseek-flash` | tắt | $0.30/M | $0.006/M | $1.20/M |
| Strong | `deepseek-v4-pro` | `high` | $1.32/M | $0.044/M | $3.96/M |

Cost model cố ý dùng **peak price** để so sánh bảo thủ và tái lập. DeepSeek có
peak/off-peak pricing, vì vậy hóa đơn thực tế có thể thấp hơn. Tài liệu chính
thức hiện liệt kê cả hai model và các mức giá trên:
[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/).

DeepSeek đã có thông báo không hoàn toàn nhất quán về vòng đời V4 Pro trong
tháng 9/2026. Vì vậy benchmark không giả định alias luôn trỏ tới cùng weights:
request model và response model đều phải được lưu trong manifest/trace, prompt
và thresholds phải khóa trước held-out run.

## Tavily

- Free Researcher: 1.000 credits/tháng, không yêu cầu thẻ.
- Development key: 100 requests/phút.
- Basic Search: 1 credit; Advanced Search: 2 credits.
- MVP chỉ gửi `search_depth=basic`, `max_results<=5`, không dùng Extract API.
- Guardrail nội bộ: rolling window 90 requests/60 giây, concurrency 4, cache
  trước request, retry 429 tối đa 4 lần và mọi attempt đều ghi credit ledger.
- Live benchmark có hard cap 350 credits; checkpoint được ghi trước provider
  attempt nên crash không làm mất dấu credit đã dành.

Nguồn: [Tavily pricing](https://www.tavily.com/pricing),
[rate limits](https://help.tavily.com/articles/3240802908-rate-limits), và
[Basic vs. Advanced Search](https://help.tavily.com/articles/6938147944-basic-vs-advanced-search-what-s-the-difference).

## Quy tắc cập nhật

Không sửa giá hoặc alias giữa một benchmark run. Khi provider thay đổi, tạo
snapshot/version cost model mới, chạy lại calibration, rồi mới bắt đầu held-out
test. Không dùng kết quả held-out để chỉnh thresholds.
