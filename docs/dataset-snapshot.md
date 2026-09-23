# Snapshot dữ liệu benchmark

Snapshot này ghi lại input đã dùng để chuẩn hóa dữ liệu ngày **2026-09-22**.
Các file nguồn và output lớn nằm trong `.gitignore`; manifest giúp kiểm tra một
lần tái lập có dùng đúng bytes hay không.

## Nguồn

- RAGRouter-Bench: repository
  [Chaplain0908/RAGRouter](https://huggingface.co/datasets/Chaplain0908/RAGRouter).
- CRAG Task 1/2 development v4:
  [facebookresearch/CRAG](https://github.com/facebookresearch/CRAG).
- CRAG được phát hành theo CC BY-NC 4.0; người chạy phải tự xác nhận mục đích sử
  dụng phù hợp license.

## SHA-256 của input

| File | Bytes | SHA-256 |
|---|---:|---|
| `crag/crag_task_1_and_2_dev_v4.jsonl.bz2` | 739,384,484 | `AFA29F2B3FACFB5D15AA9CDED00D5EC90FF76F3E67279E7B99CFE86659A641CA` |
| `graphragBench_medical/Corpus.json` | 1,055,124 | `66E792DDB429409E461F7491DA1EAFD9C297CF07306AEB5F6C9114BD317F9A81` |
| `graphragBench_medical/Question.json` | 1,053,136 | `365FD51B9B15DE84040657593E99B64021A6C362280FCA4F734A957636269464` |
| `musique/Corpus.json` | 11,320,817 | `1AB56F734729DEC791CEE247B000B1BB3BF91D2DA70B06E532A086ECB5567050` |
| `musique/Question.json` | 6,027,199 | `1F67E046B7913A5DF705BE1873E352926F4BED2A334CCF2677B345992F7E98CD` |
| `quality/Corpus.json` | 6,736,347 | `43F33F01E974E22D299714AB0E876708FBC995ACBAEBB30F142DAD49349CEF6F` |
| `quality/Question.json` | 7,188,885 | `917D3CA76FBB01E8B29D0220DED48FA64262BE9322063983CBEBD1267E873097` |
| `ultraDomain_legal/Corpus.json` | 22,269,679 | `61DD83D5634642983A53DAA2A781C9EF415AFE41362DA7E1AF29BF967582DEF5` |
| `ultraDomain_legal/Question.json` | 9,385,807 | `6F57CBE02E4205D3F2E699E44B6810547410FE59857A51A58D453FD82830CFB7` |

## Kết quả chuẩn hóa

Lệnh `rag-router prepare-data` tạo:

- 7.727 câu hỏi RAGRouter-Bench.
- 2.706 câu hỏi CRAG và 2.706 frozen/mock-web records.
- 21.460 tài liệu corpus RAGRouter-Bench.
- 36.582 passages sau chunking bằng tokenizer `BAAI/bge-m3`, cửa sổ 512 token
  và overlap 64.

| Artifact chuẩn hóa | Records | Bytes | SHA-256 |
|---|---:|---:|---|
| `data/raw/ragrouter/questions.jsonl` | 7.727 | 24.817.175 | `81528CD887350C390E7D73D58B99AEC8BD420DC929E6C67665DD117229BB699C` |
| `data/raw/crag/questions.jsonl` | 2.706 | 1.049.835 | `16485D3B8DAF4D3FEBCEE2C7BDB6043EF00C32DC1A623C97124EECB706B3E0FF` |
| `data/raw/crag/mock_web.jsonl` | 2.706 | 111.955.604 | `2D0535A7C4BF400B989AC8448AFFA182FF78E2C213CCF1AB639A3C60F4944CA0` |
| `data/processed/corpus.jsonl` | 21.460 | 43.088.571 | `8B47DC32864B8D6E516E1993E6A172B5870EE1224ABF829101C8A87CD8E5DFE2` |

RAGRouter-Bench cung cấp `doc_id` có cấu trúc cho 3.414 câu hỏi, tương ứng
9.328 supporting-document references; toàn bộ reference đều ánh xạ được sang
parent document trong corpus. Với seed/split formal đã khóa, 58 calibration và
186 held-out queries có annotation này. Metric citation và supporting-fact chỉ
tính trên các sample có annotation; passage ID dạng `#chunk-XXXX` được
canonicalize về parent document ID trước khi so sánh.

CRAG web page được giới hạn 8.192 ký tự mỗi result để giữ context và artifact
trong giới hạn kiểm soát. CRAG web pages không được đưa vào vector corpus nhằm
giữ hai nguồn `VECTOR` và `MOCK_WEB` tách biệt trong counterfactual benchmark.
