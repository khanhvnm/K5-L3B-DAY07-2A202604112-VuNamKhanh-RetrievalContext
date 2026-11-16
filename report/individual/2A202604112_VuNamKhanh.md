# Báo cáo Day 7 - Agentic RAG

## Dữ liệu và benchmark

- Nguồn: corpus Docker từ Day 6, strategy B; 427 chunks và 24 câu hỏi.
- Vector store: PostgreSQL + pgvector, embedding
  `paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, cosine similarity.
- Agent model: OpenRouter `openai/gpt-4o-mini`; dev 17 câu (15 answerable,
  2 thiếu evidence), test 7 câu (5 answerable, 2 thiếu evidence).
- Câu thiếu evidence được đánh dấu `answerable=false`; agent phải search rồi
  abstain. Citation phải trỏ tới chunk có trong context.

## Retrieval và thử nghiệm

Các kết quả benchmark agent bên dưới được tạo trước khi thêm bootstrap độc lập,
trên corpus Day 6 gồm 427 chunks. Đường chạy standalone hiện đã được kiểm tra
đến bước retrieval/index; agent dev/test cần chạy lại bằng corpus standalone
trước khi dùng làm kết quả nộp cuối.

| Cấu hình | Phương pháp / cấu hình | Split | Hit@1 | Evidence Hit@3 | Thời gian |
|---|---|---:|---:|---:|---:|
| Dense | pgvector, top-3 | dev | 5/15 (33.3%) | 7/15 (46.7%) | 45.35 ms |
| Baseline | TF-IDF, top-3 | dev | 3/15 (20.0%) | 5/15 (33.3%) | 0.98 ms |
| Hybrid | Dense + TF-IDF, RRF | dev | 5/15 (33.3%) | 9/15 (60.0%) | 45.19 ms |
| Cải thiện 1 | Hybrid, top-5, budget 2400 | dev | Không chọn | Không chọn | Provider errors |
| Cải thiện 2 | Hybrid, top-3, budget 3200, tối đa 2 search | dev | Agent 15/15 answerable | Agent 2/2 abstain đúng | 0 provider error |
| Cấu hình đã chọn | Hybrid, top-3, budget 3200, tối đa 2 search | test | Agent 4/5 answerable | 2/2 abstain đúng | 0 provider error |
| Baseline | TF-IDF, top-3, budget 3200 | test | 1/5 (20.0%) | 1/5 (20.0%) | 0.85 ms |
| Standalone dense | pgvector Day 7, top-3 | dev | 2/15 (13.3%) | 2/15 (13.3%) | 49.96 ms |
| Standalone hybrid | Dense + TF-IDF, RRF | dev | 2/15 (13.3%) | 2/15 (13.3%) | 42.50 ms |
| Standalone TF-IDF | TF-IDF, top-3, budget 3200 | dev | 1/15 (6.7%) | 4/15 (26.7%) | 1.59 ms |
| Standalone TF-IDF | TF-IDF, top-3, budget 3200 | test | 0/5 (0%) | 0/5 (0%) | 1.44 ms |

Standalone agent rerun: dev trả lời `14/15` câu answerable và abstain đúng `2/2`;
test trả lời `4/5` câu answerable, abstain đúng `2/2`, provider errors `0`.
`q18` là false abstention do hybrid top-3 không lấy được evidence. Context
Evidence Rate test là `4/5`; grounded answer `0:1, 1:0, 2:4`, trung bình
`1.6/2`. Median latency test là `5045.21 ms`; chi phí provider không được API
trả về nên ghi `N/A`.

Hai lỗi dev:

1. `q02` false abstention do evidence chưa vào context; tăng budget và cho
   phép thêm lượt search đã sửa lỗi.
2. `q14` có citation ID hợp lệ nhưng model quên marker trong answer; verifier
   được bổ sung bước append marker khi ID đã tồn tại trong context.

`q18` vẫn là false abstention trên test: hybrid top-3 không lấy được chunk
`docker-engine-security`. Đây là giới hạn recall được giữ nguyên, không dùng
test để điều chỉnh cấu hình.

## Agent và đánh giá

- Mỗi câu test dùng 1 search; search tối đa là 2.
- Context được giới hạn 3200 ký tự, loại chunk trùng ID.
- Citation không hợp lệ hoặc không có evidence dẫn tới abstain; lỗi provider
  được ghi riêng.
- Human review test cũ: grounded answer `0:1, 1:1, 2:3`, trung bình `1.4/2`.
- Human review standalone test: grounded answer `0:0, 1:0, 2:4`, trung bình `2.0/2`
  trên các case không lỗi.
- Citation support: `6/7`; answer có citation: `4/4`.
- Abstention đúng: `2/2`; false abstention: `1/5` câu answerable.

| Query | Loại | Kết quả | Nhãn |
|---|---|---|---|
| q16 | Answerable | Answer + citation | Grounded 2 |
| q17 | Answerable | Answer + 2 citations | Grounded 2; 1 citation không hỗ trợ trực tiếp |
| q18 | Answerable | Abstain | Grounded 0; false abstention |
| q19 | Answerable | Answer + 3 citations | Grounded 2 |
| q20 | Answerable | Answer + citation | Grounded 1; trả lời một phần |
| q23 | Thiếu evidence | Abstain | Đúng |
| q24 | Thiếu evidence | Abstain | Đúng |

Trace và kết quả:

- Dev: `starter/results/agent-dev-hybrid-budget3200-v2.json`
- Test: `starter/results/agent-test-hybrid-budget3200.json`
- Nhãn review: `starter/results/agent-test-hybrid-budget3200-reviews.json`
- Metrics: `starter/results/agent-test-hybrid-budget3200-summary.json`
- Standalone dev: `starter/results/agent-standalone-dev.json`
- Standalone test: `starter/results/agent-standalone-test-v2.json`
- Standalone review/metrics: `starter/results/agent-standalone-test-v2-reviews.json`,
  `starter/results/agent-standalone-test-v2-summary.json`

## Thử thách nâng cao

Không thực hiện bảng/ảnh/biểu đồ/multi-hop riêng. Phạm vi là text Docker
documentation; giới hạn chính là recall của hybrid top-3 và chất lượng model
khi evidence không nằm trong context. Có thể tái lập dữ liệu bằng
`starter/prepare_data.py` và chạy benchmark theo hướng dẫn trong
`starter/README.md`.
