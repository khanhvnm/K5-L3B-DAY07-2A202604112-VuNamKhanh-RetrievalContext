# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |
| `prepare_data.py` | Tái lập chunk Day 6 và tạo benchmark Day 7 24 câu |
| `dense_retriever.py` | Dense retriever thật cho PostgreSQL + pgvector Day 6 |
| `retrieval_methods.py` | Lexical TF-IDF và hybrid RRF retriever |
| `benchmark_retrieval.py` | Chạy so sánh ba phương pháp trên cùng split |
| `agent_controller.py` | Controller gọi model chat thật và chuẩn hóa action |
| `run_agent_cli.py` | Chạy agent thật cho một câu hỏi |
| `run_agent_benchmark.py` | Chạy agent trên toàn bộ một split và lưu trace |
| `bootstrap.py` | Tải lại nguồn Docker công khai và tái tạo corpus Day 7 |
| `index.py` | Tạo schema pgvector và index corpus Day 7 |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## Dense retriever Day 6

`dense_retriever.py` dùng cùng embedding model và bảng `ingestion_chunks` của
Day 6. Cấu hình qua `DATABASE_URL`, tùy chọn `EMBEDDING_MODEL`, và strategy
`B` mặc định. Kết quả được chuẩn hóa về schema của `agentic.py`, gồm ID, nguồn,
trang, metadata và score:

```powershell
python -m pip install -r starter\requirements.txt
$env:DATABASE_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
```

Database phải được khởi động từ source Day 6 và index đã được tạo trước khi
gọi `Day6DenseRetriever.search`. Lỗi kết nối, embedding hoặc filter không được
thay bằng kết quả giả.

## Chạy độc lập không cần source Day 6

Day 7 giữ lại manifest nguồn và benchmark contract, nhưng không commit corpus
raw hoặc vector/generated output. Bootstrap tải lại Docker Official
Documentation từ các URL công khai, tạo lại corpus và 24 câu benchmark:

```powershell
docker compose -f starter\docker-compose.yml up -d
$env:DATABASE_URL = "postgresql://day7:day7_local_only@127.0.0.1:55434/day7_lab"

& ".venv\Scripts\python.exe" starter\bootstrap.py
& ".venv\Scripts\python.exe" starter\index.py `
  --database-url $env:DATABASE_URL
```

Sau đó có thể chạy `run_agent_cli.py` hoặc `run_agent_benchmark.py` như bình
thường. Lần chạy đầu cần Internet để tải tài liệu và model embedding; cache
nguồn nằm trong `starter/source/cache/` và không được commit. Source được dùng
theo giấy phép Apache 2.0 của Docker Docs, được ghi trong
`starter/source/manifest.json`.

## So sánh ba phương pháp retrieval

Chạy trên cùng corpus và dev benchmark sau khi đã tạo dữ liệu Phase 1:

```powershell
$env:DATABASE_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
python starter\benchmark_retrieval.py `
  --chunks starter\data\day7\chunks.json `
  --questions starter\data\day7\questions.json `
  --database-url $env:DATABASE_URL `
  --output starter\results\phase3-dev.json
```

Runner so sánh dense Day 6, lexical TF-IDF và hybrid RRF. Mỗi phương pháp
xuất top-3, evidence hit, latency và trace cho cả câu answerable lẫn
unanswerable; Hit@1 và Evidence Hit@3 chỉ tính trên câu answerable.

## Chạy agent thật

`agent_controller.py` dùng API chat-completions-compatible. Với OpenRouter,
cấu hình endpoint và model ID của OpenRouter. Không ghi API key vào source hoặc
`.env` được commit:

```powershell
$env:AGENT_API_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
$env:AGENT_API_KEY = "<local-secret>"
$env:AGENT_MODEL = "openai/gpt-4o-mini"
$env:AGENT_HTTP_REFERER = "https://localhost"
$env:AGENT_X_TITLE = "Day 7 Agentic RAG"
$env:DATABASE_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
# Dùng Python environment đã cài sentence-transformers và psycopg.
& "C:\path\to\day6\starter\.venv\Scripts\python.exe" starter\run_agent_cli.py `
  --question "Docker volume giữ dữ liệu như thế nào?" `
  --filter section=storage
```

Agent chỉ nhận context do retrieval tạo; context budget mặc định là 2400 ký tự
để phù hợp chunk Day 6. Citation phải xuất hiện trong context và trong câu trả
lời dưới dạng `[chunk-id]`. Khi không đủ evidence hoặc citation không hợp lệ,
`run_agent` trả về `abstain`.

Chạy agent trên dev bằng cấu hình hybrid:

```powershell
& "C:\path\to\day6\starter\.venv\Scripts\python.exe" `
  starter\run_agent_benchmark.py `
  --chunks starter\data\day7\chunks.json `
  --questions starter\data\day7\questions.json `
  --database-url $env:DATABASE_URL `
  --split dev `
  --retriever hybrid `
  --output starter\results\agent-dev-hybrid.json
```

Output chứa `cases` và `traces` cho 17 câu dev; lỗi provider được ghi riêng
với `status: "error"`, không biến thành câu trả lời giả.

## Phase 6: cải thiện trên dev

Trace `agent-dev-hybrid.json` hiện ghi nhận hai lỗi cần theo dõi:

- `q02`: false abstention; kết quả top-k không đưa được evidence đủ rõ cho
  model.
- `q14`: model tìm được evidence nhưng trả citation không theo đúng định dạng
  `[chunk-id]`, nên verifier từ chối câu trả lời.

Chạy hai cấu hình độc lập sau trên cùng 17 câu dev:

```powershell
# Cải thiện 1: tăng candidate top-k để tăng cơ hội đưa evidence vào context.
& "C:\path\to\day6\starter\.venv\Scripts\python.exe" `
  starter\run_agent_benchmark.py `
  --chunks starter\data\day7\chunks.json `
  --questions starter\data\day7\questions.json `
  --database-url $env:DATABASE_URL `
  --split dev --retriever hybrid --top-k 5 --budget 2400 `
  --output starter\results\agent-dev-hybrid-top5.json

# Cải thiện 2: giữ top-k nhưng tăng context budget và cho phép tìm thêm một lượt.
& "C:\path\to\day6\starter\.venv\Scripts\python.exe" `
  starter\run_agent_benchmark.py `
  --chunks starter\data\day7\chunks.json `
  --questions starter\data\day7\questions.json `
  --database-url $env:DATABASE_URL `
  --split dev --retriever hybrid --top-k 3 --max-searches 2 --budget 3200 `
  --output starter\results\agent-dev-hybrid-budget3200.json
```

So sánh mỗi output với baseline và `agent-dev-hybrid.json` theo số câu
answered/abstain, false abstention, citation hợp lệ, số lượt search, latency
và chi phí. Không chạy test để điều chỉnh cấu hình.

Kết quả cấu hình được freeze sau khi chạy lại ổn định:

- `top_k=3`, `budget=3200`, `max_searches=2`, retriever `hybrid`;
- 17/17 cases không có provider error;
- 15/15 câu answerable được trả lời;
- 2/2 câu unanswerable (`q21`, `q22`) được abstain;
- mỗi case chỉ cần 1 lượt search trong trace cuối.

Output: `starter/results/agent-dev-hybrid-budget3200-v2.json`.

## Phase 7: kiểm thử test split

Sau khi freeze cấu hình, test split được chạy độc lập với:
`hybrid`, `top_k=3`, `budget=3200`, `max_searches=2`.

- Agent: `4/5` câu answerable được trả lời;
- Agent: `2/2` câu unanswerable abstain đúng;
- Provider errors: `0`;
- `q18` là false abstention còn lại vì top-3 hybrid không đưa chunk chứa
  evidence vào context.
- TF-IDF baseline: Hit@1 `1/5`, Hit@3 `1/5`, context evidence `1/5`.

Output agent: `starter/results/agent-test-hybrid-budget3200.json`.
Output baseline: `starter/results/baseline-test/`.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev --budget 2400
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

Với source Day 6 hiện tại, dùng script tái lập trực tiếp từ corpus text và
benchmark gốc:

```powershell
python starter/prepare_data.py `
  --day6-root C:\path\to\K5-L3B-DAY06-VuNamKhanh-2A202604112-DataIngestion `
  --output starter/data/day7
```

Script tạo 427 chunk theo strategy B (paragraph, tối đa 1000 ký tự) và benchmark
24 câu: 15 câu có đáp án + 2 câu thiếu evidence cho dev, 5 câu có đáp án + 2
câu thiếu evidence cho test. Các file trong `starter/data/` bị Git ignore; chỉ
cần lưu script và lệnh tái lập, không commit corpus nếu không cần thiết.

Nếu đã có file export riêng, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget
2400 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác.
File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không
được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f
<đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
