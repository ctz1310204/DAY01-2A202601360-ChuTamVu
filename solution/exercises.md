# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> "Qua bốn phản hồi, em thấy quy luật là khi temperature càng cao thì câu trả lời càng bay bổng, sử dụng nhiều cấu trúc từ ngữ phong phú và sáng tạo hơn. Ở mức 0.0 - 0.7, câu trả lời rất ổn định, chuẩn xác và mạch lạc. Khi tăng lên 1.2, văn phong bắt đầu giàu hình ảnh hơn. Khi đạt mức 1.8, phản hồi bắt đầu mất mạch lạc, xuất hiện các từ ngữ ngẫu nhiên, lặp lại hoặc sai ngữ pháp."

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> "Em sẽ chọn temperature = 0.0 cho trợ lý soạn thảo hợp đồng pháp lý để đảm bảo tính chuẩn xác, logic, nhất quán tuyệt đối và tránh tối đa việc model tự ý sáng tạo hay bịa đặt. Ngược lại, em chọn temperature = 0.9 - 1.1 cho trợ lý viết slogan quảng cáo để khuyến khích mô hình đưa ra những ý tưởng độc đáo, bất ngờ và từ ngữ có tính thu hút cao."

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> "Tổng lượng output token/ngày = 20,000 * 2 * 500 = 20,000,000 token (20,000k token). Chi phí output hàng ngày với GPT-4o ($0.010/1k) là $200 USD/ngày, trong khi GPT-4o-mini ($0.0006/1k) chỉ mất $12 USD/ngày. GPT-4o xứng đáng chi phí cho các tác vụ tư vấn pháp lý, phân tích tài chính phức tạp hoặc sinh mã nguồn lớn. GPT-4o-mini là lựa chọn đúng cho chatbot hỗ trợ khách hàng trả lời câu hỏi thường gặp (FAQ) hoặc phân loại cảm xúc văn bản."

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> "Phản hồi của nhà thơ dùng hình ảnh ẩn dụ (máy học như đứa trẻ học đi), văn phong bay bổng, không chứa thuật ngữ kỹ thuật. Ngược lại, phản hồi của kỹ sư senior đi thẳng vào định nghĩa chính xác (dữ liệu, mô hình, huấn luyện), cấu trúc mạch lạc và đính kèm ví dụ code Python. Từ đó em rút ra system prompt có thể điều khiển hiệu quả tông giọng (tone of voice), độ sâu kiến thức, phong cách định dạng và phạm vi thuật ngữ của mô hình."

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> "Số token đếm bằng tiktoken cho tiếng Việt thường lớn hơn số từ từ 30% đến 80% (do bộ mã hóa BPE tách các từ có dấu thanh thành nhiều subword token), trong khi công thức số từ / 0.75 dựa trên giả định tiếng Anh (1 token ≈ 0.75 từ). Nếu dùng công thức ước lượng thô cho tiếng Việt, chúng em sẽ dự toán THIẾU ngân sách nghiêm trọng, vì lượng token thực tế bị thu phí bởi API cao hơn rất nhiều so với số từ đếm được."

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> "Chatbot văn bản và trợ lý giọng nói hưởng lợi nhiều nhất từ streaming vì giúp giảm đáng kể thời gian chờ đợi phản hồi đầu tiên (TTFT - Time To First Token), tạo cảm giác tương tác tự nhiên real-time; riêng trợ lý giọng nói có thể bắt đầu đọc ngay câu đầu tiên được stream về. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm hoàn toàn không cần streaming vì đây là tác vụ xử lý lô (batch job) chạy ẩn, người dùng không trực tiếp chờ xem kết quả từng từ."

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> "Exponential backoff làm tăng thời gian chờ gấp đôi sau mỗi lần thất bại, giúp giãn khoảng cách các đợt thử lại để máy chủ có thời gian phục hồi thay vì bị dồn nén liên tục như delay cố định. Kỹ thuật jitter (thêm nhiễu thời gian ngẫu nhiên) giải quyết triệt để sự cố Thundering Herd Problem - khi hàng ngàn client cùng retry ở đúng những mốc thời gian trùng nhau sau backoff làm server bị sập lại."

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> "System prompt em dùng: 'Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn và dùng tiếng Việt.'\n1. Nếu xóa cụm 'ngắn gọn': Trợ lý sẽ trả lời dông dài, giải thích quá chi tiết gây tốn token và mất thời gian của người học.\n2. Nếu xóa cụm 'dùng tiếng Việt': Trợ lý có thể trả lời bằng tiếng Anh nếu câu hỏi của người dùng có chứa các từ khóa tiếng Anh hoặc thuật ngữ chuyên ngành."

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> "Tình huống: Người dùng đưa ra yêu cầu bài toán ở lượt 1, sau đó trải qua 4 lượt hỏi đáp chi tiết. Đến lượt thứ 5, người dùng yêu cầu 'Hãy tổng hợp lại bài toán ở lượt đầu', do history đã bị cắt mất lượt 1 nên trợ lý không còn nhớ yêu cầu ban đầu.\nCách khắc phục: Sử dụng kỹ thuật Conversation Summarization — dùng một mô hình nhỏ để tóm tắt các lượt thoại cũ đã bị cắt bỏ và chèn bản tóm tắt này ngay dưới system prompt, giúp duy trì ngữ cảnh dài hạn với chi phí token thấp."

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
