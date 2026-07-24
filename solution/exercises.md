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
> "Qua bốn phản hồi, em quan sát thấy quy luật: temperature càng cao thì mô hình càng sử dụng vốn từ phong phú, cấu trúc câu đa dạng và bay bổng hơn. Ở mức 0.0 - 0.7, văn phong rất tự nhiên, chuẩn xác và có tính nhất quán cao. Khi tăng lên 1.2, phản hồi xuất hiện nhiều ý tưởng độc đáo hơn nhưng câu từ bắt đầu hơi dông dài. Đến mức 1.8, phản hồi bắt đầu mất mạch lạc nghiêm trọng, xuất hiện nhiều từ ngữ không liên quan và sai lỗi ngữ pháp."

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> "Em sẽ đặt temperature = 0.0 cho trợ lý soạn thảo hợp đồng pháp lý nhằm tối ưu hóa tính chính xác, tính nhất quán và loại bỏ hoàn toàn rủi ro mô hình tự bịa đặt thông tin. Đối với trợ lý viết slogan quảng cáo, em sẽ chọn temperature = 0.9 đến 1.0 để kích thích tính sáng tạo, giúp mô hình sinh ra những ý tưởng mới lạ, gợi hình và thu hút người đọc."

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> "Với 20.000 người dùng mỗi ngày (mỗi người 2 lượt * 500 token output), tổng dung lượng là 20 triệu output token/ngày. Chi phí tương ứng với GPT-4o ($0.010/1k token) là $200 USD/ngày, trong khi GPT-4o-mini ($0.0006/1k token) chỉ tốn $12 USD/ngày. GPT-4o xứng đáng chi phí đối với các tác vụ phức tạp đòi hỏi tư duy sâu như phân tích pháp lý, tài chính hoặc sinh mã nguồn. Trong khi đó, GPT-4o-mini là lựa chọn tối ưu cho các tác vụ lặp đi lặp lại như chatbot hỗ trợ khách hàng hoặc phân loại cảm xúc."

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
> "Phản hồi của nhà thơ mang tính gợi hình cao, sử dụng các hình ảnh ẩn dụ sinh động (như việc học đi của đứa trẻ) mà không dùng thuật ngữ chuyên ngành. Ngược lại, phản hồi của kỹ sư senior đi thẳng vào bản chất kỹ thuật với định nghĩa mạch lạc, phân loại rõ ràng và kèm theo đoạn mã Python minh họa. Qua đó, em nhận thấy system prompt có thể điều khiển trực tiếp tông giọng (tone), độ sâu chuyên môn, định dạng đầu ra và đối tượng độc giả mục tiêu."

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> "Trong tiếng Việt, số token đếm bằng tiktoken thường cao hơn số từ từ 40% đến 80% do thuật toán mã hóa BPE phân tách các từ có dấu thanh thành nhiều subword token. Nếu áp dụng công thức ước lượng thô (số từ / 0.75) vốn dành cho tiếng Anh, chúng em sẽ dự toán THIẾU ngân sách nghiêm trọng, vì số lượng token thực tế bị tính phí bởi API cao hơn đáng kể so với số từ hiển thị."

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> "Chatbot văn bản và trợ lý giọng nói là hai ứng dụng hưởng lợi lớn nhất từ streaming vì giúp tối ưu thời gian phản hồi đầu tiên (Time To First Token - TTFT), mang lại cảm giác tương tác tức thì cho người dùng. Riêng trợ lý giọng nói có thể bắt đầu chuyển văn bản thành giọng đọc ngay khi nhận được những token đầu tiên. Trái lại, pipeline dịch tài liệu chạy ngầm ban đêm hoàn toàn không cần streaming vì đây là quy trình xử lý theo lô (batch processing), không đòi hỏi người dùng phải theo dõi tiến độ từng từ."

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> "Exponential backoff làm tăng thời gian chờ theo cấp số nhân sau mỗi lần thất bại, giúp giảm áp lực dồn dập lên máy chủ và tạo khoảng nghỉ đủ lớn để hệ thống tự phục hồi. Tuy nhiên, nếu hàng nghìn client cùng bị lỗi tại một thời điểm, chúng vẫn có thể gửi lại request trùng nhau ở các mốc thời gian cố định. Kỹ thuật jitter (thêm nhiễu thời gian ngẫu nhiên) sẽ phân rải các lượt retry này ra, giải quyết dứt điểm sự cố nghẽn mạng đồng loạt (Thundering Herd Problem)."

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> "System prompt em thiết lập: 'Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn và dùng tiếng Việt.'\n1. Nếu bỏ cụm 'ngắn gọn': Trợ lý sẽ giải thích quá dông dài, cung cấp nhiều thông tin phụ không cần thiết, làm tốn token và giảm tốc độ phản hồi.\n2. Nếu bỏ cụm 'dùng tiếng Việt': Trợ lý có thể phản hồi bằng tiếng Anh nếu câu hỏi của người dùng chứa từ khóa hoặc thuật ngữ chuyên ngành."

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> "Tình huống: Người dùng nêu yêu cầu tổng quan ở lượt 1, sau đó trao đổi sâu qua 4 lượt hội thoại chi tiết. Đến lượt 5, khi người dùng nhờ 'Tóm tắt lại bài toán ban đầu', trợ lý sẽ không thể trả lời do lượt 1 đã bị cắt khỏi bộ nhớ history (tối đa 4 lượt).\nCách khắc phục: Áp dụng kỹ thuật Tóm tắt hội thoại (Conversation Summarization) bằng cách dùng mô hình nhỏ tự động tóm tắt các lượt thoại cũ và đính kèm vào system prompt, giúp duy trì ngữ cảnh xuyên suốt phiên chat mà vẫn tối ưu chi phí."

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
