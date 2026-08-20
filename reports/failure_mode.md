# Failure Mode & Fixs

## Bug: `SearchClient` trả thiếu kết quả khi chỉ có 1 tài liệu khớp từ khóa

**Triệu chứng.** `SearchClient.search(query, max_results=2)` được kỳ vọng luôn trả về tối đa
`max_results` tài liệu (miễn corpus có đủ). Nhưng với một số query, hàm chỉ trả về 1 tài liệu
dù corpus có 9 tài liệu khả dụng.

**Nguyên nhân.** Logic gốc:

```python
results = [doc for _, doc in scored[:max_results]]
if not results:
    results = list(documents[:max_results])
```

`scored` là danh sách tài liệu có ít nhất 1 từ khóa trùng với query. Đoạn fallback
(`if not results`) chỉ kích hoạt khi **hoàn toàn không** có tài liệu nào trùng từ khóa — nếu có
đúng 1 tài liệu trùng, `results` có 1 phần tử (không rỗng) nên fallback không chạy, và hàm trả
về 1 kết quả thay vì lấp đầy `max_results` bằng các tài liệu khác. Hệ quả: `ResearcherAgent`
nhận ít nguồn hơn dự kiến, làm giảm chất lượng `research_notes`/citation coverage một cách âm
thầm — không có lỗi/exception nào báo hiệu.

**Cách phát hiện.** Trước khi giao code, đã dựng lại toàn bộ project trong một sandbox riêng,
cài đúng dependency thật (`langgraph==1.2.11`, v.v.) và chạy `pytest` cho toàn bộ test suite.
Test `test_search_falls_back_to_some_documents_for_an_unrelated_query` (kỳ vọng 2 kết quả với
`max_results=2`) fail với `assert 1 == 2` — lộ đúng bug này dù logic "trông có vẻ đúng" khi đọc
qua.

**Cách fix.** Đổi sang chọn tài liệu tốt nhất theo điểm số trước, sau đó **luôn lấp đầy** đến
`max_results` bằng các tài liệu còn lại trong corpus (nếu có), thay vì chỉ fallback khi danh
sách rỗng:

```python
results: list[_Document] = []
seen_ids: set[int] = set()
for _, doc in scored:              # ưu tiên tài liệu điểm cao trước
    if len(results) >= max_results:
        break
    results.append(doc)
    seen_ids.add(id(doc))
if len(results) < max_results:      # lấp đầy phần còn thiếu, bất kể scored có bao nhiêu
    for doc in documents:
        if len(results) >= max_results:
            break
        if id(doc) in seen_ids:
            continue
        results.append(doc)
        seen_ids.add(id(doc))
```

**Kiểm chứng.** Chạy lại toàn bộ test suite (19 test) — pass 100%, kèm `ruff check`,
`ruff format --check`, `mypy src` đều sạch. File liên quan:
`src/multi_agent_research_lab/services/search_client.py`,
test: `tests/test_search_client.py`.

## Failure mode khác: stall trong routing (đã có guardrail, không phải bug)

Không phải bug nhưng là failure mode thực sự có thể xảy ra: nếu một worker agent chạy xong
nhưng không cập nhật đúng field trong `state` (ví dụ lỗi logic trong prompt khiến
`research_notes` vẫn `None`), `SupervisorAgent._decide_route` sẽ chọn lại đúng route đó ở lượt
kế tiếp → lặp vô hạn nếu không có guard. Được xử lý chủ động bằng **stall guard**: Supervisor
cho phép 1 lần retry cùng route, nếu route đó chạy 2 lần liên tiếp mà state vẫn không tiến
triển thì fallback sang `done` và ghi lý do vào `state.errors` (xem
`tests/test_agents_todo.py::test_allows_one_retry_then_falls_back_to_done_on_stall`).
