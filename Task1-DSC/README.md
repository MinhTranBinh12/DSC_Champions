# Hệ Thống Truy Hồi Văn Bản Pháp Luật Tiếng Việt (Vietnamese Legal Information Retrieval)

> **Dự án dự thi Data Science Competition (DSC)**  
> **Giải pháp**: Hybrid Two-Stage Retrieval kết hợp Deep Cross-Encoder Re-ranking và Dynamic Thresholding.

---

## 📊 1. Thống Kê Các Mô Hình Pre-trained Sử Dụng & Tính Hợp Lệ

Quy chế cuộc thi quy định tổng số lượng tham số từ các mô hình học sâu Pre-trained **không vượt quá 4 tỷ tham số (<= 4B parameters)**. 

Dưới đây là bảng thống kê chi tiết các mô hình được sử dụng trong pipeline:

| Thành phần | Tên mô hình / Thuật toán | Kiến trúc cơ sở | Số lượng tham số (Parameters) | % So với giới hạn 4B |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Vector Retrieval (Stage 1)** | `bkai-foundation-models/vietnamese-bi-encoder` | PhoBERT Base (Bi-Encoder) | **~135 Triệu** (~0.135B) | ~3.38% |
| **Deep Re-ranking (Stage 2)** | `BAAI/bge-reranker-v2-m3` | XLM-RoBERTa (Cross-Encoder) | **~568 Triệu** (~0.568B) | ~14.20% |
| **Lexical Retrieval (Stage 1)** | `BM25Okapi` | Thuật toán thống kê tần suất từ khóa | **0** (Không dùng model) | 0.00% |
| **Tách từ tiếng Việt** | `PyVi` / `Underthesea` | Quy tắc từ điển & thống kê CRF | **< 1 Triệu** | ~0.00% |
| **TỔNG CỘNG** | — | — | **~703 Triệu (~0.703B)** | **~17.58% (Hợp lệ)** |

> **Kết luận**: Tổng số tham số sử dụng là **~0.703B**, nằm hoàn toàn trong ngưỡng an toàn cho phép của Ban tổ chức.

---

## 🏗️ 2. Chi Tiết Kiến Trúc Pipeline

Hệ thống được xây dựng theo kiến trúc **2 giai đoạn (Two-Stage Retrieval Pipeline)** kết hợp xử lý ngưỡng động:

```
[Kho Văn Bản Pháp Luật (.json)]
         │
         ▼
[Tiền Xử Lý: Chuẩn hóa Unicode NFC + Tách từ PyVi + Sliding Window Chunking]
         │
         ├───────────────────────────────────────────┐
         ▼                                           ▼
[Lexical Index (BM25Okapi)]               [Dense Index (Vietnamese Bi-Encoder)]
         │                                           │
         └─────────────────────┬─────────────────────┘
                               │
            [Nhận câu hỏi truy vấn (Query)]
                               │
                               ▼
        [First-Stage Hybrid Retrieval: BM25 + Dense Search]
                               │
                               ▼
            [Score Normalization & Fusion (Top 50)]
                               │
                               ▼
     [Second-Stage Deep Re-ranking: BGE Reranker V2 M3 (Cross-Encoder)]
                               │
                               ▼
     [Hậu xử lý: Dynamic Thresholding (Ngưỡng 88%) & Ràng buộc (1 <= K <= 5)]
                               │
                               ▼
                   [Danh sách doc_ids dự đoán]
```

### Các bước hoạt động chi tiết:

### Bước 1: Tiền xử lý & Cắt đoạn văn bản (`src/preprocess.py`)
* **Chuẩn hóa tiếng Việt**: Chuyển đổi toàn bộ văn bản về dạng Unicode dựng sẵn (**NFC**), loại bỏ các ký tự điều khiển ẩn và khoảng trắng thừa.
* **Tách từ tiếng Việt (Word Segmentation)**: Áp dụng `PyVi` để gom các cụm từ ghép (ví dụ: `nghị_định`, `bồi_thường_thiệt_hại`, `hợp_đồng_lao_động`), tối ưu hóa chất lượng khớp từ cho BM25.
* **Sliding Window Chunking**: Cắt văn bản pháp luật dài thành các đoạn trượt:
  - `MAX_CHUNK_TOKENS = 350` từ.
  - `CHUNK_OVERLAP = 50` từ gối đầu nhằm tránh mất ngữ cảnh tại biên cắt.
  - Tự động gắn tên văn bản/tiêu đề vào đầu mỗi chunk (`f"{doc_name}. {chunk_text}"`).

---

### Bước 2: Truy hồi sơ bộ đa phương thức (First-Stage Hybrid Retrieval)
Lọc nhanh toàn bộ kho dữ liệu để trích xuất **Top 50 văn bản tiềm năng nhất**:
1. **Lexical Search (BM25Okapi)** (`src/retrieval_bm25.py`):
   - Bắt chính xác các từ khóa hiếm, số hiệu văn bản, số điều luật, tên cơ quan ban hành.
   - Điểm số tài liệu được tổng hợp theo cơ chế **Max-Pooling** qua các chunks thuộc tài liệu đó:
     $$\text{Score}_{\text{BM25}}(D) = \max_{c \in \text{chunks}(D)} \text{BM25}(Q, c)$$
2. **Dense Vector Search (Bi-Encoder)** (`src/retrieval_dense.py`):
   - Sử dụng `vietnamese-bi-encoder` mã hóa câu hỏi và các chunks thành vector 768 chiều.
   - Tính độ tương đồng ngữ nghĩa Cosine thông qua tích vô hướng ma trận.
3. **Score Fusion**:
   - Chuẩn hóa Min-Max điểm số của 2 nhánh về thang `[0, 1]` và kết hợp với trọng số cân bằng:
     $$\text{Score}_{\text{Hybrid}}(D) = 0.5 \times \text{Norm\_BM25}(D) + 0.5 \times \text{Norm\_Dense}(D)$$

---

### Bước 3: Tái xếp hạng sâu (Second-Stage Deep Re-ranking - `src/reranker.py`)
* Sử dụng mô hình Cross-Encoder **`BAAI/bge-reranker-v2-m3`** chấm điểm trực tiếp cặp `(Câu hỏi, Đoạn văn)`.
* Nhờ cơ chế Self-Attention đa tầng giữa toàn bộ câu hỏi và văn bản, mô hình phân biệt chính xác mối quan hệ pháp lý phức tạp và loại bỏ các tài liệu nhiễu/bẫy từ vựng từ bước 1.

---

### Bước 4: Hậu xử lý & Ngưỡng động (Dynamic Thresholding - `src/evaluate.py`, `src/predict.py`)
* **Ngưỡng động (Dynamic Thresholding)**:
  - Tài liệu Top 1 luôn được giữ lại.
  - Các tài liệu đứng sau (Top 2, 3,...) chỉ được giữ lại nếu điểm số đạt từ **88%** điểm của Top 1:
    $$\text{Score}(D_k) \ge \text{Score}(D_1) \times 0.88$$
* **Ràng buộc cuộc thi**: Giới hạn tối đa **không vượt quá 5 tài liệu/câu hỏi** ($1 \le \text{len}(\text{answer}) \le 5$) để đảm bảo không bị trừ điểm vi phạm quy chế.

---

## 📁 3. Cấu Trúc Thư Mục Dự Án

```
DSC/
├── cache/                     # Chứa các file index và vector embedding đã tính toán
│   ├── processed_corpus.json  # Toàn bộ chunks văn bản sau tiền xử lý
│   ├── bm25_index.pkl         # Chỉ mục BM25
│   └── dense_embeddings.npy   # Ma trận vector embedding của kho ngữ liệu
├── src/
│   ├── config.py              # Cấu hình đường dẫn, siêu tham số (hyperparameters)
│   ├── utils.py               # Hàm bổ trợ: load/save JSON, chuẩn hóa Unicode, metric Recall/Precision
│   ├── preprocess.py          # Tiền xử lý, tách từ và phân đoạn văn bản
│   ├── retrieval_bm25.py      # Module truy hồi từ khóa BM25Okapi
│   ├── retrieval_dense.py     # Module truy hồi ngữ nghĩa Bi-Encoder
│   ├── reranker.py            # Module tái xếp hạng Cross-Encoder
│   ├── evaluate.py            # Pipeline đánh giá tập Validation (Mean Recall / Precision)
│   └── predict.py             # Pipeline sinh file submission.json và submission.zip
├── main.py                    # Entrypoint chính để chạy toàn bộ hoặc từng bước pipeline
├── requirements.txt           # Danh sách các thư viện phụ thuộc
├── .gitignore                 # Bỏ qua các file cache và dữ liệu nặng khi push git
└── README.md                  # Tài liệu hướng dẫn & thuyết minh kiến trúc
```

---

## 🚀 4. Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### Cài đặt môi trường
Khuyến nghị sử dụng Python >= 3.9 và môi trường ảo:
```bash
pip install -r requirements.txt
```

### Chạy Pipeline

1. **Chạy toàn bộ pipeline (Tiền xử lý $\rightarrow$ Đánh giá $\rightarrow$ Xuất kết quả nộp bài)**:
   ```bash
   python main.py --all
   ```

2. **Chỉ chạy bước tiền xử lý ngữ liệu**:
   ```bash
   python main.py --preprocess
   ```

3. **Chỉ chạy đánh giá trên tập Validation (5600 Train / 1400 Val)**:
   ```bash
   python main.py --eval
   ```

4. **Chỉ sinh file dự đoán nộp bài trên `public-official.json`**:
   ```bash
   python main.py --predict
   ```

Kết quả nộp bài sẽ được tự động lưu và nén vào file `submission.zip` ở thư mục gốc, chứa duy nhất `submission.json` đúng theo định dạng quy định của cuộc thi.
