HE THONG TRUY HOI VA TAO CAU TRA LOI PHAP LUAT TIENG VIET (LEGAL QA - RAG PIPELINE)

Tac vu: Hoi dap phap luat tieng Viet dua tren can cu van ban quy pham phap luat.
Muc tieu: Nhap cau hoi phap luat -> Truy xuat van ban luat phu hop -> Sinh cau tra loi tu nhien bang van xuoi dua tren can cu phap ly.
Do do danh gia: METEOR (do do chinh) va ROUGE-L (do do phu).
Gioi han tham so: Tong so tham so cua cac mo hinh Pre-trained khong vuot qua 4.0 ty tham so (<= 4B parameters).


================================================================================
1. BANG CAN DOI NGAN SACH THAM SO (TONG CONG <= 4.0B PARAMETERS)
================================================================================

He thong su dung kien truc Two-Stage Retrieval ket hop LLM Generation:

1. Stage 1 - Dense Vector Retrieval:
   - Mo hinh: BAAI/bge-m3
   - Kien truc: Multi-Lingual Bi-Encoder
   - So tham so: ~568 trieu (~0.57B, chiem ~14.2% nguong 4B)
   - Vai tro: Ma hoa ngu nghia toan dien cau hoi va doan van luat.

2. Stage 1 - Lexical Search:
   - Thuat toan: BM25Okapi ket hop PyVi tach tu
   - So tham so: 0 (Khong su dung mang nơ-ron)
   - Vai tro: Bat chinh xac tu khoa, so hieu van ban, dieu khoan luat.

3. Stage 2 - Deep Cross-Encoder Reranker:
   - Mo hinh: BAAI/bge-reranker-v2-m3
   - Kien truc: Multi-Lingual Cross-Encoder (XLM-RoBERTa)
   - So tham so: ~568 trieu (~0.57B, chiem ~14.2% nguong 4B)
   - Vai tro: Cham diem tuong tac toan phan giua cau hoi va cac doan luat tiem nang.

4. Stage 3 - Legal Answer Generator:
   - Mo hinh: Qwen/Qwen2.5-3B-Instruct
   - Kien truc: Decoder-only Large Language Model
   - So tham so: ~3.09 ty (~3.09B, chiem ~77.2% nguong 4B)
   - Vai tro: Tong hop thong tin va sinh cau tra loi van xuoi theo phong cach phap ly.

TONG CONG TOAN BO PIPELINE: ~3.66 Ty tham so (~3.66B parameters).
Ket luan: Pipeline hoan toan hop le theo quy che (3.66B <= 4.0B).

Phuong an thay the sieu nhe (Fast Setup):
- Thay the Generator bang Qwen/Qwen2.5-1.5B-Instruct (~1.54B).
- Tong tham so giam con ~2.11B, giup tiet kiem VRAM va tang toc do suy luan gap 2 lan tren 1x GPU T4.


================================================================================
2. KIEN TRUC PIPELINE VA QUY TRINH HOAT DONG
================================================================================

Quy trinh xu ly qua 4 buoc chinh:

Kho van ban luat (selected-contexts)
   |
   v
[Buoc 1: Kiem tra format, Lam sach du lieu & Phan doan Context-Aware Chunking]
   1. Kiem tra format & Lam sach du lieu kho ngu lieu (Data Audit & Sanitation):
      - Kiem tra tinh toan ven cau truc JSON cua cac tep context_*.json va tap cau hoi.
      - Phat hien va loai bo hoan toan cac van ban co truong "passage" rong (empty docs).
      - Bam MD5 noi dung de gom nhom cac van ban trung lap 100%, thiet lap bang anh xa Canonical ID dai dien nham loai bo du thua.
      - Chuan hoa Unicode toan dien sang dang dung san (NFC), loai bo cac ky tu dieu khien an (\t, \r, \f, \v) va chuan hoa khoang trang/dong trong thua.
   2. Phan doan van ban phan cap (Context-Aware Chunking):
      - Thay vi cat van ban mot cach tho bao theo so tu (de lam dut gay ngu canh), he thong ap dung thuat toan Context-Aware Chunking.
      - He thong tu dong nhan dien cac "Dieu" trong van ban luat phap.
      - Neu mot Dieu qua dai, no se bi cat nho theo dau cham cau, va Tieu de cua Dieu luat se duoc tu dong dan vao dau moi doan nho (Breadcrumb Context Header).
      - Nho vay, doan van ban du nam o dau cung khong bao gio bi mat boi canh goc.
      - Toan bo du lieu sau do duoc di qua thu vien PyVi de gop tu (Word Segmentation) giup cac thuat toan tim kiem hieu Tieng Viet tot hon (vi du: phap_luat thay vi phap va luat).
   |
   +---> Index BM25Okapi (Ban PyVi gop tu)
   +---> Index Vector Embedding BGE-M3 (Ban tho NFC nguyen ban)
   |
[Nhan cau hoi truy van (Query)]
   |
   v
[Buoc 2: Truy hoi so bo da phuong thuc (First-Stage Hybrid Retrieval)]
   - BM25Okapi tim kiem theo tu khoa va so hieu luat.
   - BGE-M3 tim kiem theo do tuong dong vector ngu nghia.
   - Hop nhat thu hang bang Reciprocal Rank Fusion (RRF).
   - Trich xuat Top 25-30 chunks co diem cao nhat.
   |
   v
[Buoc 3: Tai xep hang chuyen sau (Second-Stage Deep Reranking)]
   - BGE-Reranker-v2-m3 cham diem truc tiep cap (Cau hoi, Doan luat).
   - Loc bo cac bay dieu kien loai tru, pham vi ap dung va ngoai le.
   - Chon loc ra Top 1 hoac 2 Dieu luat chinh xac nhat lam can cu.
   |
   v
[Buoc 4: Sinh cau tra loi van xuoi chuan phap ly (Answer Generation)]
   - LLM (Qwen2.5-3B-Instruct) nhan cau hoi va noi dung Dieu luat da chon.
   - Sinh cau tra loi theo dung cau truc 3 phan chuan muc:
     1. Can cu phap ly (Dieu..., Khoan..., Van ban luat...)
     2. Trich dan nguyen van noi dung quy dinh cua dieu luat
     3. Ket luan truc tiep vao cau hoi ("Theo do,... / Nhu vay,...")
   - Toi uu hoa truc tiep cac tieu chi cua do do METEOR va ROUGE-L.
   |
   v
[Xuat ket qua vao submission.json]
   - Dien cau tra loi van xuoi vao truong "answer" cho tung "id".


================================================================================
3. BON TRU COT CAI TIEN KY THUAT COT LOI
================================================================================

1. Kiem tra format, Lam sach du lieu & Cat van ban phan cap (Context-Aware Chunking):
   - Kiem tra format & Loc du lieu sach (Data Cleansing & Audit):
     * Kiem tra dinh dang du lieu dau vao, loai bo cac mau bi thieu truong hoac bi loi cau truc JSON.
     * Quet kho ngu lieu context, loai bo tat ca cac van ban khong co noi dung passage hoac passage qua ngan/vo nghia.
     * Su dung thuat toan bam MD5 de phat hien va hop nhat cac van ban bi trung lap noi dung 100%, tao bang anh xa Canonical ID giup giam tai tinh toan va tranh phan tan diem so.
     * Chuan hoa Unicode toan bo van ban ve chuan NFC, lam sach cac ky tu dac biet, ngat dong thua va ky tu dieu khien an.
   - Thuat toan Context-Aware Chunking:
     * Thay vi cat van ban mot cach tho bao theo so tu (de lam dut gay ngu canh), he thong dung thuat toan Context-Aware Chunking.
     * He thong tu dong nhan dien cac "Dieu" trong van ban luat phap. Neu mot Dieu qua dai, no se bi cat nho theo dau cham cau, va Tieu de cua Dieu luat se duoc tu dong dan vao dau moi doan nho.
     * Nho vay, doan van ban du nam o dau cung khong bao gio bi mat boi canh goc.
     * Sau do, toan bo du lieu se duoc di qua thu vien PyVi de gop tu (Word Segmentation) giup cac thuat toan tim kiem hieu Tieng Viet tot hon (vi du: phap_luat thay vi phap va luat).
     * Tao song song 2 phien ban du lieu: Ban tho (Raw NFC) cho Dense Retriever/Reranker/LLM de giu van phong tu nhien, va Ban tach tu (PyVi Word-segmented) danh rieng cho BM25Okapi de toi uu do khop tu ghep.

2. Tim kiem Giai doan 1 (First-Stage High-Recall Hybrid Retrieval):
   - Chien thuat ket hop giua Lexical va Semantic:
     * BM25Okapi bat chinh xac so hieu nghi dinh, thong tu, ten luat va cac thuat ngu phap ly hiem.
     * BGE-M3 bat ngu nghia sau sac, xu ly tot cac cau hoi dien dat gieo neo hoac su dung tu dong nghia.
   - Su dung cong thuc Reciprocal Rank Fusion (RRF):
     RRF_Score(d) = 1 / (60 + Rank_BM25(d)) + 1 / (60 + Rank_Dense(d))
     de loai bo do lech phan phoi diem giua hai phuong phap, mo rong phieu ung vien len Top 25 de tranh bo sot van ban dung.

3. Tai xep hang Giai doan 2 (Second-Stage Deep Cross-Encoder Reranking):
   - Su dung mo hinh Cross-Encoder bge-reranker-v2-m3 de tinh toan chu y cheo (Cross-Attention) tren toan bo chuoi token cua cau hoi va doan luat.
   - Mo hinh giup phat hien cac dieu kien rang buoc phuc tap nhu "tru truong hop...", doi tuong loai tru, thoi hieu thi hanh ma cac mo hinh Bi-Encoder de bo sot.
   - Tu danh sach Top 25, reranker se giu lai 1 den 2 doan phap ly quan trong nhat de dua vao LLM.

4. Sinh cau tra loi van xuoi can chinh theo cau truc tham chieu (Style-Aligned Generation):
   - Phan tich du lieu train.json cho thay cac cau tra loi tham chieu deu tuan thu khuon mau:
     "Can cu theo Dieu... Khoan... [Ten Luat], quy dinh nhu sau:
      [Noi dung trich dan nguyen van]
      Theo do, / Nhu vay, [Giai thich van xuoi ngan gon dap ung yeu cau]."
   - Chien luoc toi uu METEOR & ROUGE-L:
     * METEOR danh gia do khop token chinh xac (chu trong Recall va thu tu token).
     * ROUGE-L danh gia chuoi con chung dai nhat (LCS).
     * Do do, cau tra loi phai bao ton toi da cac cau chu nguyen van trong van ban luat goc thay vi tu y dien giai tu do (paraphrase) lam lech tu ngu.
     * Dat nhiet do sinh thap (temperature = 0.0 den 0.1) va greedy decoding de cau tra loi luon nhat quan, chuan xac va khong bi bia dat (hallucination).


================================================================================
4. CAU TRUC DU LIEU TRONG THU MUC T2
================================================================================

Thu muc T2 bao gom cac tep du lieu:

1. train (1).json:
   - Tap du lieu huan luyen (~7,000 mau).
   - Cau truc moi phan tu:
     "id": {
         "question": "Cau hoi phap luat tieng Viet",
         "answer": "Cau tra loi mau bang van xuoi cua chuyen gia phap ly"
     }

2. public-official (1).json:
   - Tap du lieu kiem tra chinh thuc (1,000 mau).
   - Cau truc:
     "id": {
         "question": "Cau hoi can du doan",
         "answer": null
     }
   - Nhiem vu: Dien cau tra loi van xuoi vao truong "answer".

3. selected-contexts (1)/selected-contexts/:
   - Kho van ban quy pham phap luat go nguon tu Thuvienphapluat.
   - Moi tep context_*.json gom:
     "id": Ma so dinh danh cua van ban
     "name": Ten / so hieu van ban
     "link": Duong dan goc
     "passage": Toan bo noi dung van ban luat


================================================================================
5. SO SANH VOI PHUONG PHAP CU
================================================================================

- Muc tieu dau ra:
  * Phuong phap cu: Chi xuat danh sach ID van ban (vi du: ["1001", "1002"]).
  * Pipeline hien tai: Sinh cau tra loi hoan chinh bang van xuoi co can cu phap ly ro rang.

- Do do danh gia:
  * Phuong phap cu: Precision / Recall tren danh sach ID.
  * Pipeline hien tai: METEOR (chinh) va ROUGE-L (phu) tren chuoi van ban tra loi.

- Xu ly ngu canh:
  * Phuong phap cu: Cat doan co dinh 300-350 tu dan den dut gach ngu canh dieu luat.
  * Pipeline hien tai: Hierarchical Context-Aware Chunking gan Breadcrumb Header giu nguyen toa do phap ly.

- Tong tham so mo hinh:
  * Bi-Encoder (0.57B) + Reranker (0.57B) + Generator (3.09B) = ~3.66B parameters.
  * Hoan toan nam trong nguong <= 4.0B theo quy dinh cua Ban to chuc.
