import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
CORPUS_DIR = DATA_DIR / "selected-contexts" / "selected-contexts"
TRAIN_PATH = DATA_DIR / "train.json"
PUBLIC_TEST_PATH = DATA_DIR / "public-official.json"

# Output & Cache Paths
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

PROCESSED_CORPUS_PATH = CACHE_DIR / "processed_corpus.json"
BM25_INDEX_PATH = CACHE_DIR / "bm25_index.pkl"
DENSE_INDEX_PATH = CACHE_DIR / "dense_embeddings.npy"
SUBMISSION_PATH = BASE_DIR / "submission.json"
SUBMISSION_ZIP_PATH = BASE_DIR / "submission.zip"

# Preprocessing & Chunking Configuration
MAX_CHUNK_TOKENS = 350
CHUNK_OVERLAP = 50
USE_WORD_SEGMENTATION = True

# First-Stage Retrieval (BM25 + Dense)
BM25_K1 = 1.5
BM25_B = 0.75
FIRST_STAGE_TOP_K = 30

# Bi-Encoder (Dense Retrieval) Configuration
DENSE_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
# Alternative choices: "BAAI/bge-m3", "intfloat/multilingual-e5-base"

# Cross-Encoder (Re-ranking) Configuration
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 15                 # Only rerank Top 15 candidates for 10x-20x speedup
RERANK_CHUNKS_PER_DOC = 1         # Best chunk per candidate doc
RERANK_MAX_LENGTH = 256           # Optimized sequence length for fast inference

# Post-processing & Dynamic Thresholding
DYNAMIC_THRESHOLD_RATIO = 0.88  # Include item if score >= top1_score * ratio
MIN_ABSOLUTE_SCORE = 0.1
MAX_PREDICTED_DOCS = 3          # Maximum documents to output per question

# Validation Split Ratio
VAL_RATIO = 0.2
RANDOM_SEED = 42
