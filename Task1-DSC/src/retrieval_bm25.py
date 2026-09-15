import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

from src.config import BM25_INDEX_PATH, BM25_K1, BM25_B, FIRST_STAGE_TOP_K
from src.utils import word_segment

try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False


class BM25Retriever:
    """
    BM25 Lexical Retriever with document-level score aggregation.
    """
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus_chunks = []
        self.doc_ids = []

    def build_index(self, corpus_chunks: List[Dict[str, Any]], force_rebuild: bool = False):
        """
        Build or load BM25 index from corpus chunks.
        """
        self.corpus_chunks = corpus_chunks
        self.doc_ids = [c["doc_id"] for c in corpus_chunks]

        if not force_rebuild and os.path.exists(BM25_INDEX_PATH):
            try:
                with open(BM25_INDEX_PATH, 'rb') as f:
                    saved_data = pickle.load(f)
                    bm25_obj = saved_data.get("bm25")
                    if bm25_obj is not None and (len(corpus_chunks) == 0 or getattr(bm25_obj, "corpus_size", len(corpus_chunks)) == len(corpus_chunks)):
                        print(f"[BM25] Loading existing BM25 index from: {BM25_INDEX_PATH}")
                        self.bm25 = bm25_obj
                        return
                    else:
                        print(f"[BM25] Cached BM25 index empty or mismatch. Rebuilding...")
            except Exception as e:
                print(f"[BM25] Error reading cached BM25 index ({e}). Rebuilding...")

        if not corpus_chunks:
            print("[BM25] Warning: corpus_chunks is empty. Skipping BM25 indexing.")
            return

        print(f"[BM25] Tokenizing corpus for BM25 indexing...")
        tokenized_corpus = [
            c["segmented_text"].lower().split() for c in corpus_chunks
        ]

        print(f"[BM25] Building BM25Okapi index over {len(tokenized_corpus)} chunks...")
        if HAS_RANK_BM25:
            self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        else:
            # Custom simple BM25 fallback implementation
            self.bm25 = SimpleBM25(tokenized_corpus, k1=self.k1, b=self.b)

        print(f"[BM25] Saving BM25 index to: {BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump({"bm25": self.bm25}, f)

    def search(self, question: str, top_k: int = FIRST_STAGE_TOP_K) -> List[Tuple[str, float]]:
        """
        Retrieve Top-K document IDs for a question using BM25.
        Aggregates chunk scores to document scores via max pooling.
        
        Returns:
            List of (doc_id, score) pairs sorted by score descending.
        """
        if self.bm25 is None or len(self.doc_ids) == 0:
            return []

        segmented_query = word_segment(question).lower().split()
        scores = self.bm25.get_scores(segmented_query)

        # Aggregate scores per doc_id (max chunk score per doc)
        doc_scores: Dict[str, float] = {}
        for chunk_idx, score in enumerate(scores):
            if chunk_idx < len(self.doc_ids):
                doc_id = self.doc_ids[chunk_idx]
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = float(score)

        # Sort documents by score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


class SimpleBM25:
    """Fallback BM25 implementation if rank_bm25 package is not installed."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 1.0
        
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        df = {}
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            
            for word in set(doc):
                df[word] = df.get(word, 0) + 1

        for word, freq in df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size)
        for q in query:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for idx, doc_freq in enumerate(self.doc_freqs):
                if q in doc_freq:
                    freq = doc_freq[q]
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / self.avgdl))
                    scores[idx] += q_idf * (numerator / denominator)
        return scores
