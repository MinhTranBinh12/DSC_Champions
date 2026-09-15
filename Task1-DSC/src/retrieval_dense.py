import os
import numpy as np
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

from src.config import DENSE_MODEL_NAME, DENSE_INDEX_PATH, FIRST_STAGE_TOP_K

# Try importing sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class DenseRetriever:
    """
    Dense Vector Retriever (Bi-Encoder) with document-level score aggregation.
    """
    def __init__(self, model_name: str = DENSE_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.corpus_embeddings = None
        self.corpus_chunks = []
        self.doc_ids = []

    def load_model(self):
        if self.model is None and HAS_SENTENCE_TRANSFORMERS:
            print(f"[Dense] Loading Bi-Encoder model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[Dense] Warning: Could not load model '{self.model_name}': {e}")
                self.model = None

    def build_index(self, corpus_chunks: List[Dict[str, Any]], force_rebuild: bool = False):
        """
        Compute or load dense embeddings for corpus chunks.
        """
        self.corpus_chunks = corpus_chunks
        self.doc_ids = [c["doc_id"] for c in corpus_chunks]

        if not force_rebuild and os.path.exists(DENSE_INDEX_PATH):
            try:
                embeddings = np.load(DENSE_INDEX_PATH)
                if embeddings is not None and embeddings.ndim == 2 and (len(corpus_chunks) == 0 or len(embeddings) == len(corpus_chunks)):
                    print(f"[Dense] Loading existing dense embeddings shape {embeddings.shape} from: {DENSE_INDEX_PATH}")
                    self.corpus_embeddings = embeddings
                    self.load_model()
                    return
                else:
                    print(f"[Dense] Cached embeddings empty or shape mismatch. Recomputing...")
            except Exception as e:
                print(f"[Dense] Error loading cached embeddings ({e}). Recomputing...")

        self.load_model()
        if self.model is None:
            print("[Dense] SentenceTransformer model unavailable. Skipping dense indexing.")
            return

        if not corpus_chunks:
            print("[Dense] Warning: corpus_chunks is empty. Skipping dense indexing.")
            return

        texts = [c["text"] for c in corpus_chunks]
        print(f"[Dense] Encoding {len(texts)} chunks with {self.model_name}...")
        
        # Compute embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        self.corpus_embeddings = np.array(embeddings, dtype=np.float32)
        print(f"[Dense] Saving embeddings array shape {self.corpus_embeddings.shape} to: {DENSE_INDEX_PATH}")
        np.save(DENSE_INDEX_PATH, self.corpus_embeddings)

    def search(self, question: str, top_k: int = FIRST_STAGE_TOP_K) -> List[Tuple[str, float]]:
        """
        Retrieve Top-K document IDs for a question using Dense Bi-Encoder similarity.
        """
        if self.corpus_embeddings is None or len(self.corpus_embeddings) == 0 or self.corpus_embeddings.ndim != 2:
            return []

        if self.model is None:
            self.load_model()

        if self.model is None:
            return []

        # Encode query vector
        query_emb = self.model.encode([question], normalize_embeddings=True)[0]
        
        # Cosine similarity scores (since vectors are normalized)
        scores = np.dot(self.corpus_embeddings, query_emb)

        # Aggregate max chunk score per doc_id
        doc_scores: Dict[str, float] = {}
        for chunk_idx, score in enumerate(scores):
            if chunk_idx < len(self.doc_ids):
                doc_id = self.doc_ids[chunk_idx]
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = float(score)

        # Sort documents by score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
