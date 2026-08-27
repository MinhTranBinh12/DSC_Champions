import torch
from typing import List, Dict, Tuple, Any
from src.config import (
    RERANKER_MODEL_NAME, RERANK_TOP_K, RERANK_CHUNKS_PER_DOC, RERANK_MAX_LENGTH
)

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


class LegalReranker:
    """
    Cross-Encoder Re-ranker for deep query-document relevance scoring.
    """
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.model is None and HAS_CROSS_ENCODER:
            print(f"[Reranker] Loading Cross-Encoder model ({self.device}): {self.model_name}")
            try:
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=RERANK_MAX_LENGTH,
                    device=self.device
                )
            except Exception as e:
                print(f"[Reranker] Warning: Could not load Cross-Encoder '{self.model_name}': {e}")
                self.model = None

    def rerank(
        self,
        question: str,
        candidate_doc_ids: List[str],
        doc_chunks_map: Dict[str, List[Dict[str, Any]]],
        first_stage_scores: Dict[str, float] = None,
        top_k_rerank: int = RERANK_TOP_K,
        chunks_per_doc: int = RERANK_CHUNKS_PER_DOC
    ) -> List[Tuple[str, float]]:
        """
        Rerank a set of candidate documents for a given question.
        
        Args:
            question: The input legal query text.
            candidate_doc_ids: List of doc_ids retrieved in First Stage.
            doc_chunks_map: Mapping from doc_id to list of chunk dicts for that doc.
            first_stage_scores: Optional dictionary of combined BM25+Dense scores.
            top_k_rerank: Maximum number of candidate documents to score with Cross-Encoder.
            chunks_per_doc: Number of top chunks to score per document.
            
        Returns:
            List of (doc_id, rerank_score) sorted by rerank_score descending.
        """
        if not candidate_doc_ids:
            return []

        self.load_model()

        # If CrossEncoder model is available, compute deep Cross-Encoder scores
        if self.model is not None:
            # Only rerank Top-K candidates from first stage to optimize runtime
            pruned_candidates = candidate_doc_ids[:top_k_rerank]
            remaining_candidates = candidate_doc_ids[top_k_rerank:]

            pairs = []
            pair_doc_ids = []

            for doc_id in pruned_candidates:
                chunks = doc_chunks_map.get(doc_id, [])
                if not chunks:
                    continue
                
                # Pair question with top chunks of candidate document
                for chunk in chunks[:chunks_per_doc]:
                    pairs.append((question, chunk["text"]))
                    pair_doc_ids.append(doc_id)

            if pairs:
                batch_size = 64 if self.device == "cuda" else 32
                with torch.inference_mode():
                    scores = self.model.predict(pairs, batch_size=batch_size)
                
                doc_rerank_scores: Dict[str, float] = {}
                for doc_id, score in zip(pair_doc_ids, scores):
                    if doc_id not in doc_rerank_scores or score > doc_rerank_scores[doc_id]:
                        doc_rerank_scores[doc_id] = float(score)

                # Append remaining candidates with their normalized first stage score if needed
                if remaining_candidates and first_stage_scores:
                    min_rerank_score = min(doc_rerank_scores.values()) if doc_rerank_scores else 0.0
                    for doc_id in remaining_candidates:
                        # Ensure remaining have lower score than reranked candidates
                        doc_rerank_scores[doc_id] = min_rerank_score - 10.0 + first_stage_scores.get(doc_id, 0.0)

                sorted_docs = sorted(doc_rerank_scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_docs

        # Fallback if CrossEncoder is not loaded: return original candidate scores
        if first_stage_scores is not None:
            candidates_with_scores = [
                (doc_id, first_stage_scores.get(doc_id, 0.0)) for doc_id in candidate_doc_ids
            ]
            return sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)

        return [(doc_id, 1.0 / (idx + 1)) for idx, doc_id in enumerate(candidate_doc_ids)]
