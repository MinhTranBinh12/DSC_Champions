from typing import List, Dict, Tuple, Any
from src.config import RERANKER_MODEL_NAME

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

    def load_model(self):
        if self.model is None and HAS_CROSS_ENCODER:
            print(f"[Reranker] Loading Cross-Encoder model: {self.model_name}")
            try:
                self.model = CrossEncoder(self.model_name, max_length=512)
            except Exception as e:
                print(f"[Reranker] Warning: Could not load Cross-Encoder '{self.model_name}': {e}")
                self.model = None

    def rerank(
        self,
        question: str,
        candidate_doc_ids: List[str],
        doc_chunks_map: Dict[str, List[Dict[str, Any]]],
        first_stage_scores: Dict[str, float] = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank a set of candidate documents for a given question.
        
        Args:
            question: The input legal query text.
            candidate_doc_ids: List of doc_ids retrieved in First Stage.
            doc_chunks_map: Mapping from doc_id to list of chunk dicts for that doc.
            first_stage_scores: Optional dictionary of combined BM25+Dense scores.
            
        Returns:
            List of (doc_id, rerank_score) sorted by rerank_score descending.
        """
        if not candidate_doc_ids:
            return []

        self.load_model()

        # If CrossEncoder model is available, compute deep Cross-Encoder scores
        if self.model is not None:
            pairs = []
            pair_doc_ids = []

            for doc_id in candidate_doc_ids:
                chunks = doc_chunks_map.get(doc_id, [])
                if not chunks:
                    continue
                
                # Pair question with top chunks of candidate document
                for chunk in chunks[:3]:  # Score top 3 chunks per candidate doc
                    pairs.append((question, chunk["text"]))
                    pair_doc_ids.append(doc_id)

            if pairs:
                scores = self.model.predict(pairs, batch_size=32)
                
                doc_rerank_scores: Dict[str, float] = {}
                for doc_id, score in zip(pair_doc_ids, scores):
                    if doc_id not in doc_rerank_scores or score > doc_rerank_scores[doc_id]:
                        doc_rerank_scores[doc_id] = float(score)

                sorted_docs = sorted(doc_rerank_scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_docs

        # Fallback if CrossEncoder is not loaded: return original candidate scores
        if first_stage_scores is not None:
            candidates_with_scores = [
                (doc_id, first_stage_scores.get(doc_id, 0.0)) for doc_id in candidate_doc_ids
            ]
            return sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)

        return [(doc_id, 1.0 / (idx + 1)) for idx, doc_id in enumerate(candidate_doc_ids)]
