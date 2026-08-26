import random
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from src.config import (
    TRAIN_PATH, RANDOM_SEED, VAL_RATIO, FIRST_STAGE_TOP_K,
    DYNAMIC_THRESHOLD_RATIO, MIN_ABSOLUTE_SCORE, MAX_PREDICTED_DOCS
)
from src.utils import load_json, evaluate_predictions
from src.preprocess import process_corpus
from src.retrieval_bm25 import BM25Retriever
from src.retrieval_dense import DenseRetriever
from src.reranker import LegalReranker


def split_train_val(train_data: Dict[str, Any], val_ratio: float = VAL_RATIO, seed: int = RANDOM_SEED):
    """Split train.json into train and validation sets reproducibly."""
    items = list(train_data.items())
    random.seed(seed)
    random.shuffle(items)

    val_size = int(len(items) * val_ratio)
    val_items = dict(items[:val_size])
    train_items = dict(items[val_size:])

    print(f"[Dataset Split] Total: {len(items)} | Train: {len(train_items)} | Val: {len(val_items)}")
    return train_items, val_items


def build_doc_chunks_map(corpus_chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build a mapping from doc_id to list of chunks belonging to that document."""
    doc_map: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in corpus_chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = []
        doc_map[doc_id].append(chunk)
    return doc_map


def run_pipeline_for_question(
    question: str,
    bm25_retriever: BM25Retriever,
    dense_retriever: DenseRetriever,
    reranker: LegalReranker,
    doc_chunks_map: Dict[str, List[Dict[str, Any]]],
    threshold_ratio: float = DYNAMIC_THRESHOLD_RATIO,
    top_k_first_stage: int = FIRST_STAGE_TOP_K
) -> List[str]:
    """
    Run full retrieval & reranking pipeline for a single question.
    Returns list of predicted document IDs.
    """
    # 1. First-Stage Lexical Retrieval (BM25)
    bm25_results = bm25_retriever.search(question, top_k=top_k_first_stage)
    bm25_dict = {doc_id: score for doc_id, score in bm25_results}

    # 2. First-Stage Dense Retrieval
    dense_results = dense_retriever.search(question, top_k=top_k_first_stage)
    dense_dict = {doc_id: score for doc_id, score in dense_results}

    # Combine candidates from BM25 and Dense
    candidate_doc_ids = list(set(list(bm25_dict.keys()) + list(dense_dict.keys())))

    # Normalize & Combine First-Stage Scores
    max_bm25 = max(bm25_dict.values()) if bm25_dict and max(bm25_dict.values()) > 0 else 1.0
    max_dense = max(dense_dict.values()) if dense_dict and max(dense_dict.values()) > 0 else 1.0

    combined_scores: Dict[str, float] = {}
    for doc_id in candidate_doc_ids:
        norm_bm25 = bm25_dict.get(doc_id, 0.0) / max_bm25
        norm_dense = dense_dict.get(doc_id, 0.0) / max_dense
        combined_scores[doc_id] = 0.5 * norm_bm25 + 0.5 * norm_dense

    # Sort candidates by combined score
    sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_first_stage]
    top_candidate_ids = [doc_id for doc_id, _ in sorted_candidates]

    # 3. Second-Stage Re-ranking (Cross-Encoder)
    reranked_results = reranker.rerank(
        question=question,
        candidate_doc_ids=top_candidate_ids,
        doc_chunks_map=doc_chunks_map,
        first_stage_scores=combined_scores
    )

    if not reranked_results:
        return []

    # 4. Post-Processing & Dynamic Thresholding for Top-K Selection
    top1_doc, top1_score = reranked_results[0]
    predicted_docs = [top1_doc]

    threshold = top1_score * threshold_ratio
    for doc_id, score in reranked_results[1:]:
        if score >= threshold and len(predicted_docs) < MAX_PREDICTED_DOCS:
            predicted_docs.append(doc_id)
        else:
            break

    return predicted_docs


def evaluate_pipeline():
    """Run full evaluation on the validation set."""
    print("=" * 60)
    print("      EVALUATING LEGAL DOCUMENT RETRIEVAL PIPELINE      ")
    print("=" * 60)

    train_data = load_json(TRAIN_PATH)
    _, val_data = split_train_val(train_data)

    corpus_chunks = process_corpus()
    doc_chunks_map = build_doc_chunks_map(corpus_chunks)

    # Initialize Retrievers & Reranker
    bm25 = BM25Retriever()
    bm25.build_index(corpus_chunks)

    dense = DenseRetriever()
    dense.build_index(corpus_chunks)

    reranker = LegalReranker()

    print(f"\n[Validation] Running pipeline evaluation on {len(val_data)} validation samples...")
    predictions = {}

    for q_id, sample in tqdm(val_data.items(), desc="Evaluating Questions"):
        question = sample.get("question", "")
        pred_answers = run_pipeline_for_question(
            question=question,
            bm25_retriever=bm25,
            dense_retriever=dense,
            reranker=reranker,
            doc_chunks_map=doc_chunks_map
        )
        predictions[q_id] = {
            "question": question,
            "answer": pred_answers
        }

    mean_recall, mean_precision = evaluate_predictions(val_data, predictions)

    print("\n" + "=" * 60)
    print("                 VALIDATION RESULTS                     ")
    print("=" * 60)
    print(f"  Primary Metric   - Mean Recall    : {mean_recall:.4f}")
    print(f"  Secondary Metric - Mean Precision : {mean_precision:.4f}")
    print("=" * 60)

    return mean_recall, mean_precision


if __name__ == "__main__":
    evaluate_pipeline()
