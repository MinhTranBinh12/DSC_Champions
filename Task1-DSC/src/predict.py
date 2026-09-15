import os
import zipfile
from tqdm import tqdm
from typing import Dict, Any

from src.config import PUBLIC_TEST_PATH, SUBMISSION_PATH, SUBMISSION_ZIP_PATH, DYNAMIC_THRESHOLD_RATIO
from src.utils import load_json, save_json
from src.preprocess import process_corpus
from src.retrieval_bm25 import BM25Retriever
from src.retrieval_dense import DenseRetriever
from src.reranker import LegalReranker
from src.evaluate import build_doc_chunks_map, run_pipeline_for_question


def create_submission_zip(json_path: str = SUBMISSION_PATH, zip_path: str = SUBMISSION_ZIP_PATH):
    """
    Create a submission.zip archive containing ONLY submission.json as required by competition rules.
    """
    print(f"[Predict] Creating zip archive at: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Arcname ensures submission.json is at the root of the zip archive without parent folder path
        zipf.write(json_path, arcname="submission.json")
    print(f"[Predict] Successfully created {zip_path} containing submission.json.")


def generate_submission(
    test_path: str = PUBLIC_TEST_PATH,
    output_path: str = SUBMISSION_PATH,
    threshold_ratio: float = DYNAMIC_THRESHOLD_RATIO
) -> Dict[str, Any]:
    """
    Generate final predictions for public test dataset and save submission.json and submission.zip.
    Format required:
    {
      "147194": {
        "answer": ["177504", "740"]
      }
    }
    """
    print("=" * 60)
    print("      GENERATING SUBMISSION FOR PUBLIC TEST DATASET     ")
    print("=" * 60)

    print(f"[Predict] Loading public test dataset from: {test_path}")
    test_data = load_json(test_path)
    print(f"[Predict] Loaded {len(test_data)} test questions.")

    # Load corpus and build indices
    corpus_chunks = process_corpus()
    doc_chunks_map = build_doc_chunks_map(corpus_chunks)

    bm25 = BM25Retriever()
    bm25.build_index(corpus_chunks)

    dense = DenseRetriever()
    dense.build_index(corpus_chunks)

    reranker = LegalReranker()

    predictions = {}
    print(f"\n[Predict] Running inference on public test questions...")
    
    for q_id, sample in tqdm(test_data.items(), desc="Predicting Test Questions"):
        question = sample.get("question", "")
        pred_answers = run_pipeline_for_question(
            question=question,
            bm25_retriever=bm25,
            dense_retriever=dense,
            reranker=reranker,
            doc_chunks_map=doc_chunks_map,
            threshold_ratio=threshold_ratio
        )
        
        # Ensure at least 1 document ID is returned as fallback
        if not pred_answers:
            pred_answers = ["100050"]  # Fallback to default ID if unpredicted

        # Strictly enforce max 5 document_ids constraint per question to avoid zero score penalty
        pred_answers_capped = [str(ans) for ans in pred_answers[:5]]

        # Match exact schema required by Section 5 instructions: {"answer": ["id1", "id2"]}
        predictions[q_id] = {
            "answer": pred_answers_capped
        }

    # Sanity Checks
    print("\n[Predict] Performing Sanity Checks on generated submission...")
    assert len(predictions) == len(test_data), f"Mismatch in count: {len(predictions)} vs {len(test_data)}"
    
    null_count = sum(1 for v in predictions.values() if v.get("answer") is None or len(v.get("answer", [])) == 0)
    over_limit_count = sum(1 for v in predictions.values() if len(v.get("answer", [])) > 5)

    print(f"[Predict] Total questions: {len(predictions)}")
    print(f"[Predict] Null/empty answers count: {null_count}")
    print(f"[Predict] Over 5 doc_ids count: {over_limit_count}")
    
    if null_count == 0 and over_limit_count == 0:
        print("[Predict] Sanity check PASSED! All test questions strictly satisfy 1 <= len(answer) <= 5.")
    else:
        print("[Predict] Warning: Potential constraint violations detected!")

    print(f"\n[Predict] Saving submission JSON to: {output_path}")
    save_json(predictions, output_path, indent=4)

    # Compress into submission.zip containing ONLY submission.json
    create_submission_zip(output_path, SUBMISSION_ZIP_PATH)

    print("=" * 60)
    print("                   PREDICTION COMPLETE                  ")
    print("=" * 60)

    return predictions


if __name__ == "__main__":
    generate_submission()
