import argparse
import sys
from pathlib import Path

# Ensure src/ is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocess import process_corpus
from src.evaluate import evaluate_pipeline
from src.predict import generate_submission


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Legal Information Retrieval Pipeline")
    parser.add_argument("--preprocess", action="store_true", help="Force reprocessing of corpus JSON files")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on validation split")
    parser.add_argument("--predict", action="store_true", help="Generate predictions on public test set")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: Preprocess -> Evaluate -> Predict")

    args = parser.parse_args()

    # If no flags passed, run --all by default
    if not (args.preprocess or args.eval or args.predict or args.all):
        args.all = True

    if args.preprocess or args.all:
        print("[Pipeline] Step 1: Preprocessing Legal Corpus...")
        process_corpus(force_reprocess=args.preprocess)

    if args.eval or args.all:
        print("[Pipeline] Step 2: Evaluating Retrieval & Reranking Metrics...")
        evaluate_pipeline()

    if args.predict or args.all:
        print("[Pipeline] Step 3: Generating Final Submission Output...")
        generate_submission()


if __name__ == "__main__":
    main()
