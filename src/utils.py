import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Union, Tuple

# Try importing pyvi tokenizer with fallback
try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

try:
    from underthesea import word_tokenize as uts_word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False


def load_json(file_path: Union[str, Path]) -> dict:
    """Load JSON file with UTF-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, file_path: Union[str, Path], indent: int = 4) -> None:
    """Save data to JSON file with UTF-8 encoding."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def normalize_vietnamese_text(text: str) -> str:
    """
    Clean and normalize Vietnamese legal text.
    - Convert to Unicode NFC format
    - Remove extra spaces, strange tab/newline combinations
    - Retain key legal punctuation and numbers
    """
    if not text:
        return ""
    
    # Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # Replace carriage returns and weird whitespace
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\t\f\v]', ' ', text)
    
    # Replace multiple consecutive newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove leading/trailing spaces on lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = ' '.join(lines)
    
    # Clean multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_segment(text: str) -> str:
    """
    Perform Vietnamese word segmentation with fallback support.
    """
    text = normalize_vietnamese_text(text)
    if HAS_PYVI:
        return ViTokenizer.tokenize(text)
    elif HAS_UNDERTHESEA:
        return uts_word_tokenize(text, format="text")
    else:
        # Simple regex-based fallback tokenization
        return text


def evaluate_predictions(
    ground_truth: Dict[str, Dict[str, Union[str, List[str]]]],
    predictions: Dict[str, Dict[str, Union[str, List[str]]]]
) -> Tuple[float, float]:
    """
    Evaluate Mean Recall and Mean Precision using BTC's official scoring logic.
    
    Scoring rules (from BTC's scoring.py):
    - Predictions must have the same number of samples as ground truth.
    - Each prediction must have 1-5 answers (0 or >5 results in score = 0 for that sample).
    - Recall = |intersection| / |truth| averaged over all truth samples.
    - Precision = |intersection| / |predicted| averaged over all predicted samples.
    
    Args:
        ground_truth: Dict with key = q_id, val = {"question": ..., "answer": [doc_id1, doc_id2]}
        predictions: Dict with key = q_id, val = {"question": ..., "answer": [pred_doc_id1, ...]}
        
    Returns:
        (mean_recall, mean_precision)
    """
    import numpy as np

    # Extract answer lists keyed by q_id (matching BTC format)
    y_pred = {k: v['answer'] for k, v in predictions.items()}
    y_true = {k: v['answer'] if isinstance(v, dict) else v for k, v in ground_truth.items()}

    ids_preds = list(y_pred.keys())
    ids_truth = list(y_true.keys())

    if len(ids_preds) != len(ids_truth):
        print(f"WARNING: Samples mismatch - predictions: {len(ids_preds)}, truth: {len(ids_truth)}")

    # BTC's official scoring: recall and precision with top-5 constraint
    recall = np.array([
        len(set(y_true[k]) & set(y_pred.get(k, []))) / len(y_true[k])
        if y_pred.get(k) and len(y_pred.get(k)) > 0 and len(y_pred.get(k)) <= 5
        else 0
        for k in ids_truth
    ]).mean()

    precision = np.array([
        len(set(y_true[k]) & set(y_pred.get(k, []))) / len(y_pred[k])
        if y_pred.get(k) and len(y_pred.get(k)) > 0 and len(y_pred.get(k)) <= 5
        else 0
        for k in ids_preds
    ]).mean()

    return float(recall), float(precision)
