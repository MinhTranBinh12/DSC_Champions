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


def compute_sample_recall(y_true: Set[str], y_pred: Set[str]) -> float:
    """Compute Recall for a single sample. Disqualified (Recall=0) if len(y_pred) > 5."""
    if not y_true or len(y_pred) > 5:
        return 0.0
    intersection = y_true.intersection(y_pred)
    return len(intersection) / len(y_true)


def compute_sample_precision(y_true: Set[str], y_pred: Set[str]) -> float:
    """Compute Precision for a single sample. Disqualified (Precision=0) if len(y_pred) > 5."""
    if not y_pred or len(y_pred) > 5:
        return 0.0
    intersection = y_true.intersection(y_pred)
    return len(intersection) / len(y_pred)


def evaluate_predictions(
    ground_truth: Dict[str, Dict[str, Union[str, List[str]]]],
    predictions: Dict[str, Dict[str, Union[str, List[str]]]]
) -> Tuple[float, float]:
    """
    Evaluate Mean Recall and Mean Precision across all questions in the dataset.
    
    Args:
        ground_truth: Dict with key = q_id, val = {"question": ..., "answer": [doc_id1, doc_id2]}
        predictions: Dict with key = q_id, val = {"question": ..., "answer": [pred_doc_id1, ...]}
        
    Returns:
        (mean_recall, mean_precision)
    """
    total_recall = 0.0
    total_precision = 0.0
    count = 0

    for q_id, sample in ground_truth.items():
        true_answers = set(str(ans) for ans in sample.get('answer', []))
        if not true_answers:
            continue

        pred_sample = predictions.get(q_id, {})
        pred_answers = set(str(ans) for ans in pred_sample.get('answer', []))

        rec = compute_sample_recall(true_answers, pred_answers)
        prec = compute_sample_precision(true_answers, pred_answers)

        total_recall += rec
        total_precision += prec
        count += 1

    if count == 0:
        return 0.0, 0.0

    mean_recall = total_recall / count
    mean_precision = total_precision / count
    return mean_recall, mean_precision
