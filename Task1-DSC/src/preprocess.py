import glob
import os
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

from src.config import CORPUS_DIR, PROCESSED_CORPUS_PATH, MAX_CHUNK_TOKENS, CHUNK_OVERLAP
from src.utils import normalize_vietnamese_text, word_segment, save_json, load_json


def chunk_passage(text: str, max_words: int = MAX_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunk long text using a sliding window strategy based on word count.
    Attempts to break cleanly on sentences/paragraphs where possible.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        if end == len(words):
            break
        start += (max_words - overlap)

    return chunks


def process_corpus(force_reprocess: bool = False) -> List[Dict[str, Any]]:
    """
    Process all context JSON files in CORPUS_DIR and create a chunked corpus.
    Caches result to PROCESSED_CORPUS_PATH.
    
    Returns:
        List of chunk dicts: [
            {
                "chunk_id": "100050_0",
                "doc_id": "100050",
                "doc_name": "TCVN-13268-1-2021...",
                "text": "Original chunk text...",
                "segmented_text": "Tách_từ tiếng_Việt..."
            }, ...
        ]
    """
    if not force_reprocess and os.path.exists(PROCESSED_CORPUS_PATH):
        try:
            cached_data = load_json(PROCESSED_CORPUS_PATH)
            if cached_data and len(cached_data) > 0:
                print(f"[Preprocessing] Loading cached corpus ({len(cached_data)} chunks) from: {PROCESSED_CORPUS_PATH}")
                return cached_data
            else:
                print(f"[Preprocessing] Cached corpus empty. Reprocessing...")
        except Exception as e:
            print(f"[Preprocessing] Error reading cached corpus ({e}). Reprocessing...")

    print(f"[Preprocessing] Reading context files from: {CORPUS_DIR}")
    pattern = str(CORPUS_DIR / "context_*.json")
    file_list = glob.glob(pattern)
    if not file_list:
        file_list = glob.glob(str(CORPUS_DIR.parent / "**" / "context_*.json"), recursive=True)
    if not file_list:
        file_list = glob.glob(str(CORPUS_DIR.parent / "*.json"))
        
    print(f"[Preprocessing] Found {len(file_list)} document files.")

    processed_chunks = []
    
    for file_path in tqdm(file_list, desc="Processing Legal Documents"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            doc_id = str(doc_data.get('id', ''))
            doc_name = doc_data.get('name', '')
            passage = doc_data.get('passage', '')
            
            if not doc_id or not passage:
                continue

            cleaned_passage = normalize_vietnamese_text(passage)
            text_chunks = chunk_passage(cleaned_passage, max_words=MAX_CHUNK_TOKENS, overlap=CHUNK_OVERLAP)

            for i, chunk_text in enumerate(text_chunks):
                # Combine title/name with chunk for richer context
                combined_text = f"{doc_name}. {chunk_text}" if doc_name else chunk_text
                segmented = word_segment(combined_text)
                
                processed_chunks.append({
                    "chunk_id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "text": combined_text,
                    "segmented_text": segmented
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"[Preprocessing] Created total {len(processed_chunks)} chunks from {len(file_list)} documents.")
    if processed_chunks:
        save_json(processed_chunks, PROCESSED_CORPUS_PATH, indent=2)
    return processed_chunks


if __name__ == "__main__":
    process_corpus(force_reprocess=True)
