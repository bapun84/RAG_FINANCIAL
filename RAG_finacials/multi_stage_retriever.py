import os
import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ========== FILE PATHS ==========
# Adjust DATA_DIR to wherever your JSON & FAISS files are located.
DATA_DIR = r"C:\Users\bapun\Downloads\Order_approval\data"
DOCS_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")      # doc-level JSON for BM25
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")  # chunk-level text + metadata
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")   # your FAISS vector index

# ========== LOAD DOC-LEVEL DATA (BM25) ==========
if not os.path.exists(DOCS_JSON_PATH):
    raise FileNotFoundError("Need doc-level JSON for BM25 coarse retrieval!")

with open(DOCS_JSON_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)
    # Expect format: [ {"doc_id": 1, "text": "entire PDF text..."}, {...}, ...]

# Build BM25 corpus
bm25_corpus = [doc["text"].split() for doc in documents]  # Basic tokenization
bm25 = BM25Okapi(bm25_corpus)

# ========== LOAD CHUNK-LEVEL DATA & FAISS INDEX (Stage 2) ==========
if not os.path.exists(CHUNKS_JSON_PATH):
    raise FileNotFoundError(f"No chunk metadata found at {CHUNKS_JSON_PATH}!")

with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)
    # Expect format: [ {"pdf_file": "xyz.pdf", "text": "...", "doc_id": 1}, ...]

if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError("FAISS index not found! Run your embedder script first.")

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
model = SentenceTransformer("all-MiniLM-L6-v2")  # same embedding model used in embedder.py

# ========== MULTI-STAGE RETRIEVER FUNCTION ==========
def multi_stage_retrieve(query, top_k_coarse=3, top_k_fine=3):
    """
    Multi-stage retrieval pipeline:
      Stage 1: BM25 over doc-level text to pick top_k_coarse documents.
      Stage 2: FAISS search over chunk-level embeddings, but only for those top docs.
    Returns a list of final chunk hits, each with:
      {"chunk_id": idx, "distance": dist, "text": "...", "pdf_file": "..."}
    """

    # --- Stage 1: Coarse retrieval with BM25 ---
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)  # length = len(documents)
    # Sort doc indices by descending BM25 score
    doc_ranking = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    top_doc_indices = doc_ranking[:top_k_coarse]

    # Identify which chunk indices belong to those top docs
    candidate_chunk_indices = []
    for i in top_doc_indices:
        doc_id = documents[i]["doc_id"]  # Make sure your doc-level JSON has "doc_id" too
        # Collect chunks referencing this doc_id
        for idx, meta in enumerate(chunk_metadata):
            if meta.get("doc_id") == doc_id:
                candidate_chunk_indices.append(idx)

    # --- Stage 2: Fine retrieval with FAISS ---
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)  # normalize
    # Search among all chunks in FAISS, but we only keep hits for candidate chunks
    distances, indices = faiss_index.search(query_emb, top_k_fine * 10)

    # Post-filter the hits to keep only candidate_chunk_indices
    all_hits = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx in candidate_chunk_indices:
            all_hits.append((idx, dist))

    # Sort by ascending distance (closer = better match)
    all_hits.sort(key=lambda x: x[1])
    final_hits = all_hits[:top_k_fine]

    # Build a friendly list of final results
    results = []
    for idx, dist in final_hits:
        chunk_info = chunk_metadata[idx]
        results.append({
            "chunk_id": idx,
            "distance": dist,
            "text": chunk_info["text"],
            "pdf_file": chunk_info["pdf_file"],
            # "doc_id": chunk_info.get("doc_id")
        })

    return results

# ========== QUICK TEST ==========
if __name__ == "__main__":
    test_query = "What was the revenue in 2023?"
    final_results = multi_stage_retrieve(test_query)
    print(f"Query: {test_query}")
    print("Top Multi-Stage Results:\n")
    for r in final_results:
        snippet = r['text'][:100].replace("\n", " ")
        print(f"pdf_file={r['pdf_file']} dist={r['distance']:.4f} snippet={snippet}...")
