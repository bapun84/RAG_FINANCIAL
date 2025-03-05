import os
import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -- 1. LOAD DATA: We'll assume you have a per-document text store. For multi-stage,
#    often you keep one "doc-level" text array for BM25 and also your chunk metadata.

DATA_DIR = r"C:\Users\bapun\Downloads\Order_approval\data"
DOCS_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")  # store doc-level text
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")  # chunk-level text
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")

if not os.path.exists(DOCS_JSON_PATH):
    raise FileNotFoundError("Need doc-level JSON for BM25 coarse retrieval!")

# Load doc-level data for BM25
with open(DOCS_JSON_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)
    # Expect something like: documents = [{"doc_id": 1, "text": "entire PDF text ..."}]

# Prepare for BM25
bm25_corpus = [doc["text"].split() for doc in documents]  # simple tokenization
bm25 = BM25Okapi(bm25_corpus)

# Load chunk-level data & FAISS index for fine retrieval
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_metadata = []
with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError("FAISS index not found!")

faiss_index = faiss.read_index(FAISS_INDEX_PATH)

def multi_stage_retrieve(query, top_k_coarse=3, top_k_fine=3):
    """
    Multi-stage retrieval:
      1) Coarse retrieval with BM25 to get top_k_coarse documents.
      2) Fine retrieval with FAISS on chunks from those top docs.
    """
    # Stage 1: BM25 Coarse
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)  # array of length = len(documents)
    
    # Sort doc IDs by BM25 score descending
    doc_ranking = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    top_doc_indices = doc_ranking[:top_k_coarse]

    # Gather chunk indices for only those top documents
    candidate_chunk_indices = []
    for i in top_doc_indices:
        doc_id = documents[i]["doc_id"]
        # Find all chunks from chunk_metadata that belong to this doc
        for idx, meta in enumerate(chunk_metadata):
            # If meta includes the doc_id or filename that matches
            # (you can store doc_id in chunk_metadata for clarity)
            # e.g. if meta["pdf_file"] == documents[i]["file_name"] or something
            # We'll assume doc_id is the link for simplicity:
            if meta.get("doc_id") == doc_id:
                candidate_chunk_indices.append(idx)

    # Stage 2: Fine retrieval with FAISS
    # We'll embed the query and do a partial search on the selected chunk embeddings.
    # But FAISS by default indexes *all* chunks, so we either:
    #   A) Rebuild a smaller FAISS index for just the candidate chunks, or
    #   B) Search everything, but post-filter to keep only candidate_chunk_indices.

    # Let's do approach B for simplicity:
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    distances, indices = faiss_index.search(query_emb, top_k_fine * 10)  # get more than you need

    # Now filter those results to keep only chunks from candidate_chunk_indices
    # and then pick the top_k_fine among them.
    all_hits = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx in candidate_chunk_indices:
            all_hits.append((idx, dist))

    # sort by distance ascending => closer is better
    all_hits.sort(key=lambda x: x[1])
    final_hits = all_hits[:top_k_fine]

    # Build a result structure
    results = []
    for idx, dist in final_hits:
        chunk_info = chunk_metadata[idx]
        results.append({
            "chunk_id": idx,
            "distance": dist,
            "text": chunk_info["text"],
            "pdf_file": chunk_info["pdf_file"],
            # "doc_id": chunk_info["doc_id"], # if you track doc_id
        })

    return results

# Minimal test
if __name__ == "__main__":
    query = "What was the revenue in 2023?"
    final_results = multi_stage_retrieve(query)
    for r in final_results:
        print(f"{r['pdf_file']} | dist={r['distance']:.4f} | {r['text'][:100]}...")
