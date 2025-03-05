import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# 1️⃣ Define Paths
DATA_DIR = r"C:\Users\bapun\Downloads\Order_approval\data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")
TABLES_JSON_PATH = os.path.join(DATA_DIR, "financial_tables.json")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")

# 2️⃣ Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3️⃣ Load FAISS Index
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"❌ FAISS index file {FAISS_INDEX_PATH} not found! Run `embedder.py` first.")
faiss_pdf = faiss.read_index(FAISS_INDEX_PATH)
print(f"✅ FAISS index loaded with {faiss_pdf.ntotal} entries.")

# 4️⃣ Load Structured Financial Data
if os.path.exists(TABLES_JSON_PATH):
    with open(TABLES_JSON_PATH, "r", encoding="utf-8") as f:
        financial_tables = json.load(f)
else:
    print("⚠️ No structured financial data found.")
    financial_tables = []

# 5️⃣ Load Chunk Metadata (to map from index -> chunk text)
if os.path.exists(CHUNKS_JSON_PATH):
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)
else:
    raise FileNotFoundError("❌ pdf_chunks.json not found! Make sure you ran embedder.py completely.")

# 6️⃣ Retrieve Similar Documents
def retrieve_similar_documents(query, top_k=3):
    """Retrieve top_k most similar chunks and any matching financial table rows."""
    # Embed & Normalize the Query
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search
    distances, indices = faiss_pdf.search(query_embedding, top_k)

    # Store results
    results = {
        "PDF Results": [],
        "Structured Financial Data": []
    }

    # Gather chunk text from the top_k matches
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        if idx < 0:
            continue
        # Retrieve the chunk text from chunk_metadata
        chunk_info = chunk_metadata[idx]
        snippet = chunk_info["text"]
        pdf_file_name = chunk_info["pdf_file"]
        results["PDF Results"].append(
            f"[{pdf_file_name}] chunk #{idx} (distance: {dist:.4f}):\n{snippet}"
        )

    # Now search structured data
    structured_results = []
    query_lower = query.lower()
    for entry in financial_tables:
        if isinstance(entry, dict) and "data" in entry:
            for row in entry["data"]:
                row_str = " | ".join(str(v) for v in row.values())
                if query_lower in row_str.lower():
                    structured_results.append(row)

    if structured_results:
        results["Structured Financial Data"] = structured_results
    else:
        results["Structured Financial Data"] = ["No structured data found."]

    return results

# 7️⃣ Test
if __name__ == "__main__":
    test_query = "What was the revenue in 2023?"
    retrieved = retrieve_similar_documents(test_query)
    print("\n🔍 Query:", test_query)
    print("\nTop PDF Chunks:")
    for item in retrieved["PDF Results"]:
        print(item)
    print("\nStructured Data:")
    print(retrieved["Structured Financial Data"])
