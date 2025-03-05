import os
import faiss
import numpy as np
import pdfplumber
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# =============== OCR-Related Imports (optional) ===============
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    print("pytesseract/pdf2image not installed. OCR fallback disabled.")
    OCR_AVAILABLE = False
# =============================================================

# 1️⃣ Define Paths
DATA_DIR = r"C:\Users\bapun\Downloads\Order_approval\data"  # Change to your data folder
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")
TABLES_JSON_PATH = os.path.join(DATA_DIR, "financial_tables.json")

# ➡ We will also store the chunk texts in a JSON so we can map FAISS indices back to the actual text
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")

# 2️⃣ Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3️⃣ Helper: Perform OCR on a single PDF page if no text is found
def ocr_page(pdf_path, page_num):
    """
    Convert a single PDF page to an image, then run Tesseract OCR.
    Returns the OCR-extracted text as a string.
    """
    if not OCR_AVAILABLE:
        return ""  # OCR not available
    try:
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        if not images:
            return ""
        ocr_text = pytesseract.image_to_string(images[0])
        return ocr_text.strip()
    except:
        return ""

def extract_text_from_pdf(pdf_path, chunk_size=512):
    """
    Extracts text from each page of the PDF. If a page is scanned
    or has no text, optionally fall back to OCR (if installed).
    Splits the final text into chunks of size chunk_size.
    """
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if not page_text or page_text.strip() == "":
                # Attempt OCR fallback if the page is scanned
                print(f"  ⚠️  No text found on page {i+1}, attempting OCR...")
                page_text = ocr_page(pdf_path, i+1)

            if page_text:
                full_text.append(page_text)

    combined_text = "\n".join(full_text)
    chunks = [combined_text[i : i + chunk_size] for i in range(0, len(combined_text), chunk_size)]
    return chunks


def extract_tables_from_pdf(pdf_path, pdf_name):
    """Extracts tables from a PDF and stores them in structured format."""
    tables_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                # Attempt to parse table into a dataframe
                df = pd.DataFrame(table)

                # If the first row is your header row, rename columns
                df = df.rename(columns=df.iloc[0]).drop(df.index[0])  # make first row the header
                df = df.dropna(how='all')  # remove totally empty rows

                if not df.empty:
                    print(f"✅ Extracted table from {pdf_name}, Page {page_num+1}")
                    table_dict = {
                        "pdf_name": pdf_name,
                        "page": page_num + 1,
                        "data": df.to_dict(orient="records")
                    }
                    tables_data.append(table_dict)
    return tables_data

# 4️⃣ Process All PDFs
pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
all_text_chunks = []   # We'll store the text
chunk_metadata = []    # We'll store (pdf_name, chunk_text) so we can retrieve it later
all_tables = []

if not pdf_files:
    raise ValueError("❌ No PDFs found in the data folder!")

for pdf_file in pdf_files:
    pdf_path = os.path.join(DATA_DIR, pdf_file)
    print(f"📄 Processing: {pdf_file}")
    
    # --- Extract text & chunk it ---
    text_chunks = extract_text_from_pdf(pdf_path)
    for chunk in text_chunks:
        all_text_chunks.append(chunk)
        chunk_metadata.append({
            "pdf_file": pdf_file,
            "text": chunk
        })
    
    # --- Extract tables ---
    extracted_tables = extract_tables_from_pdf(pdf_path, pdf_file)
    all_tables.extend(extracted_tables)

# 5️⃣ Generate Embeddings
print("🧠 Generating embeddings for extracted text...")
embeddings = model.encode(all_text_chunks, convert_to_numpy=True)

# 6️⃣ Store Embeddings in FAISS
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

faiss.write_index(faiss_index, FAISS_INDEX_PATH)
print("✅ PDF embeddings stored in FAISS!")

# 7️⃣ Store Extracted Tables in JSON
with open(TABLES_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_tables, f, indent=4)
print("✅ Financial tables extracted and saved successfully!")

# 8️⃣ Store The Chunk Metadata
with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(chunk_metadata, f, indent=4)
print("✅ PDF chunks (metadata) saved successfully!")

print("🎉 All embeddings and tables are stored successfully!")
