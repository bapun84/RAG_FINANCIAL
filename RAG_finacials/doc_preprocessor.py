import os
import json
import pdfplumber

DATA_DIR = r"C:\Users\bapun\Downloads\Order_approval\data"  # wherever your PDFs are
DOC_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")

def extract_entire_pdf_text(pdf_path):
    """Reads a PDF and returns the entire text as a single string."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
    return "\n".join(full_text)

def create_doc_level_json(data_dir=DATA_DIR, output_path=DOC_JSON_PATH):
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError("No PDF files found in the data folder!")

    documents = []
    doc_id_counter = 1
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"Processing {pdf_file} for doc-level text...")
        entire_text = extract_entire_pdf_text(pdf_path)

        # Build the doc record
        doc_record = {
            "doc_id": doc_id_counter,       # or you could store the filename
            "pdf_file": pdf_file,
            "text": entire_text
        }
        documents.append(doc_record)
        doc_id_counter += 1

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4)

    print(f"\nAll doc-level text saved to: {output_path}")

if __name__ == "__main__":
    create_doc_level_json()
