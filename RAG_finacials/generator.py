import os
from retriever import retrieve_similar_documents
# Remove ollama, dotenv if unused
# from dotenv import load_dotenv
# import ollama

from transformers import pipeline

# Initialize a small model pipeline
model_name = "google/flan-t5-small"
generator_pipeline = pipeline("text2text-generation", model=model_name)

def generate_response(query):
    """Builds a prompt from retrieved docs & data, then calls Flan-T5 for final answer."""
    retrieved_docs = retrieve_similar_documents(query)
    top_chunks = retrieved_docs["PDF Results"]
    structured_data = retrieved_docs["Structured Financial Data"]

    # Build the prompt
    prompt_intro = "You are a financial Q&A assistant. Use the data below.\n\n"
    context_chunks = "\n\n---\n\n".join(top_chunks)

    if structured_data and structured_data[0] != "No structured data found.":
        structured_text = "\n".join(str(row) for row in structured_data)
        prompt_tables = f"\n\nStructured Data:\n{structured_text}\n\n"
    else:
        prompt_tables = "\n\n(No structured financial data matched)\n\n"

    final_prompt = (
        f"{prompt_intro}"
        f"Query: {query}\n\n"
        f"Relevant PDF Chunks:\n{context_chunks}\n"
        f"{prompt_tables}"
        "Provide a concise answer:\n"
    )

    output = generator_pipeline(final_prompt, max_length=256)
    answer = output[0]["generated_text"]
    return answer

if __name__ == "__main__":
    test_q = "What was the revenue in 2023?"
    ans = generate_response(test_q)
    print("Question:", test_q)
    print("Answer:", ans)
