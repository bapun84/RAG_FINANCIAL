import os
from retriever import retrieve_similar_documents
from dotenv import load_dotenv
import ollama

# 1️⃣ Load Environment Variables (if needed)
load_dotenv()

# 2️⃣ Initialize Ollama Model
MODEL_NAME = "phi"  # or "llama2" or any local model name recognized by Ollama

def ollama_chat_model(prompt):
    """Sends a user prompt to the locally available Ollama model and returns the model's response."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"max_tokens": 100}
    )
    return response['message']['content']

# 3️⃣ Generate Response
def generate_response(query):
    """
    1) Retrieves the top PDF chunks and any structured data relevant to the query.
    2) Builds a prompt combining the retrieved text & data.
    3) Sends to Ollama for final generation.
    """
    retrieved_docs = retrieve_similar_documents(query)
    top_chunks = retrieved_docs["PDF Results"]
    structured_data = retrieved_docs["Structured Financial Data"]

    # Build a simple prompt:
    prompt_intro = "You are a financial Q&A assistant. Use the data below to answer the user's question accurately.\n\n"
    
    # Include top PDF chunks
    context_chunks = "\n\n---\n\n".join(top_chunks)

    # Include structured data if found
    if structured_data and structured_data[0] != "No structured data found.":
        structured_text = []
        for row in structured_data:
            structured_text.append(str(row))
        structured_text = "\n".join(structured_text)
        prompt_tables = f"\n\nStructured Financial Data:\n{structured_text}\n\n"
    else:
        prompt_tables = "\n\n(No structured financial data was matched)\n\n"

    final_prompt = (
        f"{prompt_intro}"
        f"Query: {query}\n\n"
        f"Relevant PDF Chunks:\n{context_chunks}\n"
        f"{prompt_tables}"
        "Please provide a concise and accurate answer based on the above.\n"
    )

    print("\n===== DEBUG PROMPT TO OLLAMA =====\n")
    print(final_prompt)
    print("==================================\n")

    answer = ollama_chat_model(final_prompt)
    return answer

# 4️⃣ If run as a script, test with a sample query
if __name__ == "__main__":
    test_query = "What was TCS's net profit in 2023?"
    response = generate_response(test_query)
    print("User Query:", test_query)
    print("Response:\n", response)
