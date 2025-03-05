import os
# If you have both basic and multi-stage:
from retriever import retrieve_similar_documents
from multi_stage_retriever import multi_stage_retrieve

from transformers import pipeline

# 1️⃣ Initialize a small open-source model pipeline
model_name = "google/flan-t5-small"
generator_pipeline = pipeline("text2text-generation", model=model_name)

def generate_response(query, mode="basic"):
    """
    1) Decide which retriever to use (basic vs. multi-stage).
    2) Builds the final prompt from retrieved docs.
    3) Truncate if needed, then run the HF pipeline.
    """
    # Decide which retrieval function to call
    if mode == "multi-stage":
        # If multi-stage is chosen, call that function
        retrieved_docs = multi_stage_retrieve(query)
        # Note: multi_stage_retrieve() might return just a list of chunk hits
        # You can adapt how you store them
        top_chunks = []
        for r in retrieved_docs:
            # Build snippet
            snippet = f"[{r['pdf_file']}] chunk #{r['chunk_id']}, dist={r['distance']:.4f}\n{r['text']}"
            top_chunks.append(snippet)
        # For structured data, you'd handle separately if needed
        structured_data = ["(No structured data in multi-stage, unless you add it)"]
    else:
        # Otherwise default to basic single-stage
        results = retrieve_similar_documents(query)
        top_chunks = results["PDF Results"]
        structured_data = results["Structured Financial Data"]

    # 2️⃣ Build a final_prompt
    prompt_intro = "You are a financial Q&A assistant. Use the data below.\n\n"
    context_chunks = "\n\n---\n\n".join(top_chunks)

    if structured_data and structured_data[0] != "No structured data found.":
        structured_text = "\n".join(str(row) for row in structured_data)
        prompt_tables = f"\n\nStructured Data:\n{structured_text}\n\n"
    else:
        prompt_tables = "\n\n(No structured data matched)\n\n"

    final_prompt = (
        f"{prompt_intro}"
        f"Query: {query}\n\n"
        f"Relevant PDF Chunks:\n{context_chunks}\n"
        f"{prompt_tables}"
        "Provide a concise, accurate answer:\n"
    )

    # 3️⃣ Truncate the prompt if it's too long for Flan-T5
    # typical max tokens is 512, so let's keep ~450 for prompt
    max_prompt_tokens = 450
    prompt_words = final_prompt.split()
    if len(prompt_words) > max_prompt_tokens:
        final_prompt = " ".join(prompt_words[:max_prompt_tokens])
        final_prompt += "\n\n(Truncated due to size)\n"

    # 4️⃣ Generate the answer
    output = generator_pipeline(final_prompt, max_length=256)
    answer = output[0]["generated_text"]
    return answer

# Test block
if __name__ == "__main__":
    # Just try both modes:
    test_query = "What was TCS's net profit in 2023?"
    print("---- BASIC MODE ----")
    print(generate_response(test_query, mode="basic"))

    print("\n---- MULTI-STAGE MODE ----")
    print(generate_response(test_query, mode="multi-stage"))
