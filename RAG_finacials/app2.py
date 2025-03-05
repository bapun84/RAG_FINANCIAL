import streamlit as st
from generator import generate_response

def main():
    st.title("RAG Financial Q&A")

    st.write("Ask questions about TCS financials (or any provided data).")

    # If you want to demonstrate both Basic & Multi-Stage retrieval:
    # retrieval_mode = st.selectbox("Select Retrieval Mode:", ["basic", "multi-stage"])
    # Then pass mode=retrieval_mode in generate_response below.
    
    user_query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if user_query.strip():
            with st.spinner("Generating answer..."):
                # If your `generate_response` can accept a mode param, pass it here:
                # answer = generate_response(user_query, mode=retrieval_mode)
                answer = generate_response(user_query)
                st.markdown("### Answer")
                st.write(answer)
        else:
            st.warning("Please enter a query first.")

if __name__ == "__main__":
    main()

