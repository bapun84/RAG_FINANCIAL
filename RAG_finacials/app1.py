import streamlit as st
from generator import generate_response

def main():
    st.title("RAG Financial Q&A")

    st.write("**Ask about the financial statements**")
    
    # Let user pick retrieval approach if you want to show both Basic & Advanced
    retrieval_mode = st.selectbox("Select Retrieval Mode:", ["basic", "multi-stage"])

    user_query = st.text_input("Enter your query here:")
    
    if st.button("Submit"):
        if user_query.strip():
            with st.spinner("Generating answer..."):
                # If your 'generate_response' can accept a 'mode' param:
                answer = generate_response(user_query, mode=retrieval_mode)
                st.markdown("### Answer")
                st.write(answer)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
