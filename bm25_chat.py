import streamlit as st
import openai
from openai import OpenAI
from rank_bm25 import BM25Okapi

DEF_THRESHOLD = 0.75
QAS = {
    "questions": [
        {
            "question": "What does the eligibility verification agent (EVA) do?",
            "answer": "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
        },
        {
            "question": "What does the claims processing agent (CAM) do?",
            "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
        },
        {
            "question": "How does the payment posting agent (PHIL) work?",
            "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
        },
        {
            "question": "Tell me about Thoughtful AI's Agents.",
            "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
        },
        {
            "question": "What are the benefits of using Thoughtful AI's agents?",
            "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
        }
    ]
}

# if the user question is similar enough to the existing questions,
# return a pre-made response. Otherwise return None.
def select_hardcoded_response(usertext, threshold=DEF_THRESHOLD):
    tokenized_query = usertext.split(" ")
    doc_scores = list(bm25.get_scores(tokenized_query))
    print(f"{usertext}: {doc_scores}")
    top_score = max(doc_scores)

    if top_score < threshold:
        return None

    topidx = doc_scores.index(top_score)
    return QAS['questions'][topidx]['answer']

# produce corpus for BM25 sparse search algorithm
corpus = [item['question'] for item in QAS['questions']]
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Set page config
st.set_page_config(page_title="Thoughtful.ai assistant", page_icon="🤖")

# Title
st.title("🤖 Chat with Thoughtful.ai assistant")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    bm25_threshold = st.number_input(
        label="Specify BM25 score threshold",
        value=DEF_THRESHOLD,
        step=0.05,
        format="%.2f"
    )

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            client = OpenAI(api_key=api_key)
            
            with st.spinner("Thinking..."):
                
                assistant_response = select_hardcoded_response(prompt, threshold=bm25_threshold)

                # fall back on LLM
                if not assistant_response:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=st.session_state.messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    assistant_response = response.choices[0].message.content

            st.markdown(assistant_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()