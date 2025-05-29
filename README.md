This streamlit app answers simple questions using an LLM.
If the question is similar enough to a provided question (according to bm25 sparse search), it bypasses the LLM and provides a hardcoded answer.
The BM25 threshold can be configured to make the system more/less likely to respond with a hardcoded answer.

How to run:
streamlit run bm25_chat.py
