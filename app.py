import streamlit as st
from dotenv import load_dotenv
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    st.write('Made with ⚡ by [Sparks](https://www.fiverr.com/shehansolutions?source=inbox)')


def main():
    load_dotenv()
    st.header("Ask about Neuroscience 💬")

    pdf_path = "Neuro.pdf"
    store_name = os.path.basename(pdf_path)[:-4]

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        st.error(f"Pickle file '{store_name}.pkl' not found. Please make sure the file exists.")

    # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)


if __name__ == '__main__':
    main()
