import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

'''
https://wow.groq.com/why-groq/

The LPU™ Inference Engine by Groq is a hardware and software platform that 
delivers exceptional compute speed, quality, and energy efficiency. 

The LPU is designed to overcome the two LLM bottlenecks: compute density 
and memory bandwidth. An LPU has greater compute capacity than a GPU and 
CPU in regards to LLMs. This reduces the amount of time per word calculated, 
allowing sequences of text to be generated much faster. Additionally, 
eliminating external memory bottlenecks enables the LPU Inference Engine 
to deliver orders of magnitude better performance on LLMs compared to GPUs.
'''

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200 )
    st.session_state.final_docs = st.session_state.text_splitter.split_documents( st.session_state.docs)

    st.session_state.vectors = FAISS.from_documents( st.session_state.final_docs, st.session_state.embeddings )

st.title("ChatGroq demo")

llm=ChatGroq( groq_api_key=groq_api_key, 
             model="llama3-8b-8192" )

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context.
    Please provide the most accurate response based on the question. 
    Do not provide any irrelevant information.

    Context: {context}

    Questios: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input( "What question do you have?" )

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time:", time.process_time() - start)
    st.write(response["answer"])

    # streamlit expander
    with st.expander("similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------")