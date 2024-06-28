from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ( "system", "You are a helpful assistant. Please respond to the queries of the user." ),
        ( "user", "Question: {question}" )
    ]
)

## Streamlit

st.title("langchain ui (w/ Ollama)")
input_text = st.text_input("Describe hybrid search", "What is hybrid search?")

# ollama's LLaMa3 LLM
# download llm locally: ollama run llama3 
# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = Ollama(model="llama3") 
output_parser=StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))