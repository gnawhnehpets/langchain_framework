import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/openai-rhyme/invoke",
        json={ "input": { "topic": input_text } }
    )
    return response.json()["output"]["content"]


def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/ollama-rhyme/invoke",
        json={ "input": { "topic": input_text } }
    )
    return response.json()["output"]

st.title("langchain ui with langserve")
input_text = st.text_input("Write a rhyme on:")
input_text1 = st.text_input("What are major findings in:")

if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))