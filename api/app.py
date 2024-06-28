from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="langchain api",
    version="0.1.0",
    description="langchain api"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model1=ChatOpenAI(model="gpt-3.5-turbo")
model2=Ollama(model="llama3")

prompt = ChatPromptTemplate.from_template( "Write a rhyme about {topic} in 50 words or less" )
prompt1 = ChatPromptTemplate.from_template( "What are major findings in {topic} research in 100 words or less. Provide 2 bullet points." )

add_routes(
    app,
    prompt|model1,
    path="/openai-rhyme"
)

add_routes(
    app,
    prompt1|model2,
    path="/ollama-rhyme"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)