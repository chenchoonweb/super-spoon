# utils/vectorstore.py
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
