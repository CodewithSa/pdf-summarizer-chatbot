from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from src.embedding import embedding

load_dotenv()

chunk_size = os.getenv("CHUNK_SIZE")
chunk_overlap = os.getenv("CHUNK_OVERLAP")

if not chunk_size:
    raise ValueError("enter chunk size")

if not chunk_overlap:
    raise ValueError("Enter the overlap size")

def build_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = splitter.split_documents(document)
    vectorstor = FAISS.from_documents(
        chunks,
        embedding = embedding
    )

    retriever = vectorstor.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever, chunks

