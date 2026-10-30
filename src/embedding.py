from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = os.getenv("EMBEDDING_MODEL")

if not embedding_model:
    raise ValueError("Enter your embedding model")

embedding = HuggingFaceEmbeddings(
    model_name = embedding_model
)