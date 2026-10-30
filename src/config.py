from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    raise ValueError("Enter you api key")

if not MODEL:
    raise ValueError("Enter your groq model")

llm = ChatGroq(
    model = MODEL,
    api_key = GROQ_API_KEY
)