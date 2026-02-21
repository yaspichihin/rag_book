import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1/")
API_KEY = os.getenv("API_KEY", "ollama")
MODEL = os.getenv("MODEL", "qwen2.5-coder:3b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
