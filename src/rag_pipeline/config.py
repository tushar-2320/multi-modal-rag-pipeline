import os

import dotenv

dotenv.load_dotenv(".envrc")

PDF_PATH = os.getenv("PDF_PATH")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR") or "vector_db"
TEXT_EMBEDDING_MODEL = os.getenv(
    "TEXT_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
IMAGE_EMBEDDING_MODEL = os.getenv(
    "IMAGE_EMBEDDING_MODEL") or "openai/clip-vit-base-patch16"
