import os

import dotenv

dotenv.load_dotenv(".envrc")

PDF_PATH = os.getenv("PDF_PATH")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR") or "vector_db"
