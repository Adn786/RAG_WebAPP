"""Central configuration, read from environment variables."""
import os


class Config:
    GITHUB_PAT = os.getenv("GITHUB_PAT")
    REDIS_URL = os.getenv("REDIS_URL")  # optional, e.g. redis://localhost:6379
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    PORT = int(os.getenv("PORT", "5000"))

    DATA_DIR = os.getenv("DATA_DIR", "./data")
    FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.pkl")
    SQLITE_DB_PATH = os.path.join(DATA_DIR, "chunks.db")

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    EMBEDDING_DIM = 1536  # text-embedding-3-small output dimension
    TOP_K = 10

    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS = {"pdf", "docx", "pptx", "txt", "xlsx", "csv"}

    @classmethod
    def validate(cls):
        if not cls.GITHUB_PAT:
            raise ValueError("GITHUB_PAT environment variable not set")
        os.makedirs(cls.DATA_DIR, exist_ok=True)
