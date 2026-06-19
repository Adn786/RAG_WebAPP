"""Shared helpers: logging, hashing, id generation, file-type checks."""
import hashlib
import logging
import struct
import uuid


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_id() -> str:
    return str(uuid.uuid4())


def string_id_to_int64(string_id: str) -> int:
    """Deterministically maps a string UUID to a signed int64 for FAISS IndexIDMap,
    which only accepts integer ids."""
    digest = hashlib.sha256(string_id.encode("utf-8")).digest()[:8]
    return struct.unpack(">q", digest)[0]


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""
