"""Embedding/query cache. Uses Redis if a REDIS_URL is supplied and reachable,
otherwise transparently falls back to an in-memory dict so the app keeps working
without Redis installed."""
import json

from utils import setup_logger

logger = setup_logger(__name__)


class CacheService:
    def __init__(self, redis_url: str = None):
        self.backend = None
        self._memory = {}
        if redis_url:
            try:
                import redis  # imported lazily so it's only required if REDIS_URL is set
                self.backend = redis.from_url(redis_url, decode_responses=True)
                self.backend.ping()
                logger.info("Connected to Redis at %s", redis_url)
            except Exception as exc:
                logger.warning("Redis unavailable (%s); falling back to in-memory cache", exc)
                self.backend = None

    def get(self, key: str):
        if self.backend:
            try:
                val = self.backend.get(key)
                return json.loads(val) if val else None
            except Exception as exc:
                logger.warning("Redis GET failed: %s", exc)
                return None
        return self._memory.get(key)

    def set(self, key: str, value, ttl: int = 3600):
        if self.backend:
            try:
                self.backend.set(key, json.dumps(value), ex=ttl)
                return
            except Exception as exc:
                logger.warning("Redis SET failed: %s", exc)
        self._memory[key] = value

    # Convenience wrappers -------------------------------------------------
    def get_embedding(self, text_hash: str):
        return self.get(f"emb:{text_hash}")

    def set_embedding(self, text_hash: str, vector):
        self.set(f"emb:{text_hash}", vector, ttl=86400)

    def get_query(self, key: str):
        return self.get(f"qry:{key}")

    def set_query(self, key: str, value):
        self.set(f"qry:{key}", value, ttl=300)
