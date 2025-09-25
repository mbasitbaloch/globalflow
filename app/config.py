import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    POSTGRE_SQL: str = os.getenv("POSTGRE_SQL", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "")
    GEMINI_API_KEY_1: str = os.getenv("GEMINI_API_KEY_1", "")
    GEMINI_API_KEY_2: str = os.getenv("GEMINI_API_KEY_2", "")
    OPENAI_API_KEY_1: str = os.getenv("OPENAI_API_KEY_1", "")
    OPENAI_API_KEY_2: str = os.getenv("OPENAI_API_KEY_2", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    MONGO_URL: str = os.getenv("MONGO_URL", "")
    MONGO_DB: str = os.getenv("MONGO_DB", "")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "")
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")


settings = Settings()
