import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    POSTGRE_SQL: str = os.getenv("POSTGRE_SQL")
    QDRANT_URL: str = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    REDIS_URL: str = os.getenv("REDIS_URL")
    MONGO_URL: str = os.getenv("MONGO_URL")
    MONGO_DB: str = os.getenv("MONGO_DB")


settings = Settings()
