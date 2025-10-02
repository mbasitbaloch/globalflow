from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from ..config import settings
import datetime
import os

# Qdrant client
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

# OpenAI embeddings model (small = fast, large = better accuracy)
embedding_model = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY_1, model="text-embedding-3-small")
collection_name = os.getenv("COLLECTION_NAME")

# Recreate / ensure collection exists
# client.recreate_collection(
#     collection_name="glflow",
#     # text-embedding-3 returns 1536 dims
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
# )


def embed_and_store(text: str, translation_id: int):
    """Generate OpenAI embedding for text and store in Qdrant"""
    # Get embedding from OpenAI
    embedding = embedding_model.embed_query(text)

    today_date = datetime.date.today().isoformat()

    # Upsert into Qdrant
    client.upsert(
        collection_name="glflow",
        points=[
            models.PointStruct(
                id=translation_id,
                vector=embedding,
                payload={
                    "text": text,
                    "date_created": today_date,
                    "translation_id": translation_id
                }
            )
        ],
    )


# def embed_and_store(text: str, translation_id: int):
#     model = genai.GenerativeModel("text-embedding-004")
#     embedding = model.embed_content([text]).embedding

#     today_date = datetime.date.today().isoformat()
#     client.upsert(
#         collection_name="translations",
#         points=[models.PointStruct(
#             id=translation_id,
#             vector=embedding,
#             payload={"text": text,
#                      "date_created": today_date,
#                      "translation_id": translation_id
#                      }
#         )],
#     )
