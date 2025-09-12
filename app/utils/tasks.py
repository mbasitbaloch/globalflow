import json
import uuid
from celery import Celery
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from ..config import settings
from ..models import Translation
from datetime import datetime
from openai import OpenAI
from qdrant_client import QdrantClient
from ..database import SessionLocal
from ..mongodb import users_collection

celery_app = Celery(
    'tasks',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

COLLECTION_NAME = settings.COLLECTION_NAME


@celery_app.task()
def store_data(translated_data, req, raw_data):
    db = SessionLocal()

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    qdrant = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=60
    )
    collection_exists = False
    if COLLECTION_NAME:
        collection_exists = qdrant.collection_exists(COLLECTION_NAME)
        if not collection_exists:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                ),
            )
    try:
        print("Storing data in the postgresql...")
        today_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        json_blob = json.dumps(translated_data, ensure_ascii=False)

        user = users_collection.find_one(
            {"shopifyStores.shopDomain": req["shopDomain"]})

        newObj = Translation(
            user_id=str(user["_id"]) if user else None,
            industry=user.get("industry") if user else "Unknown",
            shop_domain=req["shopDomain"],
            brand_tone=req["brandTone"],
            original_text=raw_data,
            translated_text=translated_data,
            target_lang=req["targetLanguage"],
            content_type="json",
            translated_at=today_date
        )
        db.add(newObj)
        db.commit()
        db.refresh(newObj)

        print("Postgresql data stored successfully.")
        print("Storing embedding in Qdrant...")

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=json_blob
        )

        embedding = response.data[0].embedding  # <-- correct way

        translation_point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "shopDomain": req["shopDomain"],
                "targetLanguage": req["targetLanguage"],
                "translated_text": translated_data,
                "brandTone": req["brandTone"],
                "date": today_date
            }
        )

        if COLLECTION_NAME is None:
            raise ValueError(
                "COLLECTION_NAME environment variable is not set.")
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[translation_point]
        )
        count = qdrant.count(
            collection_name=COLLECTION_NAME,
            exact=True
        )
        print("Qdrant embedding stored successfully.")
    except Exception as e:
        db.rollback()
        return str(e)
    finally:
        db.close()
