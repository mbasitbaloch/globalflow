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
    __name__,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

COLLECTION_NAME = settings.COLLECTION_NAME

@celery_app.task()
def store_data(translated_data, req, raw_data, postgres_id):
    db = SessionLocal()
    
    # OpenAI client with error handling
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY_1)
    except Exception as e:
        print(f"OpenAI client error: {e}")
        return f"OpenAI client error: {e}"

    # Qdrant client
    qdrant = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=60
    )
    
    # Collection setup with better error handling
    COLLECTION_NAME = settings.COLLECTION_NAME
    if not COLLECTION_NAME:
        raise ValueError("COLLECTION_NAME environment variable is not set.")
    
    collection_exists = qdrant.collection_exists(COLLECTION_NAME)
    if not collection_exists:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1536,  # text-embedding-3-small has 1536 dimensions
                distance=Distance.COSINE
            ),
        )

    try:
        print("Storing data in PostgreSQL...")
        today_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert to JSON string for embedding
        json_blob = json.dumps(raw_data, ensure_ascii=False)

        # Check if JSON is too large for embedding
        if len(json_blob) > 8000:  # OpenAI's limit is 8191 tokens
            print("JSON is large, creating summary for embedding...")
            # Create a summary for large JSON
            summary = create_json_summary(raw_data)
            json_blob = summary

        # Get user data
        user = users_collection.find_one(
            {"shopifyStores.shopDomain": req["shopDomain"]})

        # # Store in PostgreSQL
        # # newObj = Translation(
        # #     user_id=str(user["_id"]) if user else None,
        # #     industry=(user or {}).get("industry") or "Unknown",
        # #     shop_domain=req["shopDomain"],
        # #     brand_tone=req["brandTone"],
        # #     original_text=raw_data,
        # #     translated_text=translated_data,
        # #     target_lang=req["targetLanguage"],
        # #     content_type="json",
        # #     translated_at=today_date
        # # )

        # newObj = Translation(
        #     user_id=str(user["_id"]) if user else None,
        #     industry=(user or {}).get("industry") or "Unknown",
        #     shop_domain=req["shopDomain"],
        #     brand_tone=req["brandTone"],
        #     target_lang=req["targetLanguage"],
        #     content_type="json",
        #     original_text_raw=json.dumps(
        #         raw_data, ensure_ascii=False),
        #     original_text_json=raw_data,
        #     translated_text_raw=json.dumps(
        #         translated_data, ensure_ascii=False),
        #     translated_text_json=translated_data
        # )
        # db.add(newObj)
        # db.commit()
        # db.refresh(newObj)

        # print("PostgreSQL data stored successfully.")
        print("Creating embedding for Qdrant...")

        # Create embedding with OpenAI
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=json_blob,
            encoding_format="float"
        )

        embedding = response.data[0].embedding

        # Create Qdrant point
        translation_point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "shopDomain": req["shopDomain"],
                "user_id": str(user["_id"]) if user else None,
                "targetLanguage": req["targetLanguage"],
                "original_text": raw_data,  
                "translated_text": translated_data,     
                "brandTone": req["brandTone"],
                "date": today_date,
                "postgres_id": postgres_id,           
                "data_type": "complete_json"
            }
        )

        # Store in Qdrant
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[translation_point]
        )
        
        count = qdrant.count(
            collection_name=COLLECTION_NAME,
            exact=True
        )
        
        print("Qdrant embedding stored successfully.")
        print(f"Total points in collection: {count}")
        
        return {
            "status": "success",
            "postgres_id": postgres_id,
            "qdrant_id": translation_point.id,
            "embedding_size": len(embedding)
        }
        
    except Exception as e:
        db.rollback()
        print(f"Error in store_data task: {e}")
        return f"Error: {e}"
    finally:
        db.close()

def create_json_summary(json_data, max_length=8000):
    """
    Large JSON ko summarize karta hai embedding ke liye
    """
    try:
        if isinstance(json_data, dict):
            # Important keys extract karo
            summary = {
                "keys": list(json_data.keys()),
                "total_items": len(json_data),
                "sample_data": {}
            }
            
            # Har key ka sample value
            for key in list(json_data.keys())[:10]:  # First 10 keys
                value = json_data[key]
                if isinstance(value, (dict, list)):
                    summary["sample_data"][key] = f"{type(value).__name__} ({len(value)} items)"
                else:
                    summary["sample_data"][key] = str(value)[:100]
            
            return json.dumps(summary, ensure_ascii=False)[:max_length]
        
        elif isinstance(json_data, list):
            return json.dumps({
                "type": "list",
                "total_items": len(json_data),
                "sample_items": json_data[:5] if len(json_data) > 5 else json_data
            }, ensure_ascii=False)[:max_length]
        
        else:
            return str(json_data)[:max_length]
            
    except Exception as e:
        print(f"Error creating summary: {e}")
        return str(json_data)[:max_length]



@celery_app.task()
def store_examples(orig, trans, paths, shopDomain, target_lang, brand_tone):
    raw_data = []
    today_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for str1, str2, path in zip(orig, trans, paths):
        if "locale" in path:
            continue
        pair = {"original": str1, "translated": str2}
        raw_data.append(pair)

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY_2)
    except Exception as e:
        print(f"OpenAI client error: {e}")
        return f"OpenAI client error: {e}"

    # Qdrant client
    qdrant = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=60
    )
    
    # Collection setup with better error handling
    COLLECTION_NAME = settings.COLLECTION_NAME
    if not COLLECTION_NAME:
        raise ValueError("COLLECTION_NAME environment variable is not set.")
    
    collection_exists = qdrant.collection_exists(COLLECTION_NAME)
    if not collection_exists:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1536,  # text-embedding-3-small has 1536 dimensions
                distance=Distance.COSINE
            ),
        )
    json_blob = json.dumps(raw_data, ensure_ascii=False)
    if len(json_blob) > 8000:  # OpenAI's limit is 8191 tokens
            print("JSON is large, creating summary for embedding...")
            # Create a summary for large JSON
            summary = create_json_summary(raw_data)
            json_blob = summary

    print("Creating embedding for Qdrant...")

        # Create embedding with OpenAI
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=json_blob,
        encoding_format="float"
    )

    embedding = response.data[0].embedding
    # Get user data
    user = users_collection.find_one(
        {"shopifyStores.shopDomain": shopDomain})

    # Create Qdrant point
    translation_point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "type": "prompt_data",
            "shopDomain": shopDomain,
            "user_id": str(user["_id"]) if user else None,
            "targetLanguage": target_lang,
            "fewshot_data": raw_data,     
            "brandTone": brand_tone,
            "date": today_date,          
            "data_type": "complete_json"
        }
    )

    # Store in Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[translation_point]
    )
        
    count = qdrant.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )
        
    print("Qdrant embedding stored successfully.")
    print(f"Total points in collection: {count}")