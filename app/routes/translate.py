from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session
import requests
from ..database import SessionLocal
from ..models import Translation
from ..services.translator import fast_translate_json
from fastapi.responses import JSONResponse, FileResponse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, PointStruct, VectorParams, Distance
import json
import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
# from langchain.embeddings import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai

load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

router = APIRouter()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
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


@router.post("/shopify/translate")
async def shopify_translate(req: dict, db: Session = Depends(get_db)):
    today_date = datetime.now().strftime("%Y-%m-%d")
    """
    Expect body:
    {
      "shopDomain": "...",
      "accessToken": "...",
      "targetLanguage": "fr",
      "brandTone": "neutral"
    }
    """
    url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
    response = requests.post(url, json={
        "shopDomain": req["shopDomain"],
        "accessToken": req["accessToken"],
        "targetLanguage": req["targetLanguage"],
        "brandTone": req["brandTone"]
    })
    response.raise_for_status()
    raw_data = response.json()

    translated_data = await fast_translate_json(
        raw_data,
        req["targetLanguage"],
        req["brandTone"]
    )

    json_blob = json.dumps(translated_data, ensure_ascii=False)

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
        raise ValueError("COLLECTION_NAME environment variable is not set.")
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[translation_point]
    )
    count = qdrant.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    newObj = Translation(
        source_text=raw_data,
        target_text=translated_data,
        target_lang=req["targetLanguage"],
        content_type="json"
    )
    db.add(newObj)
    db.commit()
    db.refresh(newObj)

    # Save translated JSON to file
    file_name = f"translated_{uuid.uuid4().hex}.json"
    file_path = os.path.join("tmp", file_name)
    os.makedirs("tmp", exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    translated_data = await fast_translate_json(
        raw_data,
        req["targetLanguage"],
        req["brandTone"]
    )

    # Save translated JSON to file
    file_name = f"translated_{uuid.uuid4().hex}.json"
    file_path = os.path.join("tmp", file_name)
    os.makedirs("tmp", exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    # Return file for download
    return FileResponse(
        file_path,
        media_type="application/json",
        filename=file_name
    )

    # return {"translated_json": translated_data}


# 2 Generic JSON translator (returns JSON in response)
@router.post("/translate-json")
async def translate_json(
    payload: dict = Body(...),
    target_lang: str = "fr",
    brand_tone: str = "neutral"
):
    translated = await fast_translate_json(payload, target_lang, brand_tone)
    return JSONResponse(content=translated)


# 3 JSON translator (returns downloadable file)
@router.post("/translate-json/download")
async def translate_json_download(
    payload: dict = Body(...),
    target_lang: str = "fr",
    brand_tone: str = "neutral"
):
    translated = await fast_translate_json(payload, target_lang, brand_tone)

    filename = f"translated_{uuid.uuid4().hex}.json"
    filepath = os.path.join("tmp", filename)
    os.makedirs("tmp", exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    return FileResponse(
        filepath,
        filename=filename,
        media_type="application/json"
    )


# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# import requests
# from ..workers.tasks import generate_embedding
# import openai

# from ..database import SessionLocal
# from .. import schemas
# from ..services import translator


# router = APIRouter()


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# @router.post("/shopify/translate")
# def shopify_translate(req: schemas.ShopifyTranslateRequest, db: Session = Depends(get_db)):
#     # STEP 1: Fetch Shopify data
#     url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
#     payload = {
#         "shopDomain": req.shopDomain,
#         "accessToken": req.accessToken,
#         "targetLanguage": req.targetLanguage,
#         "brandTone": req.brandTone
#     }
#     response = requests.post(url, json=payload)
#     raw_data = response.json()

#     # STEP 2: Translate recursively with OpenAI
#     translated_data = translator.traverse_and_translate(
#         raw_data, req.targetLanguage, req.brandTone
#     )

#     return {"translated_json": translated_data}


# def traverse_and_translate(data, target_lang: str, brandTone: str):
#     """Recursively translate all string values in JSON"""
#     if isinstance(data, str):
#         # yahan direct string translate hogi with target_lang + brandTone
#         return translator.translate_text(data, target_lang, brandTone)
#     elif isinstance(data, dict):
#         # har key ka value recursively traverse karega
#         return {k: traverse_and_translate(v, target_lang, brandTone) for k, v in data.items()}
#     elif isinstance(data, list):
#         # har item translate karega agar string ya nested dict/list hai
#         return [traverse_and_translate(item, target_lang, brandTone) for item in data]
#     return data


# @router.post("/shopify/translate")
# def shopify_translate(shopDomain: str, accessToken: str, targetLanguage: str, brandTone: str, db: Session = Depends(get_db)):
#     # STEP 1: Fetch Shopify data
#     url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
#     payload = {
#         "shopDomain": shopDomain,
#         "accessToken": accessToken,
#         "targetLanguage": targetLanguage,
#         "brandTone": brandTone
#     }
#     response = requests.post(url, json=payload)
#     raw_data = response.json()

#     # STEP 2: Translate recursively with tone
#     translated_data = traverse_and_translate(
#         raw_data, targetLanguage, brandTone
#     )

#     # STEP 3: Save metadata in DB
#     # record = models.Translation(
#     #     source_text="Shopify JSON",
#     #     target_text="Translated Shopify JSON",
#     #     target_lang=targetLanguage,
#     #     content_type="json"
#     # )

#     # db.add(record)
#     # db.commit()
#     # db.refresh(record)


# # "id": record.id,
#     return {"translated_json": translated_data}
