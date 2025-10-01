from datetime import datetime
from fastapi import APIRouter, Depends, Body, HTTPException
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from sqlalchemy.orm import Session
import requests
from ..database import SessionLocal
# from ..models import Translation
from ..services.translator5 import fast_translate_json
# from ..mongodb import users_collection
# from fastapi.responses import JSONResponse, FileResponse
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, PointStruct, VectorParams, Distance
import json
import os
import uuid
# from bson import ObjectId
from dotenv import load_dotenv
# from openai import OpenAI
from ..utils.tasks import store_data
# from langchain.embeddings import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores.qdrant import Qdrant
# from langchain_qdrant import Qdrant
from ..config import settings
from ..mongodb import users_collection


COLLECTION_NAME = settings.COLLECTION_NAME

router = APIRouter()


@router.post("/shopify/translate")
async def shopify_translate(req: dict):
    """
    Expect body:
    {
      "shopDomain": "...",
      "accessToken": "...",
      "targetLanguage": "fr",
      "brandTone": "neutral",
    }
    """
    user = users_collection.find_one({"shopifyStores.shopDomain": req["shopDomain"]})
    if not user:
        raise HTTPException(status_code=404, detail="Shop not found")
    url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
    response = requests.post(url, json={
        "shopDomain": req["shopDomain"],
        "accessToken": req["accessToken"],
        "targetLanguage": req["targetLanguage"],
        "brandTone": req["brandTone"]
    })
    response.raise_for_status()
    raw_data = response.json()
    today_date = datetime.now().strftime("%Y-%m-%d")

    # Save original JSON to file
    file_name = f"fetched_{uuid.uuid4().hex}.json"
    file_path = os.path.join("fetched_data", file_name)
    os.makedirs("fetched_data", exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    translated_data = await fast_translate_json(
        raw_data,
        req["targetLanguage"],
        req["brandTone"]
    )

    print("Celery task started...")
    task = store_data.delay(translated_data, req, raw_data)  # type: ignore
    print(f"New task ID: {task.id}")

    # Save translated JSON to file
    print("Saving translated JSON to file...")

    file_name = f"translated_{uuid.uuid4().hex}.json"
    file_path = os.path.join("tmp", file_name)
    os.makedirs("tmp", exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    print("Translated JSON saved to file:", file_path)

    # Return file for download
    return {
        "message": "Translation started",
        "task_id": task.id,
        "file_path": file_path
    }


# @router.put("/shopify/update-string")
# async def update_translated_string(req: dict):
#     """
#     Expect body:
#     {
#       "shopDomain": "...",
#       "targetLanguage": "fr",
#       "path": "products.0.title",   # dot notation path
#       "newValue": "Mon nouveau titre"
#     }
#     """
#     shop_domain = req.get("shopDomain")
#     lang = req.get("targetLanguage")
#     path = req.get("path")
#     new_value = req.get("newValue")

#     if not all([shop_domain, lang, path, new_value]):
#         raise HTTPException(
#             status_code=400, detail="Missing fields in request")

#     db: Session = SessionLocal()
#     try:
#         # 1. Fetch the latest translation
#         translation = db.query(Translation).filter_by(
#             shop_domain=shop_domain,
#             target_lang=lang
#         ).order_by(Translation.translated_at.desc()).first()

#         if not translation:
#             raise HTTPException(status_code=404, detail="No translation found")

#         data = translation.translated_text  # already JSON/dict

#         # 2. Walk JSON to set new value
#         keys = path.split(".")
#         ref = data
#         for k in keys[:-1]:
#             ref = ref[int(k)] if k.isdigit() else ref[k]
#         ref[keys[-1]] = new_value

#         # 3. Save back to Postgres
#         translation.translated_text = data
#         translation.translated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         db.commit()
#         db.refresh(translation)

#         return {
#             "status": "ok",
#             "updatedPath": path,
#             "newValue": new_value,
#             "updatedJson": data
#         }

#     finally:
#         db.close()
