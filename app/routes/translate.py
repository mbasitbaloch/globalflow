from fastapi import HTTPException
from datetime import datetime
from fastapi import APIRouter, Depends, Body, HTTPException
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy.orm import Session
import requests

from app.routes.ingest import get_db
from ..database import SessionLocal
from ..models import Translation
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

db: Session = SessionLocal()


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
    user = users_collection.find_one(
        {"shopifyStores.shopDomain": req["shopDomain"]})
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

    # print("Celery task started...")
    # task = store_data.delay(translated_data, req, raw_data)  # type: ignore
    # print(f"New task ID: {task.id}")

    # Save translated JSON to file
    print("Saving translated JSON to file...")

    file_name = f"translated_{uuid.uuid4().hex}.json"
    file_path = os.path.join("tmp", file_name)
    os.makedirs("tmp", exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    print("Translated JSON saved to file:", file_path)

    try:
        # Save record to Postgres
        # translation_record = Translation(
        #     user_id=str(user["_id"]),
        # industry=user.get("industry", "general"),
        #     shop_domain=req["shopDomain"],
        #     brand_tone=req["brandTone"],
        #     original_text=raw_data,
        #     translated_text=translated_data,
        #     target_lang=req["targetLanguage"],
        #     content_type="json"
        # )

        translation_record = Translation(
            user_id=str(user["_id"]),
            industry=user.get("industry", "general"),
            shop_domain=req["shopDomain"],
            brand_tone=req["brandTone"],
            target_lang=req["targetLanguage"],
            content_type="json",
            original_text_raw=json.dumps(
                raw_data, ensure_ascii=False),
            original_text_json=raw_data,
            translated_text_raw=json.dumps(
                translated_data, ensure_ascii=False),
            translated_text_json=translated_data

            # # Save raw text
            # original_text_raw=json.dumps(raw_data, ensure_ascii=False),
            # translated_text_raw=json.dumps(
            #     translated_data, ensure_ascii=False),

            # # Save queryable JSONB
            # original_text_json=raw_data,
            # translated_text_json=translated_data

            # user_id=req.str(user["_id"]),
            # industry=user.get("industry", "general"),
            # shop_domain=req["shopDomain"],
            # brand_tone=req["brandTone"],
            # target_lang=req["targetLanguage"],
            # content_type="json",
            # original_text_raw=json.dumps(raw_data, ensure_ascii=False),
            # original_text_json=raw_data,
            # translated_text_raw=json.dumps(
            #     translated_data, ensure_ascii=False),
            # translated_text_json=translated_data
        )

        db.add(translation_record)
        db.commit()
        db.refresh(translation_record)
    finally:
        db.close()

    # Return file for download
    return {
        "message": "Translation completed successfully",
        # "task_id": task.id,
        "file_path": file_path,
        "translation_id": translation_record.id,
        "translation": translated_data,
    }


@router.put("/shopify/update-string")
async def update_translated_string(req: dict, db: Session = Depends(get_db)):
    """
    Body example:
    {
        "translation_id": 123,
        "shopDomain": "globalflow-ai-esp.myshopify.com",
        "targetLanguage": "fr",
        "path": "fullData.storeData.products.2.title",
        "newValue": "Ceramic Aromatherapy Diffuser",
        "originalValue": "Old Title",
        "expertEdit": true,
        "customerEdit": false
    }
    """

    translation_id = req.get("translation_id")
    shop_domain = req.get("shopDomain")
    lang = req.get("targetLanguage")
    path = req.get("path")
    new_value = req.get("newValue")
    original_value = req.get("originalValue")

    if not all([translation_id, shop_domain, lang, path, new_value]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    # 1. Fetch record by ID
    translation = db.query(Translation).filter_by(id=translation_id).first()
    if not translation:
        raise HTTPException(status_code=404, detail="Translation not found")

    # 2. Verify domain + language match
    if translation.shop_domain != shop_domain or translation.target_lang != lang:
        raise HTTPException(
            status_code=400, detail="Shop or language mismatch")

    # 3. Load existing translated JSON
    data = translation.translated_text_raw
    if isinstance(data, str):
        data = json.loads(data)

    # 4. Walk through JSON path to update value
    keys = path.split(".")
    ref = data
    for k in keys[:-1]:
        ref = ref[int(k)] if k.isdigit() else ref[k]

    # set new value
    ref[keys[-1]] = new_value

    # 5. Save updated JSON + flags
    print("Saving updated JSON into PostGreSQL...")
    translation.translated_text_json = data
    translation.translated_text_raw = json.dumps(
        data, ensure_ascii=False)
    translation.updated_at = datetime.now()

    if "expertEdit" in req:
        translation.expertEdit = req["expertEdit"]
    if "customerEdit" in req:
        translation.customerEdit = req["customerEdit"]

    db.add(translation)
    db.commit()
    db.refresh(translation)

    return {
        "status": "ok",
        "translation_id": translation.id,
        "updatedPath": path,
        "oldValue": original_value,
        "newValue": new_value,
        "shopDomain": translation.shop_domain,
        "targetLanguage": translation.target_lang,
        "updatedJson": translation.translated_text_json
    }


# @router.put("/shopify/update-string")
# async def update_translated_string(req: dict):
#     """
#     Expect body:
#     {
#         "translation_id": 123,
#         "shopDomain": "globalflow-ai-esp.myshopify.com",
#         "targetLanguage": "fr",
#         "path": "fullData.storeData.products.2.title",
#         "newValue": "Ceramic Aromatherapy Diffuser and you're duffer",
#         "originalValue": "",
#         "expertEdit": true,
#         "customerEdit": false
#     }
#     """
#     shop_domain = req.get("shopDomain")
#     lang = req.get("targetLanguage")
#     path = req.get("path")
#     new_value = req.get("newValue")

#     if not all([shop_domain, lang, path, new_value]):
#         raise HTTPException(
#             status_code=400, detail="Missing fields in request")

#     try:

#         # 1. Fetch by ID
#         translation = db.query(Translation).filter_by(
#             id=req["translation_id"]).first()
#         if not translation:
#             raise HTTPException(
#                 status_code=404, detail="Translation not found")

#         # 2. Verify domain + language
#         if translation.shop_domain != req["shopDomain"] or translation.target_lang != req["targetLanguage"]:
#             raise HTTPException(
#                 status_code=400, detail="Shop or language mismatch")

#         # 3. Fetch the latest translation
#         translation = db.query(Translation).filter_by(
#             shop_domain=shop_domain,
#             target_lang=lang
#         ).order_by(Translation.translated_at.desc()).first()

#         if not translation:
#             raise HTTPException(status_code=404, detail="No translation found")

#         data = translation.translated_text  # already JSON/dict

#         # 4. Walk JSON to set new value
#         keys = path.split(".")
#         ref = data
#         for k in keys[:-1]:
#             ref = ref[int(k)] if k.isdigit() else ref[k]
#         ref[keys[-1]] = new_value

#         # 5. Save back to Postgres
#         translation.translated_text = data
#         translation.translated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         # 6. Optional flags
#         if req.get("expertEdit") is not None:
#             translation.expert_edit = req["expertEdit"]
#         if req.get("customerEdit") is not None:
#             translation.customer_edit = req["customerEdit"]

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
