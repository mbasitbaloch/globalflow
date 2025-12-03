from fastapi import HTTPException
from datetime import datetime
from fastapi import APIRouter, Depends, Body, HTTPException
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy.orm import Session
import requests

from app.routes.ingest import get_db
from ..database import SessionLocal
from ..models.models import Translation
from ..models.UpdateRequest import UpdateRequest
from ..services.translator_without_classification_and_vote_old import fast_translate_json
# from ..mongodb import users_collection
# from fastapi.responses import JSONResponse, FileResponse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, PointStruct, VectorParams, Distance
import json
import os
import uuid
# from bson import ObjectId
# from dotenv import load_dotenv
from openai import OpenAI
from ..utils.tasks import store_data
# from langchain.embeddings import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores.qdrant import Qdrant
# from langchain_qdrant import Qdrant
from ..config import settings
from ..mongodb import users_collection
from datetime import datetime
# NEW: Cache imports
# from app.utils.cache_manager import (
#     get_full_translation_from_cache, set_full_translation_in_cache,
#     invalidate_full_translation_cache, compute_raw_hash, invalidate_extracted_data_cache
# )
from ..utils.cache_manager import (
    get_full_translation_from_cache, set_full_translation_in_cache,
    invalidate_full_translation_cache, compute_raw_hash,
    get_cached_extracted_data, cache_extracted_data,
    invalidate_extracted_data_cache
)
import re
from langdetect import detect
import pycountry
import logging
from ..validator.translationValidator import (
    translationValidator,
    validate_json_path,
    SafeJsonParser,
    BRAND_TONES
)


COLLECTION_NAME = settings.COLLECTION_NAME

router = APIRouter()

db: Session = SessionLocal()

# Get user data
# user = users_collection.find_one(
#     {"shopifyStores.shopDomain": req["shopDomain"]})


logger = logging.getLogger(__name__)
router = APIRouter()

# Supported languages + regions (from your countryValidator.py)
LANGUAGES = {
    "en-US": "American English",
    "en-GB": "British English",
    "es-ES": "Spanish (Spain)",
    "es-MX": "Spanish (Mexico)",
    "fr-FR": "French (France)",
    "fr-CA": "French (Canada)",
    "en-CA": "Canadian English (Canada)",
    "de-DE": "German",
    "it-IT": "Italian",
    "pt-PT": "Portuguese (Portugal)",
    "pt-BR": "Portuguese (Brazil)",
    "ru-RU": "Russian",
    "zh-CN": "Chinese (Simplified)",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "tr-TR": "Turkish",
    "nl-NL": "Dutch",
    "sv-SE": "Swedish",
    "pl-PL": "Polish",
    "uk-UA": "Ukrainian",
    "ro-RO": "Romanian",
    "th-TH": "Thai",
    "vi-VN": "Vietnamese",
    "id-ID": "Indonesian",
    "el-GR": "Greek",
    "cs-CZ": "Czech",
    "ur-PK": "Urdu (Pakistan)",
    "ar-SA": "Arabic (Saudi Arabia)",
    "ar-AE": "Arabic (UAE)",
    "ar-EG": "Arabic (Egypt)"
}


# Qdrant client init
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


@router.post("/shopify/translate")
async def shopify_translate(req: dict, db: Session = Depends(get_db)):
    """
    Translate Shopify store data.
    Expect body:
    {
      "shopDomain": "...",
      "accessToken": "...",
      "targetLanguage": "fr",
      "targetcountry": "CA",
      "brandTone": "neutral"
    }
    """
    try:
        # Define allowed fields
        allowed_fields = {"shopDomain", "accessToken",
                          "targetLanguage", "targetcountry", "brandTone"}
        required_fields = ["shopDomain", "accessToken",
                           "targetLanguage", "targetcountry", "brandTone"]

        # Check for unknown fields
        unknown_fields = [f for f in req.keys() if f not in allowed_fields]
        if unknown_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown fields in payload: {', '.join(unknown_fields)}. Allowed fields: {', '.join(allowed_fields)}"
            )

        # Validate required fields
        missing = [f for f in required_fields if f not in req]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

        # Check for empty values
        empty_fields = [f for f in required_fields if not str(req[f]).strip()]
        if empty_fields:
            raise HTTPException(
                status_code=400, detail=f"Empty values found in: {', '.join(empty_fields)}")

        # Validate brand tone (prioritized for early failure)
        brand_tone = req["brandTone"].strip()
        if brand_tone not in BRAND_TONES:
            supported_tones = ", ".join(BRAND_TONES)
            raise HTTPException(
                status_code=400, detail=f"Invalid brand tone '{brand_tone}'. Supported tones: {supported_tones}.")

        # Validate shop domain
        shop_domain = req["shopDomain"].strip()
        domain_pattern = re.compile(r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$")
        if not domain_pattern.match(shop_domain):
            raise HTTPException(
                status_code=400, detail="Invalid shop domain format. Please enter a valid domain like 'example.myshopify.com'.")

        # Validate target language and country
        target_language = req["targetLanguage"]
        target_country = req["targetcountry"]
        valid, msg = translationValidator(
            target_language, target_country)
        if not valid:
            raise HTTPException(status_code=400, detail=msg)

        # Get user from MongoDB
        user = users_collection.find_one(
            {"shopifyStores.shopDomain": shop_domain})
        if not user:
            raise HTTPException(status_code=404, detail="Shop not found")
        industry = user.get("industry", "general")
        user_id = str(user["_id"])

        # Check extracted data cache
        cache_key = f"shopify_data:{shop_domain}:{target_language}:{target_country}"
        cached_data = get_cached_extracted_data(cache_key)
        raw_data = None

        if cached_data:
            url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
            response = requests.post(url, json={
                "shopDomain": shop_domain,
                "accessToken": req["accessToken"],
                "targetLanguage": target_language,
                "brandTone": brand_tone,
                "targetcountry": target_country
            })
            response.raise_for_status()
            fresh_data = response.json()
            fresh_hash = compute_raw_hash(fresh_data)
            cached_hash = compute_raw_hash(cached_data)

            if fresh_hash == cached_hash:
                logger.info(
                    f"Extracted data cache hit for {shop_domain}:{target_language}:{target_country}")
                raw_data = cached_data
            else:
                logger.info(
                    f"Extracted data cache miss (hash mismatch: {cached_hash[:8]} != {fresh_hash[:8]})")
                raw_data = fresh_data
        else:
            logger.info(
                f"No cached extracted data for {shop_domain}:{target_language}:{target_country}")
            url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
            response = requests.post(url, json={
                "shopDomain": shop_domain,
                "accessToken": req["accessToken"],
                "targetLanguage": target_language,
                "brandTone": brand_tone,
                "targetcountry": target_country
            })
            response.raise_for_status()
            raw_data = response.json()

        # Cache the extracted data
        cache_extracted_data(cache_key, raw_data)

        # Check full translation cache
        fresh_hash = compute_raw_hash(raw_data)
        cached_translated = get_full_translation_from_cache(
            shop_domain, target_language, brand_tone, fresh_hash, target_country)
        if cached_translated:
            logger.info(
                f"Translation served from cache for {shop_domain} (hash match)")
            file_name = f"Today_fetched_{uuid.uuid4().hex}.json"
            file_path = os.path.join("fetched_data", file_name)
            os.makedirs("fetched_data", exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)

            translation_record = Translation(
                user_id=user_id,
                industry=industry,
                shop_domain=shop_domain,
                brand_tone=brand_tone,
                target_lang=target_language,
                targetCountry=target_country,
                content_type="json",
                original_text_raw=json.dumps(raw_data, ensure_ascii=False),
                original_text_json=raw_data,
                translated_text_raw=json.dumps(
                    cached_translated, ensure_ascii=False),
                translated_text_json=cached_translated
            )
            db.add(translation_record)
            db.commit()
            db.refresh(translation_record)

            logger.info("Celery task started...")
            task = store_data.delay(
                cached_translated, req, raw_data, translation_record.id)
            logger.info(f"New task ID: {task.id}")

            return {
                "message": "Translation served from cache (data unchanged)",
                "file_path": file_path,
                "translation_id": translation_record.id,
                "translation": cached_translated
            }

        # Cache miss: Run full translation pipeline
        translated_data = await fast_translate_json(
            raw_data,
            user_id=user_id,
            shopDomain=shop_domain,
            target_lang=target_language,
            targetCountry=target_country,
            brand_tone=brand_tone,
            industry=industry
        )

        # Cache the full translated JSON
        set_full_translation_in_cache(
            shop_domain, target_language, brand_tone, translated_data, fresh_hash, target_country)

        # Save original JSON to file
        file_name = f"Today_fetched_{uuid.uuid4().hex}.json"
        file_path = os.path.join("fetched_data", file_name)
        os.makedirs("fetched_data", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        # Save translated JSON to file
        logger.info("Saving translated JSON to file...")
        file_name = f"Today_translated_{uuid.uuid4().hex}.json"
        file_path = os.path.join("tmp", file_name)
        os.makedirs("tmp", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Translated JSON saved to file: {file_path}")

        # Save to PostgreSQL
        translation_record = Translation(
            user_id=user_id,
            industry=industry,
            shop_domain=shop_domain,
            brand_tone=brand_tone,
            target_lang=target_language,
            targetCountry=target_country,
            content_type="json",
            original_text_raw=json.dumps(raw_data, ensure_ascii=False),
            original_text_json=raw_data,
            translated_text_raw=json.dumps(
                translated_data, ensure_ascii=False),
            translated_text_json=translated_data
        )
        db.add(translation_record)
        db.commit()
        db.refresh(translation_record)

        logger.info("Celery task started...")
        task = store_data.delay(translated_data, req,
                                raw_data, translation_record.id)
        logger.info(f"New task ID: {task.id}")

        return {
            "message": "Translation completed successfully",
            "file_path": file_path,
            "translation_id": translation_record.id,
            "translation": translated_data
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in shopify/translate: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}")


# version of translate endpoint with improved code at 15-11-2025 at 12:40 PM
# @router.post("/shopify/translate")
# async def shopify_translate(req: dict, db: Session = Depends(get_db)):
#     """
#     Translate Shopify store data or specific text.
#     Expect body:
#     {
#       "shopDomain": "...",
#       "accessToken": "...",
#       "targetLanguage": "fr",
#       "targetcountry": "FR",
#       "brandTone": "neutral",
#       "text": "..."  # Optional
#     }
#     """
#     try:
#         # Validate required fields
#         allowed_fields = {"shopDomain", "accessToken",
#                           "targetLanguage", "targetcountry", "brandTone"}
#         required_fields = ["shopDomain", "accessToken",
#                            "targetLanguage", "brandTone", "targetcountry"]

#         # Check for unknown fields
#         unknown_fields = [f for f in req.keys() if f not in allowed_fields]
#         if unknown_fields:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Unknown fields in payload: {', '.join(unknown_fields)}. Allowed fields: {', '.join(allowed_fields)}"
#             )

#         missing = [f for f in required_fields if f not in req]
#         if missing:
#             raise HTTPException(
#                 status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

#         # Check for empty values
#         empty_fields = [f for f in required_fields if not str(req[f]).strip()]
#         if empty_fields:
#             raise HTTPException(
#                 status_code=400, detail=f"Empty values found in: {', '.join(empty_fields)}")

#         # Validate shop domain
#         shop_domain = req["shopDomain"].strip()
#         domain_pattern = re.compile(r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$")
#         if not domain_pattern.match(shop_domain):
#             raise HTTPException(
#                 status_code=400, detail="Invalid shop domain format. Please enter a valid domain like 'example.myshopify.com'.")

#         # Validate target language and country
#         target_language = req["targetLanguage"]
#         target_country = req["targetcountry"]
#         text = req.get("text")
#         valid, msg = validate_language_and_country(
#             target_language, target_country, text)
#         if not valid:
#             raise HTTPException(status_code=400, detail=msg)

#         # Validate brand tone
#         brand_tone = req["brandTone"].strip()
#         # if brand_tone not in BRAND_TONES:
#         #     supported_tones = ", ".join(BRAND_TONES)
#         #     raise HTTPException(
#         #         status_code=400, detail=f"Invalid brand tone '{brand_tone}'. Supported tones: {supported_tones}.")

#         # Get user from MongoDB
#         user = users_collection.find_one(
#             {"shopifyStores.shopDomain": shop_domain})
#         if not user:
#             raise HTTPException(status_code=404, detail="Shop not found")
#         industry = user.get("industry", "general")
#         user_id = str(user["_id"])

#         # Check extracted data cache
#         cache_key = f"shopify_data:{shop_domain}:{target_language}:{target_country}"
#         cached_data = get_cached_extracted_data(cache_key)
#         raw_data = None

#         if cached_data:
#             url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
#             response = requests.post(url, json={
#                 "shopDomain": shop_domain,
#                 "accessToken": req["accessToken"],
#                 "targetLanguage": target_language,
#                 "brandTone": brand_tone,
#                 "targetcountry": target_country
#             })
#             response.raise_for_status()
#             fresh_data = response.json()
#             fresh_hash = compute_raw_hash(fresh_data)
#             cached_hash = compute_raw_hash(cached_data)

#             if fresh_hash == cached_hash:
#                 logger.info(
#                     f"Extracted data cache hit for {shop_domain}:{target_language}:{target_country}")
#                 raw_data = cached_data
#             else:
#                 logger.info(
#                     f"Extracted data cache miss (hash mismatch: {cached_hash[:8]} != {fresh_hash[:8]})")
#                 raw_data = fresh_data
#         else:
#             logger.info(
#                 f"No cached extracted data for {shop_domain}:{target_language}:{target_country}")
#             url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
#             response = requests.post(url, json={
#                 "shopDomain": shop_domain,
#                 "accessToken": req["accessToken"],
#                 "targetLanguage": target_language,
#                 "brandTone": brand_tone,
#                 "targetcountry": target_country
#             })
#             response.raise_for_status()
#             raw_data = response.json()

#         # Cache the extracted data
#         cache_extracted_data(cache_key, raw_data)

#         # Check full translation cache
#         fresh_hash = compute_raw_hash(raw_data)
#         cached_translated = get_full_translation_from_cache(
#             shop_domain, target_language, brand_tone, fresh_hash, target_country)
#         if cached_translated:
#             logger.info(
#                 f"Translation served from cache for {shop_domain} (hash match)")
#             file_name = f"Today_fetched_{uuid.uuid4().hex}.json"
#             file_path = os.path.join("fetched_data", file_name)
#             os.makedirs("fetched_data", exist_ok=True)
#             with open(file_path, "w", encoding="utf-8") as f:
#                 json.dump(raw_data, f, ensure_ascii=False, indent=2)

#             translation_record = Translation(
#                 user_id=user_id,
#                 industry=industry,
#                 shop_domain=shop_domain,
#                 brand_tone=brand_tone,
#                 target_lang=target_language,
#                 targetCountry=target_country,
#                 content_type="json",
#                 original_text_raw=json.dumps(raw_data, ensure_ascii=False),
#                 original_text_json=raw_data,
#                 translated_text_raw=json.dumps(
#                     cached_translated, ensure_ascii=False),
#                 translated_text_json=cached_translated
#             )
#             db.add(translation_record)
#             db.commit()
#             db.refresh(translation_record)

#             logger.info("Celery task started...")
#             task = store_data.delay(
#                 cached_translated, req, raw_data, translation_record.id)
#             logger.info(f"New task ID: {task.id}")

#             return {
#                 "message": "Translation served from cache (data unchanged)",
#                 "file_path": file_path,
#                 "translation_id": translation_record.id,
#                 "translation": cached_translated
#             }

#         # Cache miss: Run full translation pipeline
#         translated_data = await fast_translate_json(
#             raw_data,
#             user_id=user_id,
#             shopDomain=shop_domain,
#             target_lang=target_language,
#             targetCountry=target_country,
#             brand_tone=brand_tone,
#             industry=industry,
#             text=text
#         )

#         # Cache the full translated JSON
#         set_full_translation_in_cache(
#             shop_domain, target_language, brand_tone, translated_data, fresh_hash, target_country)

#         # Save original JSON to file
#         file_name = f"Today_fetched_{uuid.uuid4().hex}.json"
#         file_path = os.path.join("fetched_data", file_name)
#         os.makedirs("fetched_data", exist_ok=True)
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(raw_data, f, ensure_ascii=False, indent=2)

#         # Save translated JSON to file
#         logger.info("Saving translated JSON to file...")
#         file_name = f"Today_translated_{uuid.uuid4().hex}.json"
#         file_path = os.path.join("tmp", file_name)
#         os.makedirs("tmp", exist_ok=True)
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(translated_data, f, ensure_ascii=False, indent=2)
#         logger.info(f"Translated JSON saved to file: {file_path}")

#         # Save to PostgreSQL
#         translation_record = Translation(
#             user_id=user_id,
#             industry=industry,
#             shop_domain=shop_domain,
#             brand_tone=brand_tone,
#             target_lang=target_language,
#             targetCountry=target_country,
#             content_type="json",
#             original_text_raw=json.dumps(raw_data, ensure_ascii=False),
#             original_text_json=raw_data,
#             translated_text_raw=json.dumps(
#                 translated_data, ensure_ascii=False),
#             translated_text_json=translated_data
#         )
#         db.add(translation_record)
#         db.commit()
#         db.refresh(translation_record)

#         logger.info("Celery task started...")
#         task = store_data.delay(translated_data, req,
#                                 raw_data, translation_record.id)
#         logger.info(f"New task ID: {task.id}")

#         return {
#             "message": "Translation completed successfully",
#             "file_path": file_path,
#             "translation_id": translation_record.id,
#             "translation": translated_data
#         }

#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         logger.error(f"Unexpected error in shopify/translate: {str(e)}")
#         raise HTTPException(
#             status_code=500, detail=f"Internal server error: {str(e)}")


# @router.post("/shopify/translate")
# async def shopify_translate(req: dict, db: Session = Depends(get_db)):
#     """
#     Expect body:
#     {
#       "shopDomain": "...",
#       "accessToken": "...",
#       "targetLanguage": "fr",
#       "targetcountry":"FR",
#       "brandTone": "neutral"
#     }
#     """
#     # Validate request
#     required_fields = ["shopDomain", "accessToken",
#                        "targetLanguage", "brandTone", "targetcountry"]

#     # Check all required fields exist
#     missing = [f for f in required_fields if f not in req]
#     if missing:
#         raise HTTPException(
#             status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

#     # Check none are empty or null
#     empty_fields = [f for f in required_fields if not str(req[f]).strip()]
#     if empty_fields:
#         raise HTTPException(
#             status_code=400, detail=f"Empty values found in: {', '.join(empty_fields)}, please enter an entry for {', '.join(empty_fields)}")

#     # Validate shop domain properly
#     shop_domain = req["shopDomain"].strip()
#     domain_pattern = re.compile(
#         # e.g., something.com or abc.myshopify.com
#         r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$"
#     )
#     if not domain_pattern.match(shop_domain):
#         raise HTTPException(
#             status_code=400, detail="Invalid shop domain format. Please enter a valid domain like 'example.myshopify.com'.")

#     # Get user from MongoDB
#     shop_domain = req["shopDomain"]
#     targetLanguage = req["targetLanguage"]
#     targetCountry = req["targetcountry"]
#     brand_tone = req["brandTone"]

#     user = users_collection.find_one({"shopifyStores.shopDomain": shop_domain})
#     if not user:
#         raise HTTPException(status_code=404, detail="Shop not found")
#     industry = user.get("industry", "general")
#     user_id = str(user["_id"])

#     # Check extracted data cache
#     cache_key = f"shopify_data:{shop_domain}:{targetLanguage}:{targetCountry}"
#     cached_data = get_cached_extracted_data(cache_key)
#     raw_data = None

#     if cached_data:
#         # Fetch fresh data to compare
#         url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
#         response = requests.post(url, json={
#             "shopDomain": shop_domain,
#             "accessToken": req["accessToken"],
#             "targetLanguage": targetLanguage,
#             "brandTone": brand_tone,
#             "targetcountry": targetCountry
#         })
#         response.raise_for_status()
#         fresh_data = response.json()
#         fresh_hash = compute_raw_hash(fresh_data)
#         cached_hash = compute_raw_hash(cached_data)

#         if fresh_hash == cached_hash:
#             print(
#                 f"Extracted data cache hit for {shop_domain}:{targetLanguage}:{targetCountry}")
#             raw_data = cached_data
#         else:
#             print(
#                 f"Extracted data cache miss (hash mismatch: {cached_hash[:8]} != {fresh_hash[:8]})")
#             raw_data = fresh_data
#     else:
#         print(
#             f"No cached extracted data for {shop_domain}:{targetLanguage}:{targetCountry}")
#         # Fetch fresh data from Shopify
#         url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
#         response = requests.post(url, json={
#             "shopDomain": shop_domain,
#             "accessToken": req["accessToken"],
#             "targetLanguage": targetLanguage,
#             "brandTone": brand_tone,
#             "targetcountry": targetCountry
#         })
#         response.raise_for_status()
#         raw_data = response.json()

#     # Cache the extracted data
#     cache_extracted_data(cache_key, raw_data)

#     # Check full translation cache
#     fresh_hash = compute_raw_hash(raw_data)
#     cached_translated = get_full_translation_from_cache(
#         shop_domain, targetLanguage, brand_tone, fresh_hash, targetCountry)
#     if cached_translated:
#         print(f"Translation served from cache for {shop_domain} (hash match)")
#         # Save original JSON
#         file_name = f"Today_fetched_{uuid.uuid4().hex}.json"
#         file_path = os.path.join("fetched_data", file_name)
#         os.makedirs("fetched_data", exist_ok=True)
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(raw_data, f, ensure_ascii=False, indent=2)

#         # Save to PostgreSQL
#         translation_record = Translation(
#             user_id=user_id,
#             industry=industry,
#             shop_domain=shop_domain,
#             brand_tone=brand_tone,
#             target_lang=targetLanguage,
#             targetCountry=targetCountry,
#             content_type="json",
#             original_text_raw=json.dumps(raw_data, ensure_ascii=False),
#             original_text_json=raw_data,
#             translated_text_raw=json.dumps(
#                 cached_translated, ensure_ascii=False),
#             translated_text_json=cached_translated
#         )
#         db.add(translation_record)
#         db.commit()
#         db.refresh(translation_record)

#         print("Celery task started...")
#         task = store_data.delay(cached_translated, req,
#                                 raw_data, translation_record.id)
#         print(f"New task ID: {task.id}")

#         return {
#             "message": "Translation served from cache (data unchanged)",
#             "file_path": file_path,
#             "translation_id": translation_record.id,
#             "translation": cached_translated
#         }

#     # Cache miss: Run full translation pipeline
#     translated_data = await fast_translate_json(
#         raw_data,
#         user_id=user_id,
#         shopDomain=shop_domain,
#         target_lang=targetLanguage,
#         targetCountry=targetCountry,
#         brand_tone=brand_tone,
#         industry=industry
#     )

#     # Cache the full translated JSON
#     set_full_translation_in_cache(
#         shop_domain, targetLanguage, brand_tone, translated_data, fresh_hash, targetCountry)

#     # Save original JSON to file
#     file_name = f"Today_fetched_{uuid.uuid4().hex}.json"
#     file_path = os.path.join("fetched_data", file_name)
#     os.makedirs("fetched_data", exist_ok=True)
#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(raw_data, f, ensure_ascii=False, indent=2)

#     # Save translated JSON to file
#     print("Saving translated JSON to file...")
#     file_name = f"Today_translated_{uuid.uuid4().hex}.json"
#     file_path = os.path.join("tmp", file_name)
#     os.makedirs("tmp", exist_ok=True)
#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(translated_data, f, ensure_ascii=False, indent=2)
#     print("Translated JSON saved to file:", file_path)

#     # Save to PostgreSQL
#     translation_record = Translation(
#         user_id=user_id,
#         industry=industry,
#         shop_domain=shop_domain,
#         brand_tone=brand_tone,
#         target_lang=targetLanguage,
#         targetCountry=targetCountry,
#         content_type="json",
#         original_text_raw=json.dumps(raw_data, ensure_ascii=False),
#         original_text_json=raw_data,
#         translated_text_raw=json.dumps(translated_data, ensure_ascii=False),
#         translated_text_json=translated_data
#     )
#     db.add(translation_record)
#     db.commit()
#     db.refresh(translation_record)

#     print("Celery task started...")
#     task = store_data.delay(translated_data, req,
#                             raw_data, translation_record.id)
#     print(f"New task ID: {task.id}")

#     # Return file for download
#     return {
#         "message": "Translation completed successfully",
#         "file_path": file_path,
#         "translation_id": translation_record.id,
#         "translation": translated_data
#     }


def validate_language_and_country(target_language: str, target_country: str, text: str) -> tuple[bool, str]:
    """
    Validate if the target language, country, and text are compatible.
    """
    # Normalize country code
    country = target_country.upper().replace("FRANCE", "FR").split("-")[0]
    supported_regions = {lang.split("-")[1]
                         for lang in LANGUAGES.keys() if "-" in lang}
    print("Supported regions:", supported_regions)
    if country not in supported_regions:
        return False, f"Country '{target_country}' is not supported. Supported regions: {', '.join(sorted(supported_regions))}."

    # Validate target language
    supported_langs = {lang.split("-")[0] for lang in LANGUAGES.keys()}
    print("supported languages:", supported_langs)
    if target_language not in supported_langs:
        return False, f"Target language '{target_language}' is not supported. Supported languages: {', '.join(sorted(supported_langs))}."

    # Check if target language is valid for the country
    valid_lang_region = False
    for lang_region in LANGUAGES.keys():
        lang, region = lang_region.split("-")
        if lang == target_language and region == country:
            valid_lang_region = True
            break
    if not valid_lang_region:
        supported_combinations = [
            k for k in LANGUAGES.keys() if k.startswith(target_language + "-")]
        combinations = ", ".join(
            supported_combinations) if supported_combinations else "none"
        return False, f"Target language '{target_language}' is not valid for country '{target_country}'. Supported combinations: {combinations}."

    # Detect language of the text
    try:
        detected_lang = detect(text)
    except Exception as e:
        return False, f"Language detection failed: {str(e)}"

    # Check if text matches target language
    if detected_lang != target_language:
        target_name = pycountry.languages.get(
            alpha_2=target_language).name if target_language in pycountry.languages else target_language
        detected_name = pycountry.languages.get(
            alpha_2=detected_lang).name if detected_lang in pycountry.languages else detected_lang
        return False, f"The provided text is in {detected_name}, but the target language is {target_name}."

    return True, "OK"


def validate_json_path(data: dict, path: str) -> tuple[bool, str]:
    """
    Validate if the JSON path is valid and exists in the data.
    """
    try:
        keys = path.split(".")
        ref = data
        for k in keys[:-1]:
            if k.isdigit():
                ref = ref[int(k)]
            else:
                ref = ref[k]
        last_key = keys[-1]
        if last_key.isdigit():
            if not isinstance(ref, list) or int(last_key) >= len(ref):
                return False, f"Invalid path: Index '{last_key}' out of range or not a list."
        else:
            if not isinstance(ref, dict) or last_key not in ref:
                return False, f"Invalid path: Key '{last_key}' does not exist."
        return True, "OK"
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return False, f"Invalid path: {str(e)}"


class SafeJsonParser:
    def parse(self, text: str):
        # Step 1️⃣ — Trim whitespace and markdown fences
        text = text.strip()
        text = re.sub(r"^```(?:json|json5|javascript)?\s*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

        # Step 2️⃣ — Remove common prefixes
        text = re.sub(r'^[\s`]*[Oo]utput\s*[:\-]*\s*', '', text)
        text = re.sub(r'^[\s`]*[Rr]esponse\s*[:\-]*\s*', '', text)
        text = text.strip()

        # Step 3️⃣ — Extract JSON-like content
        match = re.search(r'(\[.*|\{.*)', text, re.S)
        if not match:
            raise ValueError(
                "❌ No valid JSON object or array found in response")
        text = match.group(1).strip()

        # Step 4️⃣ — Sanitize common issues
        text = (
            text.replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )
        text = re.sub(r',(\s*[\]}])', r'\1', text)
        text = re.sub(r'\]\s*\[', '], [', text)

        # Step 5️⃣ — Fix single quotes for JSON keys (e.g., 'rating' -> "rating")
        # Use regex to replace single quotes around keys, but not in values
        text = re.sub(r"(\{|\s|,)'\s*([^'{\[\]:,]+)\s*'\s*:", r'\1"\2":', text)
        # Fix single quotes in values that might have been escaped (e.g., L\'update)
        text = text.replace("\\'", "'")

        # Step 6️⃣ — Detect & fix missing closing bracket
        open_brackets = text.count("[")
        close_brackets = text.count("]")
        if open_brackets > close_brackets:
            missing = open_brackets - close_brackets
            text += "]" * missing
        open_braces = text.count("{")
        close_braces = text.count("}")
        if open_braces > close_braces:
            missing = open_braces - close_braces
            text += "}" * missing

        # Step 7️⃣ — Attempt to parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Final fallback: fix fancy quotes and try again
            text = text.replace("’", "'").replace("“", '"').replace("”", '"')
            try:
                return json.loads(text)
            except Exception as e2:
                raise ValueError(
                    f"❌ Failed to parse sanitized JSON: {e2}\nRaw: {text[:500]}")


@router.put("/shopify/update-string")
async def update_translated_string(req: UpdateRequest, db: Session = Depends(get_db)):
    """
    Safely updates a translation with AI validation & Qdrant embedding.
    """
    try:
        # Validate required fields
        required_fields = ["translation_id", "shopDomain",
                           "targetLanguage", "targetcountry", "path", "newValue"]
        missing = [f for f in required_fields if getattr(
            req, f, None) is None or not str(getattr(req, f)).strip()]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Missing or empty required field: {', '.join(missing)}")

        # Check if newValue equals originalValue
        if req.newValue == req.originalValue:
            logger.info("No changes provided: newValue equals originalValue")
            ai_rating = 1.0
            ai_reason = "No changes made; original value retained."

        else:
            # Validate language and country
            valid, msg = validate_language_and_country(
                req.targetLanguage, req.targetcountry, req.newValue)
            if not valid:
                raise HTTPException(status_code=400, detail=msg)

        # Validate language and country
        # valid, msg = validate_language_and_country(
        #     req.targetLanguage, req.targetcountry, req.newValue)
        # if not valid:
        #     raise HTTPException(status_code=400, detail=msg)

        # Fetch translation record
        translation = db.query(Translation).filter_by(
            id=req.translation_id).first()
        if not translation:
            raise HTTPException(
                status_code=404, detail="Translation not found")

        # Verify domain and language consistency
        if translation.shop_domain != req.shopDomain:
            raise HTTPException(
                status_code=400,
                detail=f"Shop domain mismatch: Request domain '{req.shopDomain}' does not match stored domain '{translation.shop_domain}'."
            )
        if translation.target_lang != req.targetLanguage:
            raise HTTPException(
                status_code=400,
                detail=f"Target language mismatch: The translation record was created for '{translation.target_lang}', but you are trying to update using '{req.targetLanguage}'."
            )
        if translation.targetCountry and translation.targetCountry.lower() != req.targetcountry.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Target country mismatch: Existing translation country is '{translation.targetCountry}', but you tried to update using '{req.targetcountry}'."
            )

        #         # --- Get user ---
        user = users_collection.find_one(
            {"shopifyStores.shopDomain": req.shopDomain})
        if not user:
            raise HTTPException(status_code=404, detail="Shop not found")

        # Validate JSON path
        data = translation.translated_text_raw
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid JSON in translation record.")
        valid_path, path_msg = validate_json_path(data, req.path)
        if not valid_path:
            raise HTTPException(status_code=400, detail=path_msg)

        # Initialize OpenAI client
        # try:
        #     client = OpenAI(api_key=settings.OPENAI_API_KEY_1)
        # except Exception as e:
        #     logger.error(f"OpenAI client error: {e}")
        #     raise HTTPException(
        #         status_code=500, detail=f"OpenAI initialization failed: {e}")

        # # AI validation
        # ai_rating, ai_reason = 0, "AI validation failed"
        # try:
        #     logger.info("Validating update with AI...")
        #     ai_validation = client.chat.completions.create(
        #         model="gpt-4o-mini",
        #         messages=[
        #             {
        #                 "role": "system",
        #                 "content": (
        #                     f"You are an AI evaluator responsible for rating text updates based on their quality, meaning, and contextual relevance. "
        #                     f"The update must be in the target language ({req.targetLanguage}) and appropriate for the target country ({req.targetcountry}). "
        #                     f"Ensure the text uses the natural tone, expressions, and spelling conventions of {req.targetLanguage} in {req.targetcountry}. "
        #                     f"For example, for French in France, use standard French spellings and avoid regional variants not common in France. "
        #                     f"Reject updates that are in the wrong language, unnatural, off-tone, or contextually irrelevant. "
        #                     f"Return your response in JSON format: {{ 'rating': <float between 0 and 1>, 'reason': '<explanation>' }}."
        #                 ),
        #             },
        #             {
        #                 "role": "user",
        #                 "content": f"Original: {req.originalValue}\nUpdated: {req.newValue}"
        #             },
        #         ],
        #         temperature=0.3,
        #     )
        #     ai_response = ai_validation.choices[0].message.content
        #     logger.info(f"AI Response: {ai_response}")

        #     try:
        #         rating_data = json.loads(ai_response)
        #         ai_rating = float(rating_data.get("rating", 0))
        #         ai_reason = rating_data.get("reason", "")
        #     except Exception:
        #         ai_rating = 0
        #         ai_reason = "Invalid AI response format"
        # except Exception as e:
        #     logger.error(f"AI validation error: {e}")
        #     ai_rating = 0
        #     ai_reason = "AI validation request failed"

        # if ai_rating < 0.6:

        #     raise HTTPException(
        #         status_code=400,
        #         detail=f"Update rejected by AI (rating={ai_rating:.2f}): {ai_reason}"
        #     )

        # AI validation (skip if no changes)
        parser = SafeJsonParser()
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY_1)
        except Exception as e:
            logger.error(f"OpenAI client error: {e}")
            raise HTTPException(
                status_code=500, detail=f"OpenAI initialization failed: {e}")
        if req.newValue != req.originalValue:
            try:

                logger.info("Validating update with AI...")
                ai_validation = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"You are an AI evaluator responsible for rating text updates based on their quality, meaning, and contextual relevance. "
                                f"The update must be in the target language ({req.targetLanguage}) and appropriate for the target country ({req.targetcountry}). "
                                f"Ensure the text uses the natural tone, expressions, and spelling conventions of {req.targetLanguage} in {req.targetcountry}. "
                                f"For example, for French in France, use standard French spellings and avoid regional variants not common in France. "
                                f"Reject updates that are in the wrong language, unnatural, off-tone, or contextually irrelevant. "
                                f"If the update is identical to the original, assign a rating of 1.0 with reason 'No changes made'. "
                                f"Return your response in JSON format with double-quoted keys and values: {{ \"rating\": <float between 0 and 1>, \"reason\": \"<explanation>\" }}."
                            ),
                        },

                        {
                            "role": "user",
                            "content": f"Original: {req.originalValue}\nUpdated: {req.newValue}"
                        },
                    ],
                    temperature=0.3,
                )
                ai_response = ai_validation.choices[0].message.content
                logger.info(f"AI Response: {ai_response}")
                try:
                    rating_data = parser.parse(ai_response)
                    ai_rating = float(rating_data.get("rating", 0))
                    ai_reason = rating_data.get(
                        "reason", "Invalid AI response format")
                    if not isinstance(ai_rating, (int, float)) or not isinstance(ai_reason, str):
                        raise ValueError("Invalid AI response structure")
                except ValueError as e:
                    logger.error(
                        f"AI response parsing error: {e}, raw response: {ai_response}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"AI validation failed: Invalid response format from AI: {str(e)}"
                    )
            except Exception as e:
                logger.error(f"AI validation error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"AI validation request failed: {str(e)}"
                )
            if ai_rating < 0.6:
                raise HTTPException(
                    status_code=400,
                    detail=f"Update rejected by AI (rating={ai_rating:.2f}): {ai_reason}"
                )

        logger.info(f"AI approved (rating={ai_rating:.2f})")

        # Apply JSON update
        keys = req.path.split(".")
        ref = data
        for k in keys[:-1]:
            ref = ref[int(k)] if k.isdigit() else ref[k]
        last_key = keys[-1]
        ref[last_key] = req.newValue

        # Handle priority and AI flags
        priority_key = f"priorityReview_{last_key}"
        ai_flag_key = f"aiTranslated_{last_key}"
        if req.expertEdit:
            if priority_key in ref:
                ref[priority_key] = False
            if ai_flag_key in ref:
                ref[ai_flag_key] = False

        # Update translation record
        translation.translated_text_json = data
        translation.translated_text_raw = json.dumps(data, ensure_ascii=False)
        translation.updated_at = datetime.now()
        translation.targetCountry = req.targetcountry
        translation.expert_edit = req.expertEdit
        translation.customer_edit = req.customerEdit
        translation.transAccept = req.transAccept
        translation.transEdit = req.transEdit

        db.add(translation)
        db.commit()
        db.refresh(translation)
        logger.info("PostgreSQL record updated successfully.")

        # Invalidate caches
        try:
            invalidate_full_translation_cache(
                req.shopDomain, req.targetLanguage, translation.brand_tone, req.targetcountry)
            invalidate_extracted_data_cache(
                req.shopDomain, req.targetLanguage, req.targetcountry)
            logger.info(
                f"Caches invalidated for shopDomain: {req.shopDomain}, targetLanguage: {req.targetLanguage}:{req.targetcountry}")
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")

        # Qdrant embedding
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=req.newValue,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            correction_point = PointStruct(
                id=str(uuid.uuid4()),
                vector={"GlobalFlow": embedding},
                payload={
                    "data_type": "correction",
                    "user_id": str(user["_id"]) if user else None,
                    "industry": (user or {}).get("industry") or "Unknown",
                    "postgres_id": translation.id,
                    "shopDomain": req.shopDomain,
                    "targetLanguage": req.targetLanguage,
                    "targetCountry": req.targetcountry,
                    "path": req.path,
                    "newValue": req.newValue,
                    "originalValue": req.originalValue,
                    "expertEdit": req.expertEdit,
                    "customerEdit": req.customerEdit,
                    "transAccept": req.transAccept,
                    "transEdit": req.transEdit,
                    "ai_rating": ai_rating,
                    "ai_reason": ai_reason,
                    "date": datetime.utcnow().isoformat(),
                }
            )
            qdrant.upsert(collection_name=COLLECTION_NAME,
                          points=[correction_point])
            logger.info("Qdrant embedding stored successfully.")
        except Exception as e:
            logger.error(f"Qdrant embedding failed: {e}")

        return {
            "status": "success",
            "translation_id": translation.id,
            "updatedPath": req.path,
            "oldValue": req.originalValue,
            "newValue": req.newValue,
            "shopDomain": translation.shop_domain,
            "targetLanguage": translation.target_lang,
            "targetCountry": translation.targetCountry,
            "ai_rating": ai_rating,
            "ai_reason": ai_reason,
            "updatedJson": translation.translated_text_json,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected API error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")


# update API backup on 14-11-2025 at 12:39 PM

# @router.put("/shopify/update-string")
# async def update_translated_string(req: UpdateRequest, db: Session = Depends(get_db)):
#     """
#     Safely updates a translation with AI validation & Qdrant embedding.
#     Prevents crashes if AI or Qdrant fails.
#     """
#     try:
#         # --- Validate request ---
#         # required_fields = ["translation_id", "shopDomain",
#         #                    "targetLanguage", "path", "newValue", "targetcountry"]
#         # # if not all(k in req for k in required_fields):
#         # #     return {"status": "error", "message": "Missing required fields"}

#         # missing = [f for f in required_fields if f not in req or req[f] is None]
#         # if missing:
#         #     raise HTTPException(
#         #         status_code=400,
#         #         detail=f"Missing required field: {', '.join(missing)}"
#         #     )

#         translation_id = req.translation_id
#         shop_domain = req.shopDomain
#         targetLanguage = req.targetLanguage
#         targetCountry = req.targetcountry
#         path = req.path
#         new_value = req.newValue
#         original_value = req.originalValue or ""
#         expertEdit = req.expertEdit
#         customerEdit = req.customerEdit
#         transAccept = req.transAccept
#         transEdit = req.transEdit

#         # --- Fetch translation record ---
#         translation = db.query(Translation).filter_by(
#             id=translation_id).first()
#         if not translation:
#             return {"status": "error", "message": "Translation not found"}

#         # if translation.shop_domain != shop_domain or translation.target_lang != targetLanguage:
#         #     return {"status": "error", "message": "Shop or language mismatch"}

#             # --- Verify domain, language, and country consistency ---
#         if translation.shop_domain != shop_domain:
#             return {
#                 "status": "error",
#                 "message": (
#                     f"Shop domain mismatch: Request domain '{shop_domain}' "
#                     f"does not match stored domain '{translation.shop_domain}'."
#                 )
#             }

#         # If request targetLanguage doesn’t match stored target_lang
#         if translation.target_lang != targetLanguage:
#             return {
#                 "status": "error",
#                 "message": (
#                     f"Target language mismatch: The translation record was created for "
#                     f"'{translation.target_lang}', but you are trying to update using '{targetLanguage}'. "
#                     "Please use the same language as the existing translation."
#                 )
#             }

#         # Handle country logic smartly:
#         if translation.targetCountry:
#             # If record already has a country, enforce consistency
#             if translation.targetCountry.lower() != targetCountry.lower():
#                 return {
#                     "status": "error",
#                     "message": (
#                         f"Target country mismatch: Existing translation country is '{translation.targetCountry}', "
#                         f"but you tried to update using '{targetCountry}'."
#                     )
#                 }
#         else:
#             # If DB has no country, safely assign it
#             translation.targetCountry = targetCountry
#             # db.add(translation)
#             # db.commit()
#             print(
#                 f"Target country '{targetCountry}' saved for translation {translation.id}.")

#         # --- Get user ---
#         user = users_collection.find_one(
#             {"shopifyStores.shopDomain": shop_domain})
#         if not user:
#             return {"status": "error", "message": "Shop not found"}

#         # --- Initialize OpenAI client ---
#         try:
#             client = OpenAI(api_key=settings.OPENAI_API_KEY_1)
#         except Exception as e:
#             print(f"OpenAI client error: {e}")
#             return {"status": "error", "message": f"OpenAI init failed: {e}"}

#         # --- AI validation ---
#         ai_rating, ai_reason = 0, "AI validation failed"
#         try:
#             print("Validating update with AI...")
#             ai_validation = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": (
#                             f"You are an AI evaluator responsible for rating text updates based on their quality, meaning, and contextual relevance. "
#                                 f"Only approve updates that are meaningful, relevant, and linguistically appropriate to the user's target language and region. "
#                                 f"Evaluate the text using the natural tone, expressions, and writing style used in the specified country for that language. "
#                                 f"For example, if the language is English and the country is the United Kingdom, prefer 'colour' over 'color', and if the country is the United States, prefer 'color' over 'colour'. "
#                                 f"Apply equivalent tone and spelling distinctions for other languages and regions. "
#                                 f"Be strict — reject updates that are incorrect, unnatural, off-tone, or contextually irrelevant. "
#                                 f"Target Language: {targetLanguage} | Target Country/Region: {targetCountry}. "
#                                 f"Return your response strictly in JSON format with the following structure: "
#                                 "{ 'rating': <float between 0 and 1>, 'reason': '<clear explanation of your decision>' }."
#                         ),
#                     },
#                     {
#                         "role": "user",
#                         "content": f"Original: {original_value}\nUpdated: {new_value}"
#                     },
#                 ],
#                 temperature=0.3,
#             )

#             ai_response = ai_validation.choices[0].message.content
#             print("AI Response:", ai_response)

#             try:
#                 rating_data = json.loads(ai_response)  # type: ignore
#                 ai_rating = float(rating_data.get("rating", 0))
#                 ai_reason = rating_data.get("reason", "")
#             except Exception:
#                 ai_rating = 0
#                 ai_reason = "Invalid AI response format"

#         except Exception as e:
#             print(f"AI validation error: {e}")
#             ai_rating = 0
#             ai_reason = "AI validation request failed"

#         # --- If AI rejects, return safely ---
#         if ai_rating < 0.6:
#             return {
#                 "status": "rejected",
#                 "ai_rating": ai_rating,
#                 "ai_reason": ai_reason,
#                 "message": f"Update rejected by AI (rating={ai_rating:.2f})"
#             }

#         print(f"AI approved (rating={ai_rating:.2f})")

#         # --- Fetch translation record ---
#         # translation = db.query(Translation).filter_by(
#         #     id=translation_id).first()
#         # if not translation:
#         #     return {"status": "error", "message": "Translation not found"}

#         # if translation.shop_domain != shop_domain or translation.target_lang != targetLanguage:
#         #     return {"status": "error", "message": "Shop or language mismatch"}

#         # --- Apply JSON update ---
#         data = translation.translated_text_raw
#         if isinstance(data, str):
#             data = json.loads(data)

#         keys = path.split(".")
#         ref = data
#         for k in keys[:-1]:
#             ref = ref[int(k)] if k.isdigit() else ref[k]

#         # --- Apply the new value ---
#         last_key = keys[-1]
#         ref[last_key] = new_value  # type: ignore
#         # ref[keys[-1]] = new_value

#         # --- Handle priority + aiTranslated flags dynamically ---
#         priority_key = f"priorityReview_{last_key}"
#         ai_flag_key = f"aiTranslated_{last_key}"

#         # If expert edited, clear review + AI flags
#         if req.expertEdit:
#             if priority_key in ref:
#                 ref[priority_key] = False  # type: ignore
#             if ai_flag_key in ref:
#                 ref[ai_flag_key] = False  # type: ignore

#         translation.translated_text_json = data
#         translation.translated_text_raw = json.dumps(
#             data, ensure_ascii=False)  # type: ignore
#         translation.updated_at = datetime.now()  # type: ignore
#         translation.targetCountry = targetCountry
#         translation.expert_edit = expertEdit
#         translation.customer_edit = customerEdit
#         translation.transAccept = transAccept
#         translation.transEdit = transEdit

#         db.add(translation)
#         db.commit()
#         db.refresh(translation)
#         print("PostgreSQL record updated successfully.")

#         # --- Invalidate caches ---
#         try:
#             invalidate_full_translation_cache(
#                 shop_domain, targetLanguage, translation.brand_tone, targetCountry)
#             invalidate_extracted_data_cache(
#                 shop_domain, targetLanguage, targetCountry)
#             print(
#                 f"Caches invalidated for shopDomain: {shop_domain}, targetLanguage: {targetLanguage}:{targetCountry}")
#         except Exception as e:
#             print(f"Cache invalidation failed: {e}")

#         # --- Qdrant embedding (optional & safe) ---
#         # print(qdrant.get_collection(COLLECTION_NAME))
#         try:
#             response = client.embeddings.create(
#                 model="text-embedding-3-small",
#                 input=new_value,
#                 encoding_format="float"
#             )
#             embedding = response.data[0].embedding
#             correction_point = PointStruct(
#                 id=str(uuid.uuid4()),
#                 vector={"GlobalFlow": embedding},
#                 payload={
#                     "data_type": "correction",
#                     "user_id": str(user["_id"]) if user else None,
#                     "industry": (user or {}).get("industry") or "Unknown",
#                     "postgres_id": translation.id,
#                     "shopDomain": shop_domain,
#                     "targetLanguage": targetLanguage,
#                     "targetCountry": targetCountry,
#                     "path": path,
#                     "newValue": new_value,
#                     "originalValue": original_value,
#                     "expertEdit": req["expertEdit"] if "expertEdit" in req else None,
#                     "customerEdit": req["customerEdit"] if "customerEdit" in req else None,
#                     "transAccept": req["transAccept"] if "transAccept" in req else None,
#                     "transEdit": req["transEdit"] if "transEdit" in req else None,
#                     "ai_rating": ai_rating,
#                     "ai_reason": ai_reason,
#                     "date": datetime.utcnow().isoformat(),
#                 }
#             )
#             qdrant.upsert(collection_name=COLLECTION_NAME,
#                           points=[correction_point])
#             print("Qdrant embedding stored successfully.")
#         except Exception as e:
#             print(f" Qdrant embedding failed: {e}")

#         # ---  Return clean response ---
#         return {
#             "status": "success",
#             "translation_id": translation.id,
#             "updatedPath": path,
#             "oldValue": original_value,
#             "newValue": new_value,
#             "shopDomain": translation.shop_domain,
#             "targetLanguage": translation.target_lang,
#             "targetCountry": translation.targetCountry,
#             "ai_rating": ai_rating,
#             "ai_reason": ai_reason,
#             "updatedJson": translation.translated_text_json,
#         }

#     except Exception as e:
#         print(" Unexpected API error:", str(e))
#         return {"status": "error", "message": str(e)}
