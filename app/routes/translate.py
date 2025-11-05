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
from ..services.translator import fast_translate_json
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

COLLECTION_NAME = settings.COLLECTION_NAME

router = APIRouter()

db: Session = SessionLocal()

# Get user data
# user = users_collection.find_one(
#     {"shopifyStores.shopDomain": req["shopDomain"]})


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
    Expect body:
    {
      "shopDomain": "...",
      "accessToken": "...",
      "targetLanguage": "fr",
      "targetcountry":"FR",
      "brandTone": "neutral"
    }
    """
    # Validate request
    required_fields = ["shopDomain", "accessToken",
                       "targetLanguage", "brandTone", "targetcountry"]

    # Check all required fields exist
    missing = [f for f in required_fields if f not in req]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

    # Check none are empty or null
    empty_fields = [f for f in required_fields if not str(req[f]).strip()]
    if empty_fields:
        raise HTTPException(
            status_code=400, detail=f"Empty values found in: {', '.join(empty_fields)}, please enter an entry for {', '.join(empty_fields)}")

    # Validate shop domain properly
    shop_domain = req["shopDomain"].strip()
    domain_pattern = re.compile(
        # e.g., something.com or abc.myshopify.com
        r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$"
    )
    if not domain_pattern.match(shop_domain):
        raise HTTPException(
            status_code=400, detail="Invalid shop domain format. Please enter a valid domain like 'example.myshopify.com'.")

    # Get user from MongoDB
    shop_domain = req["shopDomain"]
    targetLanguage = req["targetLanguage"]
    targetCountry = req["targetcountry"]
    brand_tone = req["brandTone"]

    user = users_collection.find_one({"shopifyStores.shopDomain": shop_domain})
    if not user:
        raise HTTPException(status_code=404, detail="Shop not found")
    industry = user.get("industry", "general")
    user_id = str(user["_id"])

    # Check extracted data cache
    cache_key = f"shopify_data:{shop_domain}:{targetLanguage}:{targetCountry}"
    cached_data = get_cached_extracted_data(cache_key)
    raw_data = None

    if cached_data:
        # Fetch fresh data to compare
        url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
        response = requests.post(url, json={
            "shopDomain": shop_domain,
            "accessToken": req["accessToken"],
            "targetLanguage": targetLanguage,
            "brandTone": brand_tone,
            "targetcountry": targetCountry
        })
        response.raise_for_status()
        fresh_data = response.json()
        fresh_hash = compute_raw_hash(fresh_data)
        cached_hash = compute_raw_hash(cached_data)

        if fresh_hash == cached_hash:
            print(
                f"Extracted data cache hit for {shop_domain}:{targetLanguage}:{targetCountry}")
            raw_data = cached_data
        else:
            print(
                f"Extracted data cache miss (hash mismatch: {cached_hash[:8]} != {fresh_hash[:8]})")
            raw_data = fresh_data
    else:
        print(
            f"No cached extracted data for {shop_domain}:{targetLanguage}:{targetCountry}")
        # Fetch fresh data from Shopify
        url = "https://stagingapi.globalflow.ai/api/shopify/unauth/get-all-store-data"
        response = requests.post(url, json={
            "shopDomain": shop_domain,
            "accessToken": req["accessToken"],
            "targetLanguage": targetLanguage,
            "brandTone": brand_tone,
            "targetcountry": targetCountry
        })
        response.raise_for_status()
        raw_data = response.json()

    # Cache the extracted data
    cache_extracted_data(cache_key, raw_data)

    # Check full translation cache
    fresh_hash = compute_raw_hash(raw_data)
    cached_translated = get_full_translation_from_cache(
        shop_domain, targetLanguage, brand_tone, fresh_hash, targetCountry)
    if cached_translated:
        print(f"Translation served from cache for {shop_domain} (hash match)")
        # Save original JSON
        file_name = f"Today_fetched_{uuid.uuid4().hex}.json"
        file_path = os.path.join("fetched_data", file_name)
        os.makedirs("fetched_data", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        # Save to PostgreSQL
        translation_record = Translation(
            user_id=user_id,
            industry=industry,
            shop_domain=shop_domain,
            brand_tone=brand_tone,
            target_lang=targetLanguage,
            targetCountry=targetCountry,
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

        print("Celery task started...")
        task = store_data.delay(cached_translated, req,
                                raw_data, translation_record.id)
        print(f"New task ID: {task.id}")

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
        target_lang=targetLanguage,
        targetCountry=targetCountry,
        brand_tone=brand_tone,
        industry=industry
    )

    # Cache the full translated JSON
    set_full_translation_in_cache(
        shop_domain, targetLanguage, brand_tone, translated_data, fresh_hash, targetCountry)

    # Save original JSON to file
    file_name = f"Today_fetched_{uuid.uuid4().hex}.json"
    file_path = os.path.join("fetched_data", file_name)
    os.makedirs("fetched_data", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    # Save translated JSON to file
    print("Saving translated JSON to file...")
    file_name = f"Today_translated_{uuid.uuid4().hex}.json"
    file_path = os.path.join("tmp", file_name)
    os.makedirs("tmp", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    print("Translated JSON saved to file:", file_path)

    # Save to PostgreSQL
    translation_record = Translation(
        user_id=user_id,
        industry=industry,
        shop_domain=shop_domain,
        brand_tone=brand_tone,
        target_lang=targetLanguage,
        targetCountry=targetCountry,
        content_type="json",
        original_text_raw=json.dumps(raw_data, ensure_ascii=False),
        original_text_json=raw_data,
        translated_text_raw=json.dumps(translated_data, ensure_ascii=False),
        translated_text_json=translated_data
    )
    db.add(translation_record)
    db.commit()
    db.refresh(translation_record)

    print("Celery task started...")
    task = store_data.delay(translated_data, req,
                            raw_data, translation_record.id)
    print(f"New task ID: {task.id}")

    # Return file for download
    return {
        "message": "Translation completed successfully",
        "file_path": file_path,
        "translation_id": translation_record.id,
        "translation": translated_data
    }


@router.put("/shopify/update-string")
async def update_translated_string(req: UpdateRequest, db: Session = Depends(get_db)):
    """
    Safely updates a translation with AI validation & Qdrant embedding.
    Prevents crashes if AI or Qdrant fails.
    """
    try:
        # --- Validate request ---
        # required_fields = ["translation_id", "shopDomain",
        #                    "targetLanguage", "path", "newValue", "targetcountry"]
        # # if not all(k in req for k in required_fields):
        # #     return {"status": "error", "message": "Missing required fields"}

        # missing = [f for f in required_fields if f not in req or req[f] is None]
        # if missing:
        #     raise HTTPException(
        #         status_code=400,
        #         detail=f"Missing required field: {', '.join(missing)}"
        #     )

        translation_id = req.translation_id
        shop_domain = req.shopDomain
        targetLanguage = req.targetLanguage
        targetCountry = req.targetcountry
        path = req.path
        new_value = req.newValue
        original_value = req.originalValue or ""
        expertEdit = req.expertEdit
        customerEdit = req.customerEdit
        transAccept = req.transAccept
        transEdit = req.transEdit

        # --- Fetch translation record ---
        translation = db.query(Translation).filter_by(
            id=translation_id).first()
        if not translation:
            return {"status": "error", "message": "Translation not found"}

        # if translation.shop_domain != shop_domain or translation.target_lang != targetLanguage:
        #     return {"status": "error", "message": "Shop or language mismatch"}

            # --- Verify domain, language, and country consistency ---
        if translation.shop_domain != shop_domain:
            return {
                "status": "error",
                "message": (
                    f"Shop domain mismatch: Request domain '{shop_domain}' "
                    f"does not match stored domain '{translation.shop_domain}'."
                )
            }

        # If request targetLanguage doesn’t match stored target_lang
        if translation.target_lang != targetLanguage:
            return {
                "status": "error",
                "message": (
                    f"Target language mismatch: The translation record was created for "
                    f"'{translation.target_lang}', but you are trying to update using '{targetLanguage}'. "
                    "Please use the same language as the existing translation."
                )
            }

        # Handle country logic smartly:
        if translation.targetCountry:
            # If record already has a country, enforce consistency
            if translation.targetCountry.lower() != targetCountry.lower():
                return {
                    "status": "error",
                    "message": (
                        f"Target country mismatch: Existing translation country is '{translation.targetCountry}', "
                        f"but you tried to update using '{targetCountry}'."
                    )
                }
        else:
            # If DB has no country, safely assign it
            translation.targetCountry = targetCountry
            # db.add(translation)
            # db.commit()
            print(
                f"Target country '{targetCountry}' saved for translation {translation.id}.")

        # --- Get user ---
        user = users_collection.find_one(
            {"shopifyStores.shopDomain": shop_domain})
        if not user:
            return {"status": "error", "message": "Shop not found"}

        # --- Initialize OpenAI client ---
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY_1)
        except Exception as e:
            print(f"OpenAI client error: {e}")
            return {"status": "error", "message": f"OpenAI init failed: {e}"}

        # --- AI validation ---
        ai_rating, ai_reason = 0, "AI validation failed"
        try:
            print("Validating update with AI...")
            ai_validation = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are an AI evaluator responsible for rating text updates based on their quality, meaning, and contextual relevance. "
                                f"Only approve updates that are meaningful, relevant, and linguistically appropriate to the user's target language and region. "
                                f"Evaluate the text using the natural tone, expressions, and writing style used in the specified country for that language. "
                                f"For example, if the language is English and the country is the United Kingdom, prefer 'colour' over 'color', and if the country is the United States, prefer 'color' over 'colour'. "
                                f"Apply equivalent tone and spelling distinctions for other languages and regions. "
                                f"Be strict — reject updates that are incorrect, unnatural, off-tone, or contextually irrelevant. "
                                f"Target Language: {targetLanguage} | Target Country/Region: {targetCountry}. "
                                f"Return your response strictly in JSON format with the following structure: "
                                "{ 'rating': <float between 0 and 1>, 'reason': '<clear explanation of your decision>' }."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Original: {original_value}\nUpdated: {new_value}"
                    },
                ],
                temperature=0.3,
            )

            ai_response = ai_validation.choices[0].message.content
            print("AI Response:", ai_response)

            try:
                rating_data = json.loads(ai_response)  # type: ignore
                ai_rating = float(rating_data.get("rating", 0))
                ai_reason = rating_data.get("reason", "")
            except Exception:
                ai_rating = 0
                ai_reason = "Invalid AI response format"

        except Exception as e:
            print(f"AI validation error: {e}")
            ai_rating = 0
            ai_reason = "AI validation request failed"

        # --- If AI rejects, return safely ---
        if ai_rating < 0.6:
            return {
                "status": "rejected",
                "ai_rating": ai_rating,
                "ai_reason": ai_reason,
                "message": f"Update rejected by AI (rating={ai_rating:.2f})"
            }

        print(f"AI approved (rating={ai_rating:.2f})")

        # --- Fetch translation record ---
        # translation = db.query(Translation).filter_by(
        #     id=translation_id).first()
        # if not translation:
        #     return {"status": "error", "message": "Translation not found"}

        # if translation.shop_domain != shop_domain or translation.target_lang != targetLanguage:
        #     return {"status": "error", "message": "Shop or language mismatch"}

        # --- Apply JSON update ---
        data = translation.translated_text_raw
        if isinstance(data, str):
            data = json.loads(data)

        keys = path.split(".")
        ref = data
        for k in keys[:-1]:
            ref = ref[int(k)] if k.isdigit() else ref[k]

        # --- Apply the new value ---
        last_key = keys[-1]
        ref[last_key] = new_value  # type: ignore
        # ref[keys[-1]] = new_value

        # --- Handle priority + aiTranslated flags dynamically ---
        priority_key = f"priorityReview_{last_key}"
        ai_flag_key = f"aiTranslated_{last_key}"

        # If expert edited, clear review + AI flags
        if req.expertEdit:
            if priority_key in ref:
                ref[priority_key] = False  # type: ignore
            if ai_flag_key in ref:
                ref[ai_flag_key] = False  # type: ignore

        translation.translated_text_json = data
        translation.translated_text_raw = json.dumps(
            data, ensure_ascii=False)  # type: ignore
        translation.updated_at = datetime.now()  # type: ignore
        translation.targetCountry = targetCountry
        translation.expert_edit = expertEdit
        translation.customer_edit = customerEdit
        translation.transAccept = transAccept
        translation.transEdit = transEdit

        db.add(translation)
        db.commit()
        db.refresh(translation)
        print("PostgreSQL record updated successfully.")

        # --- Invalidate caches ---
        try:
            invalidate_full_translation_cache(
                shop_domain, targetLanguage, translation.brand_tone, targetCountry)
            invalidate_extracted_data_cache(
                shop_domain, targetLanguage, targetCountry)
            print(
                f"Caches invalidated for shopDomain: {shop_domain}, targetLanguage: {targetLanguage}:{targetCountry}")
        except Exception as e:
            print(f"Cache invalidation failed: {e}")

        # --- Qdrant embedding (optional & safe) ---
        # print(qdrant.get_collection(COLLECTION_NAME))
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=new_value,
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
                    "shopDomain": shop_domain,
                    "targetLanguage": targetLanguage,
                    "targetCountry": targetCountry,
                    "path": path,
                    "newValue": new_value,
                    "originalValue": original_value,
                    "expertEdit": req["expertEdit"] if "expertEdit" in req else None,
                    "customerEdit": req["customerEdit"] if "customerEdit" in req else None,
                    "transAccept": req["transAccept"] if "transAccept" in req else None,
                    "transEdit": req["transEdit"] if "transEdit" in req else None,
                    "ai_rating": ai_rating,
                    "ai_reason": ai_reason,
                    "date": datetime.utcnow().isoformat(),
                }
            )
            qdrant.upsert(collection_name=COLLECTION_NAME,
                          points=[correction_point])
            print("Qdrant embedding stored successfully.")
        except Exception as e:
            print(f" Qdrant embedding failed: {e}")

        # ---  Return clean response ---
        return {
            "status": "success",
            "translation_id": translation.id,
            "updatedPath": path,
            "oldValue": original_value,
            "newValue": new_value,
            "shopDomain": translation.shop_domain,
            "targetLanguage": translation.target_lang,
            "targetCountry": translation.targetCountry,
            "ai_rating": ai_rating,
            "ai_reason": ai_reason,
            "updatedJson": translation.translated_text_json,
        }

    except Exception as e:
        print(" Unexpected API error:", str(e))
        return {"status": "error", "message": str(e)}
