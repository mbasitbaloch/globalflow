from datetime import datetime
from fastapi import APIRouter, Depends, Body
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy.orm import Session
import requests
from ..database import SessionLocal
from ..models import Translation
from ..services.translator import fast_translate_json
# from ..mongodb import users_collection
from fastapi.responses import JSONResponse, FileResponse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, PointStruct, VectorParams, Distance
import json
import os
import uuid
from bson import ObjectId
from dotenv import load_dotenv
from openai import OpenAI
from ..utils.tasks import store_data
# from langchain.embeddings import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores.qdrant import Qdrant
from langchain_qdrant import Qdrant


load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

router = APIRouter()


# @celery_app.task()


@router.post("/shopify/translate")
async def shopify_translate(req: dict):
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
    return FileResponse(
        file_path,
        media_type="application/json",
        filename=file_name
    )
