import json
import random
import re
import asyncio
import os
import sys
from datetime import datetime
import time
# from openai import AsyncOpenAI
# import google.generativeai as genai  # Gemini SDK
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import settings
from qdrant_client import QdrantClient
from app.services.hs_langchain import fewshotTranslation, TranslationQuery, promptClassification, voteClassification, SafeJsonParser
from ..utils.tasks import store_examples
from pydantic import SecretStr
from qdrant_client.http import models
from collections import defaultdict


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

BATCHES_DIR = os.path.join(LOG_DIR, "batches")
os.makedirs(BATCHES_DIR, exist_ok=True)

REPORT_DIR = os.path.join(LOG_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

CONSOLE_DIR = os.path.join(LOG_DIR, "console")
os.makedirs(CONSOLE_DIR, exist_ok=True)
console_file = os.path.join(CONSOLE_DIR, "console.json")

raw_logs = [{"message": "Logs"}]
with open(console_file, "w", encoding="utf-8") as f:
    json.dump(raw_logs[0], f, ensure_ascii=False, indent=4)

# ===================== GLOBAL REPORT TRACKER =====================
TRANSLATION_STATS = {
    "rate_limits": {
        "openai1": [],
        "openai2": [],
        "gemini1": [],
        "total": []
    },
    "fallbacks": {
        "openai1": [],
        "openai2": [],
        "gemini1": []
    },
    "tokens": {
        "openai1": [],
        "openai2": [],
        "gemini1": []
    },
    "mismatches": {
        "batches": [],
        "total_mismatched": 0
    }
}


# ==== CLIENTS ====

langchain_openai_1 = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    api_key=SecretStr(settings.OPENAI_API_KEY_1)
)
langchain_openai_2 = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    api_key=SecretStr(settings.OPENAI_API_KEY_2)
)

langchain_gemini_1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    google_api_key=settings.GEMINI_API_KEY_1
)

# openai_model_1 = AsyncOpenAI(
#     api_key=settings.OPENAI_API_KEY_1,
#     timeout=150.0
# )

# openai_model_2 = AsyncOpenAI(
#     api_key=settings.OPENAI_API_KEY_2,
#     timeout=150.0
# )

# genai.configure(api_key=settings.GEMINI_API_KEY_1) # type: ignore
# gemini_model_1 = genai.GenerativeModel("gemini-2.0-flash-lite") # type: ignore # gemini-1.5-flash

# genai.configure(api_key=settings.GEMINI_API_KEY_2)
# gemini_model_2 = genai.GenerativeModel("gemini-1.5-flash")

qdrant = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    prefer_grpc=False,
    timeout=60
)

# ==== CONFIG ====
CLASSIFICATION_BATCH_SIZE = 50
MAX_CONCURRENCY_CLASSIFICATION = 20
semaphore_classification = asyncio.Semaphore(MAX_CONCURRENCY_CLASSIFICATION)
VOTING_BATCH_SIZE = 50
MAX_CONCURRENCY_VOTING = 20
semaphore_voting = asyncio.Semaphore(MAX_CONCURRENCY_VOTING)
TRANSLATION_BATCH_SIZE = 50
MAX_CONCURRENCY_TRANSLATION = 20
semaphore_translation = asyncio.Semaphore(MAX_CONCURRENCY_TRANSLATION)

classification_model_cycle = ["openai1", "openai2", "gemini1"]
model_index_classify = 0

voting_model_cycle = ["openai1", "openai2"]
model_index_voting = 0

translation_model_cycle = ["openai1", "openai2", "gemini1"]
model_index_translation = 0

sys.setrecursionlimit(3000)


# ===================== HELPERS =====================
def is_translateable(text: str) -> bool:
    unused = [
        "<strong style=\"text-transform:uppercase\">%{discount_rejection_message}</strong>",
        "%{product_name} / %{variant_label}",
        "%{price}%{accessible_separator}%{per_unit}",
        "%{price}/%{unit}",
        "%{price}/%{count}%{unit}",
        "•••• %{last_characters}",
        "%{quantity} × %{product_title}",
        "%{min_time}–%{max_time}",
        "%{firstMethod}, %{secondMethod}",
        "%{rest}, %{current},",
        "%{merchandise_title} ×%{quantity}",
        "•••• %{last_digits}",
        "%{currency} (%{currency_symbol})",
        "1 %{from_currency_code} = %{rate} %{to_currency_code}",
        "%{tip_percent}%",
        "+{{numberOfAdditionalProducts}}",
        "-",
        "{{count}}+",
        "{{ quantity }}+",
        "<p></p>",
        "CPF/CNPJ",
        "RUT",
        "CI/RUC/IVA",
        "NIT/IVA",
        "NPWP",
        "RFC",
        "DNI/RUC/CE",
        "NIF/IVA",
        "DNI/NIF",
        "SKU",
    ]
    if not text or not text.strip():
        return False
    if text.isdigit():
        return False
    if re.match(r"^\d+(\.\d+)?$", text):
        return False
    if re.match(r"^\d{4}-\d{2}-\d{2}T", text):
        return False
    if re.match(r"^[a-f0-9]{32,64}$", text):
        return False
    if text.startswith("gid://"):
        return False
    if re.match(r"^\{\{.*\}\}$", text):
        return False
    if "@" in text and "." in text:
        return False
    if re.match(r"^https?://", text):
        return False
    if text.startswith(("shopify.", "customer.", "customer_", "templates.", "section.", "sections.", "GlobalFlow.", "shopify:")):
        return False
    if text in unused:
        return False
    return True


def clean_line(line: str) -> str:
    line = line.replace("\u0000", "").replace("\x00", "")
    return re.sub(r'^\d+[\.\)]\s*', '', line).strip()


async def with_retry(fn, *args, retries=3, **kwargs):
    provider = kwargs.get("provider", "unknown")
    batch_num = kwargs.get("batch_num", 0)
    type = kwargs.get("type", "N/A")
    for i in range(retries):
        try:
            # filtered_kwargs = {
            #     k: v for k, v in kwargs.items()
            #     if k not in ("provider", "batch_num", "type")
            # }
            return await fn(*args, **kwargs)
        except Exception as e:
            if "Rate limit" in str(e) or "quota" in str(e).lower():
                wait = (2 ** i) + random.random()
                msg = f"⚠ Rate limit for {provider}, batch {type} {batch_num}, retrying in {wait:.2f}s..."
                print(msg)
                TRANSLATION_STATS["rate_limits"]["total"].append(
                    {"provider": provider, "time": datetime.now().isoformat(), "wait": wait})
                TRANSLATION_STATS["rate_limits"][provider].append(
                    {"time": datetime.now().isoformat(), "wait": wait})
                await asyncio.sleep(wait)
            else:
                print(
                    f"⚠ Error in, batch {type} {batch_num}, {provider}: {e}, retrying...")
                await asyncio.sleep(2)
                break
    raise Exception(
        f"Max retries reached for {type} {batch_num} by {provider}")


def qdrant_examples(shopDomain, targetLanguage, user_id):
    start = datetime.now()
    fewshot_data = None
    qdrant_collection = settings.COLLECTION_NAME

    qdrant.create_payload_index(
        collection_name=qdrant_collection,
        field_name="shopDomain",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    qdrant.create_payload_index(
        collection_name=qdrant_collection,
        field_name="targetLanguage",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    qdrant.create_payload_index(
        collection_name=qdrant_collection,
        field_name="user_id",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    qdrant.create_payload_index(
        collection_name=qdrant_collection,
        field_name="type",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    scroll_results, _ = qdrant.scroll(
        collection_name=qdrant_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="shopDomain",
                    match=models.MatchValue(value=shopDomain)
                ),
                models.FieldCondition(
                    key="targetLanguage",
                    match=models.MatchValue(value=targetLanguage)
                ),
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                ),
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value="prompt_data")
                )
            ]
        ),
        with_payload=True,
        limit=1
    )

    for point in scroll_results:
        fewshot_data = point.payload.get(
            "fewshot_data") if point.payload else None

    examples = []
    if fewshot_data:
        if isinstance(fewshot_data, str):
            fewshot_data = json.loads(fewshot_data)

        examples = [
            {
                "original": json.dumps([ex.get("original", "") for ex in fewshot_data[:50]], ensure_ascii=False),
                "translated": json.dumps([ex.get("translated", "") for ex in fewshot_data[:50]], ensure_ascii=False)
            },
            {
                "original": json.dumps([ex.get("original", "") for ex in fewshot_data[50:100]], ensure_ascii=False),
                "translated": json.dumps([ex.get("translated", "") for ex in fewshot_data[50:100]], ensure_ascii=False)
            },
            {
                "original": json.dumps([ex.get("original", "") for ex in fewshot_data[100:150]], ensure_ascii=False),
                "translated": json.dumps([ex.get("translated", "") for ex in fewshot_data[100:150]], ensure_ascii=False)
            },
        ]

    end = datetime.now()
    print(
        f"Examples for fewshot retrieved from qdrant, time taken: {end-start}")
    return examples

# # ===================== CLASSIFICATION FUNCTIONS =====================


async def _classify_openai(strings_batch, classification_model):
    """
    Classify strings into 'business' or 'ordinary'.
    """

    labels = await promptClassification(classification_model, strings_batch)
    # return [clean_line(label) for label in labels]

    if isinstance(labels, str):
        labels = labels.strip()
        labels = re.sub(r"^```[a-zA-Z]*\n?", "", labels)
        labels = re.sub(r"```$", "", labels)
        labels = labels.strip()
    try:
        data = json.loads(labels) if isinstance(labels, str) else labels
        if isinstance(data, dict) and "labels" in data:
            return [clean_line(x) for x in data["labels"]]
        elif isinstance(data, list):
            return [clean_line(x) for x in data]
        else:
            raise ValueError("Unexpected Openai response")
    except Exception as e:
        print(f"⚠ Parse fallback: {e}")
        labels_str = str(labels)
        lines = [line for line in labels_str.splitlines() if line.strip()]
        return [clean_line(line) for line in lines]


async def classify_openai_1(strings_batch, *args, **kwargs):
    return await _classify_openai(strings_batch, langchain_openai_1)


async def classify_openai_2(strings_batch, *args, **kwargs):
    return await _classify_openai(strings_batch, langchain_openai_1)


async def _classify_gemini(strings_batch, classification_model):
    """
    Classify strings into 'business' or 'ordinary'.
    """

    labels = await promptClassification(classification_model, strings_batch)
    # return [clean_line(label) for label in labels]

    if isinstance(labels, str):
        labels = labels.strip()
        labels = re.sub(r"^```[a-zA-Z]*\n?", "", labels)
        labels = re.sub(r"```$", "", labels)
        labels = labels.strip()
    try:
        data = json.loads(labels) if isinstance(labels, str) else labels
        if isinstance(data, dict) and "labels" in data:
            return [clean_line(x) for x in data["labels"]]
        elif isinstance(data, list):
            return [clean_line(x) for x in data]
        else:
            raise ValueError("Unexpected Gemini response")
    except Exception as e:
        print(f"⚠ Parse fallback: {e}")
        labels_str = str(labels)
        lines = [line for line in labels_str.splitlines() if line.strip()]
        return [clean_line(line) for line in lines]


async def classify_gemini_1(strings_batch, *args, **kwargs):
    return await _classify_gemini(strings_batch, langchain_gemini_1)


# ===================== BATCH CLASSIFICATION =====================
async def _classify_batch(indexed_strings, batch_num, total_batches, classification_progress=None):
    global model_index_classify
    strings = [s for _, s in indexed_strings]
    classification_model_cycle = ["openai1", "openai2", "gemini1"]
    VALID_LABELS = ["ordinary", "business"]
    raw_ordinary = ["ordinary" for i in range(50)]
    async with semaphore_classification:
        current_model = classification_model_cycle[model_index_classify % len(
            classification_model_cycle)]
        model_index_classify += 1
        print(
            f"\n[DEBUG] Batch {batch_num}/{total_batches} via {current_model} → {len(strings)} strings ")

        try:
            if current_model == "openai1":
                result = await with_retry(classify_openai_1, strings, batch_num, total_batches, provider=current_model)
            elif current_model == "openai2":
                result = await with_retry(classify_openai_2, strings, batch_num, total_batches, provider=current_model)
            else:  # gemini1
                result = await with_retry(classify_gemini_1, strings, batch_num, total_batches, provider=current_model)

            labels = []
            for l in result:
                if l in VALID_LABELS:
                    labels.append(l)

        except Exception as e:
            if classification_progress is not None:
                classification_progress["partial"] += 1
                print(f"[CLASSIFICATION PROGRESS: failed] {classification_progress['valid']} valid, {classification_progress['partial']} partial, total {classification_progress['valid']+classification_progress['partial']}/{classification_progress['total']} (batch {batch_num} via {current_model})")
            return [(i, l) for (i, _), l in zip(indexed_strings, raw_ordinary)]

        if labels:
            expected = len(strings)
            got = len(labels)

            if expected == got:
                if classification_progress is not None:
                    classification_progress["valid"] += 1
                    print(f"[CLASSIFICATION PROGRESS: valid] {classification_progress['valid']} valid, {classification_progress['partial']} partial, total {classification_progress['valid']+classification_progress['partial']}/{classification_progress['total']} (batch {batch_num} via {current_model})")
                return [(i, l) for (i, _), l in zip(indexed_strings, labels)]
            # --- FIX: force align translations ---
            elif got < expected:
                # Pad missing with ordinary
                print(f"Expected {expected}, got {got} -> Padding ordinary")
                labels.extend(raw_ordinary[got:])
            else:
                print(f"Expected {expected}, got {got} -> Truncating extra")
                labels = labels[:expected]

            if classification_progress is not None:
                classification_progress["partial"] += 1
                print(f"[CLASSIFICATION PROGRESS: partial] {classification_progress['valid']} valid, {classification_progress['partial']} partial, total {classification_progress['valid']+classification_progress['partial']}/{classification_progress['total']} (batch {batch_num} via {current_model})")
            return [(i, l) for (i, _), l in zip(indexed_strings, labels)]


# ===================== VOTING FUNCTIONS =====================

async def _vote_openai(strings_batch, voting_model):
    """
    Vote classification results.
    """

    votes = await voteClassification(voting_model, strings_batch)
    # return [clean_line(vote) for vote in votes]
    return votes

    # if isinstance(votes, str):
    #     votes = votes.strip()
    #     votes = re.sub(r"^```[a-zA-Z]*\n?", "", votes)
    #     votes = re.sub(r"```$", "", votes)
    #     votes = votes.strip()
    # try:
    #     data = json.loads(votes) if isinstance(votes, str) else votes
    #     if isinstance(data, dict) and "votes" in data:
    #         return [clean_line(x) for x in data["votes"]]
    #     elif isinstance(data, list):
    #         return [clean_line(x) for x in data]
    #     else:
    #         raise ValueError("Unexpected Openai response")
    # except Exception as e:
    #     print(f"⚠ Parse fallback: {e}")
    #     votes_str = str(votes)
    #     lines = [line for line in votes_str.splitlines() if line.strip()]
    #     return [clean_line(line) for line in lines]


async def vote_openai_1(strings_batch, *args, **kwargs):
    return await _vote_openai(strings_batch, langchain_openai_1)


async def vote_openai_2(strings_batch, *args, **kwargs):
    return await _vote_openai(strings_batch, langchain_openai_2)


# ===================== VOTE CLASSIFICATION =====================

async def _voting_batch(strings_batch, batch_num, total_batches, voting_progress=None):
    global model_index_voting
    strings = [[string, label] for (_, string, label) in strings_batch]
    voting_model_cycle = ["openai1", "openai2"]
    # VALID_VOTES = [True, False]
    raw_votes = [True] * 50

    async with semaphore_voting:
        current_model = voting_model_cycle[model_index_voting % len(
            voting_model_cycle)]
        model_index_voting += 1
        print(
            f"\n[DEBUG] Batch {batch_num}/{total_batches} via {current_model} → {len(strings)} strings ")

        try:
            if current_model == "openai1":
                result = await with_retry(vote_openai_1, strings, batch_num, total_batches, provider=current_model)
            else:  # openai2
                result = await with_retry(vote_openai_2, strings, batch_num, total_batches, provider=current_model)

            votes = []
            for v in result:
                if isinstance(v, bool):
                    votes.append(v)
                else:
                    votes.append(True)

        except Exception as e:
            if voting_progress is not None:
                voting_progress["partial"] += 1
                print(
                    f"[VOTING PROGRESS: failed] {voting_progress['valid']} valid, {voting_progress['partial']} partial, total {voting_progress['valid']+voting_progress['partial']}/{voting_progress['total']} (batch {batch_num} via {current_model})")
            return [(i, v) for (i, _, _), v in zip(strings_batch, raw_votes)]

        if votes:
            expected = len(strings)
            got = len(votes)

            if expected == got:
                if voting_progress is not None:
                    voting_progress["valid"] += 1
                    print(
                        f"[VOTING PROGRESS: valid] {voting_progress['valid']} valid, {voting_progress['partial']} partial, total {voting_progress['valid']+voting_progress['partial']}/{voting_progress['total']} (batch {batch_num} via {current_model})")
                return [(i, v) for (i, _, _), v in zip(strings_batch, votes)]
            # --- FIX: force align translations ---
            elif got < expected:
                # Pad missing with ordinary
                print(f"Expected {expected}, got {got} -> Padding raw votes")
                votes.extend(raw_votes[got:])
            else:
                print(f"Expected {expected}, got {got} -> Truncating extra")
                votes = votes[:expected]

            if voting_progress is not None:
                voting_progress["partial"] += 1
                print(
                    f"[VOTING PROGRESS: partial] {voting_progress['valid']} valid, {voting_progress['partial']} partial, total {voting_progress['valid']+voting_progress['partial']}/{voting_progress['total']} (batch {batch_num} via {current_model})")
            return [(i, v) for (i, _, _), v in zip(strings_batch, votes)]


# ===================== TRANSLATION FUNCTIONS =====================
async def _translate_openai(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, model, batch_num, type, provider):
    query = TranslationQuery(
        input=strings,
        user_id=user_id,
        shopDomain=shopDomain,
        targetLanguage=target_lang,
        brandTone=brand_tone,
        industry=industry,
        num_strings=len(strings),
    )

    content = await fewshotTranslation(examples, model, query, SafeJsonParser)

    if isinstance(content, str):
        content = content.strip()
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"```$", "", content)
        content = content.strip()
    try:
        data = json.loads(content) if isinstance(content, str) else content
        if isinstance(data, dict) and "translations" in data:
            return [clean_line(x) for x in data["translations"]]
        elif isinstance(data, list):
            return [clean_line(x) for x in data]
        else:
            raise ValueError("Unexpected OpenAI response")
    except Exception as e:
        print(f"⚠ Parse fallback {provider} for {type} batch {batch_num}: {e}")
        content_str = str(content)
        lines = [line for line in content_str.splitlines() if line.strip()]
        return [clean_line(line) for line in lines]


async def translate_openai_1(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, type, provider):
    return await _translate_openai(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, langchain_openai_1, batch_num, type, provider)


async def translate_openai_2(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, type, provider):
    return await _translate_openai(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, langchain_openai_2, batch_num, type, provider)


async def _translate_gemini(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, model, batch_num, type, provider):
    query = TranslationQuery(
        input=strings,
        user_id=user_id,
        shopDomain=shopDomain,
        targetLanguage=target_lang,
        brandTone=brand_tone,
        industry=industry,
        num_strings=len(strings),
    )

    content = await fewshotTranslation(examples, model, query, SafeJsonParser)

    if isinstance(content, str):
        content = content.strip()
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"```$", "", content)
        content = content.strip()
    try:
        data = json.loads(content) if isinstance(content, str) else content
        if isinstance(data, dict) and "translations" in data:
            return [clean_line(x) for x in data["translations"]]
        elif isinstance(data, list):
            return [clean_line(x) for x in data]
        else:
            raise ValueError("Unexpected Gemini response")
    except Exception as e:
        print(f"⚠ Parse fallback {provider} for {type} batch {batch_num}: {e}")
        content_str = str(content)
        lines = [line for line in content_str.splitlines() if line.strip()]
        return [clean_line(line) for line in lines]


async def translate_gemini_1(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, type, provider):
    return await _translate_gemini(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, langchain_gemini_1, batch_num, type, provider)


# # async def translate_gemini_2(strings, target_lang, brand_tone):
# #     return await _translate_gemini(strings, target_lang, brand_tone, gemini_model_2)


# # ===================== BATCH TRANSLATION =====================
async def _translate_batch(indexed_strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, total_batches, type, translation_progress=None, logs={}):
    global model_index_translation
    strings = [s for _, s in indexed_strings]
    # print(strings)
    logs[f"{type}_{batch_num}"] = {}

    # Define provider order
    translation_model_cycle = ["openai1", "openai2", "gemini1"] if type == "business" else [
        "gemini1", "openai1", "openai2"]
    models_used = []

    async with semaphore_translation:
        current_model = translation_model_cycle[model_index_translation % len(
            translation_model_cycle)]
        model_index_translation += 1
        logs[f"{type}_{batch_num}"]["print"] = f"\n[DEBUG] {type.upper()} Batch {batch_num}/{total_batches} via {current_model} → {len(strings)} strings"
        print(logs[f"{type}_{batch_num}"]["print"])

        try:
            start = time.time()
            if current_model == "openai1":
                translations = await with_retry(translate_openai_1, strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, provider=current_model, type=type)
            elif current_model == "openai2":
                translations = await with_retry(translate_openai_2, strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, provider=current_model, type=type)
            else:
                translations = await with_retry(translate_gemini_1, strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, provider=current_model, type=type)

            elapsed = time.time() - start
            # rough estimate if API doesn’t return usage
            tokens_used = len(" ".join(strings)) // 4
            TRANSLATION_STATS["tokens"][current_model].append(
                {"tokens": tokens_used, "time": elapsed})
            models_used.append(current_model)

        except Exception as e:
            logs[f"{type}_{batch_num}"]["exc1"] = f"⚠ {current_model} failed for {type} batch {batch_num}, falling back: {e}"
            print(logs[f"{type}_{batch_num}"]["exc1"])
            for alt in translation_model_cycle:
                current_model = alt
                if current_model in models_used:
                    continue
                try:
                    start = time.time()
                    if current_model == "openai1":
                        translations = await translate_openai_1(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, type, provider=current_model)
                    elif current_model == "openai2":
                        translations = await translate_openai_2(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, type, provider=current_model)
                    else:
                        translations = await translate_gemini_1(strings, examples, user_id, shopDomain, target_lang, brand_tone, industry, batch_num, type, provider=current_model)

                    models_used.append(current_model)

                    elapsed = time.time() - start
                    # rough estimate if API doesn’t return usage
                    tokens_used = len(" ".join(strings)) // 4
                    TRANSLATION_STATS["tokens"][current_model].append(
                        {"tokens": tokens_used, "time": elapsed})
                    break
                except Exception as e2:
                    logs[f"{type}_{batch_num}"]["exc2"] = f"⚠ Fallback {current_model} also failed for {type} batch {batch_num}: {e2}"
                    print(logs[f"{type}_{batch_num}"]["exc2"])
            else:
                raise Exception(
                    "All providers failed for {type} batch {batch_num}")

        if translations:
            expected = len(strings)
            got = len(translations)

            if expected == got:
                if translation_progress is not None:
                    translation_progress["valid"] += 1
                    logs[f"{type}_{batch_num}"][
                        "valid"] = f"[TRANSLATION PROGRESS: valid] {translation_progress['valid']} valid, {translation_progress['partial']} partial, total {translation_progress['valid']+translation_progress['partial']}/{translation_progress['total']} ({type} batch {batch_num} via {current_model})"
                    print(logs[f"{type}_{batch_num}"]["valid"])
                return [(i, t) for (i, _), t in zip(indexed_strings, translations)]

            # --- FIX: force align translations ---
            elif got < expected:
                # Pad missing with originals
                logs[f"{type}_{batch_num}"]["padding"] = f"Expected {expected}, got {got} -> Padding {expected-got} from original batch"
                print(logs[f"{type}_{batch_num}"]["padding"])
                translations.extend(strings[got:])
            else:  # got > expected
                # Trim extras
                translations = translations[:expected]
                logs[f"{type}_{batch_num}"]["truncation"] = f"Expected {expected}, got {got} -> Truncating {got-expected} from translation batch"
                print(logs[f"{type}_{batch_num}"]["truncation"])

            # Record mismatch stats
            if expected != got:
                TRANSLATION_STATS["mismatches"]["batches"].append({
                    "batch": batch_num, "provider": current_model,
                    "expected": expected, "got": got, "adjusted_to": len(translations),
                    "mismatched": abs(expected - got)
                })
                TRANSLATION_STATS["mismatches"]["total_mismatched"] += abs(
                    expected - got)

            if translation_progress is not None:
                translation_progress["partial"] += 1
                logs[f"{type}_{batch_num}"][
                    "partial"] = f"[TRANSLATION PROGRESS: partial] {translation_progress['valid']} valid, {translation_progress['partial']} partial, total {translation_progress['valid']+translation_progress['partial']}/{translation_progress['total']} ({type} batch {batch_num} via {current_model})"
                print(logs[f"{type}_{batch_num}"]["partial"])
            return [(i, t) for (i, _), t in zip(indexed_strings, translations)]


# ===================== SAVE REPORT =====================
def save_report():
    existing = len([f for f in os.listdir(
        REPORT_DIR) if f.startswith("report_")])
    report_file = os.path.join(REPORT_DIR, f"report_{existing+1}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(TRANSLATION_STATS, f, ensure_ascii=False, indent=2)
    print(f" Report saved to {report_file}")


# ===================== MAIN TRANSLATOR =====================
async def fast_translate_json(target_data, user_id, shopDomain, target_lang, brand_tone, industry):
    positions = []  # (path, string, path_str)

    # ---------- CUSTOM COLLECTION RULES ----------
    def collect_strings(d, path=None, parent_key=None):
        if path is None:
            path = []

        if isinstance(d, dict):
            for k, v in d.items():
                # PRODUCTS
                if parent_key == "products" and k in ["title", "descriptionHtml", "productType", "vendor", "status"]:
                    if isinstance(v, str) and is_translateable(v):
                        positions.append(
                            (path + [k], v, ".".join(map(str, path + [k]))))
                elif parent_key == "images" and k == "altText":
                    if isinstance(v, str) and is_translateable(v):
                        positions.append(
                            (path + [k], v, ".".join(map(str, path + [k])))
                        )

                # VARIANTS
                elif parent_key == "variants" and k == "title":
                    if isinstance(v, str) and is_translateable(v):
                        positions.append(
                            (path + [k], v, ".".join(map(str, path + [k]))))
                # COLLECTIONS
                elif parent_key == "collections" and k in ["title", "descriptionHtml", "handle"]:
                    if isinstance(v, str) and is_translateable(v):
                        positions.append(
                            (path + [k], v, ".".join(map(str, path + [k]))))
                # BLOGS
                elif parent_key == "blogs" and k in ["title", "handle"]:
                    if isinstance(v, str) and is_translateable(v):
                        positions.append(
                            (path + [k], v, ".".join(map(str, path + [k]))))

                        # SHOP POLICIES - translatableContent → only value + locale
                elif parent_key == "shopPolicies" and k == "translatableContent" and isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            for field in ["value", "locale"]:  # only pick value and locale
                                if field in item and isinstance(item[field], str) and is_translateable(item[field]):
                                    positions.append(
                                        (path + [k, i, field], item[field],
                                         ".".join(map(str, path + [k, i, field])))
                                    )

                # TRANSLATABLE CONTENT - COMPLETE HANDLING (including locale)
                elif k == "translatableContent" and isinstance(v, list):
                    for i, item in enumerate(v):
                        # if isinstance(item, dict):
                        #     for field in ["value", "locale"]:
                        #         if field in item and isinstance(item[field], str) and is_translateable(item[field]):
                        #             positions.append(
                        #                 (path + [k, i, field], item[field],
                        #                  ".".join(map(str, path + [k, i, field])))
                        #             )

                        if isinstance(item, dict):
                            # Handle ALL translatable fields in translatableContent
                            for field in item:
                                if field in ["value", "locale"] and isinstance(item[field], str) and is_translateable(item[field]):
                                    positions.append(
                                        (path + [k, i, field], item[field],
                                         ".".join(map(str, path + [k, i, field])))
                                    )

                # Handle other common translatable fields
                elif k in ["title", "body", "value", "altText", "description", "name"]:
                    if isinstance(v, str) and is_translateable(v):
                        positions.append(
                            (path + [k], v, ".".join(map(str, path + [k]))))
                else:
                    collect_strings(v, path + [k], k)

        elif isinstance(d, list):
            for i, item in enumerate(d):
                collect_strings(item, path + [i], parent_key)

    # Splittter for strings using regex
    def split_into_chunks(text, max_len=300):
        """
        Split text into chunks of <= max_len, trying to split at '.' boundaries.
        """
        # sentences = re.split( r'(?<=[.?!])\s+', text)
        sentences = re.split(
            r'(?<=[.?!])(?=\s|<)|</p>(?=\s|<)|</li>(?=\s|<)|</h[1-6]>(?=\s|<)|</div>(?=\s|<)|'
            r'<br\s*/?>(?=\s|<)|</tr>(?=\s|<)|</td>(?=\s|<)|</ul>(?=\s|<)|</table>(?=\s|<)|\n{2,}',
            text,
            flags=re.IGNORECASE
        )
        chunks, current = [], ""

        for sentence in sentences:
            # sentence = sentence.strip()
            # if not sentence:
            #     continue
            if len(current) + len(sentence) + 1 > max_len:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                current += (" " if current else "") + sentence

        if current:
            chunks.append(current.strip())

        return chunks

    # Splitting large strings into chunks
    def expand_strings(strings, max_len=300):
        """
        Expand long strings into chunks inside the main list.
        Returns (expanded_list, mapping) for later reconstruction.
        """
        expanded = []
        mapping = []  # (original_index, number_of_chunks)
        counter = 0

        for idx, text in enumerate(strings):
            if len(text) > max_len:
                counter += 1
                chunks = split_into_chunks(text, max_len)
                expanded.extend(chunks)
                mapping.append((idx, len(chunks)))
                print(
                    f"Splitting string on index {idx} into {len(chunks)} chunks")
            else:
                expanded.append(text)
                mapping.append((idx, 1))

        return expanded, mapping

    # Recombine splitted strings with mapper
    def collapse_strings(processed_expanded, mapping):
        """
        Collapse processed expanded list back to original structure.
        """
        collapsed = []
        pos = 0
        for idx, count in mapping:
            merged = " ".join(processed_expanded[pos:pos+count])
            collapsed.append(merged)
            pos += count
            if count > 1:
                print(
                    f"String at index {idx} recombined by joining {count} strings")
        return collapsed

    def dedup_with_index_map(items):
        index_map = defaultdict(list)
        for i, item in enumerate(items):
            index_map[item].append(i)
        return list(index_map.keys()), index_map

    def reconstruct_from_map(unique_processed, unique_items, index_map, length):
        reconstructed = [None] * length
        for item, indices in index_map.items():
            for i in indices:
                reconstructed[i] = unique_processed[unique_items.index(item)]
        return reconstructed

    collect_strings(target_data)

    strings_to_translate = [s for _, s, _ in positions]
    print(f"Total strings: {len(strings_to_translate)}")

    locales = [s for _, s, path_str in positions if "locale" in path_str]
    locale = locales[0]

    counter = 0
    for line in strings_to_translate:
        words = len(line.split(" "))
        counter += words

    print(f"Total words: {counter}")

    expanded, mapping = expand_strings(strings_to_translate, max_len=300)
    unique_texts, index_map = dedup_with_index_map(expanded)
    strings_to_classify = [(i, s) for i, s in enumerate(unique_texts)]

    print(
        f"Total strings to be processed after deduplication and chunking: {len(unique_texts)}")

    # ---- SAVE EXTRACTED ----
    extracted_log = [{"path": p, "string": s} for _, s, p in positions]
    extracted_file = os.path.join(
        LOG_DIR, f"extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(extracted_file, "w", encoding="utf-8") as f:
        json.dump(extracted_log, f, ensure_ascii=False, indent=2)
    print(f"Saved extracted strings to {extracted_file}")

    def serialize_batches(batches, batch_type):
        serialized = []
        for b_idx, batch in enumerate(batches, start=1):
            serialized.append({
                "batch_num": b_idx,
                "type": batch_type,
                "items": [
                    {"index": i, "string": s} for (i, s) in batch
                ]
            })
        return serialized

    # ---- CLASSIFY ----
    batches = [strings_to_classify[i:i+CLASSIFICATION_BATCH_SIZE]
               for i in range(0, len(strings_to_classify), CLASSIFICATION_BATCH_SIZE)]
    total_batches = len(batches)

    start = datetime.now()

    classification_progress = {"valid": 0,
                               "partial": 0,
                               "total": total_batches
                               }

    # Run classifications in parallel
    classification_tasks = []

    for idx, batch in enumerate(batches):
        classification_tasks.append(_classify_batch(
            batch, idx+1, total_batches, classification_progress=classification_progress))

    # all_classification_results = await asyncio.gather(*classification_tasks)

    results = []
    for coro in asyncio.as_completed(classification_tasks):
        res = await coro
        results.append(res)
    all_classification_results = results

    # Flatten list of lists into a single list
    final_classification_pairs = [item
                                  for sublist in all_classification_results
                                  for item in sublist]

    # ---------- RECOMBINE ----------
    final_results = [label for _, label in sorted(
        final_classification_pairs, key=lambda x: x[0])]

    classified = [(index, string, label)
                  for index, (string, label) in enumerate(zip(unique_texts, final_results))]

    end = datetime.now()
    print(f"Total time consumed for classification: {end-start}")

    business_items = [(i, s) for i, s, l in classified if (
        l.strip().lower()) == "business"]
    ordinary_items = [(i, s) for i, s, l in classified if (
        l.strip().lower()) == "ordinary"]

    print(
        f"[CLASSIFY] Business: {len(business_items)}, Ordinary: {len(ordinary_items)}")

    # ---- VOTE ----
    voting_batches = [classified[i:i+VOTING_BATCH_SIZE]
                      for i in range(0, len(classified), VOTING_BATCH_SIZE)]
    total_batches = len(voting_batches)

    start = datetime.now()

    voting_progress = {"valid": 0,
                       "partial": 0,
                       "total": total_batches
                       }

    # Run classifications in parallel
    voting_tasks = []

    for idx, batch in enumerate(voting_batches):
        voting_tasks.append(_voting_batch(
            batch, idx+1, total_batches, voting_progress=voting_progress))

    # all_classification_results = await asyncio.gather(*classification_tasks)

    results = []
    for coro in asyncio.as_completed(voting_tasks):
        res = await coro
        results.append(res)
    all_voting_results = results

    # Flatten list of lists into a single list
    final_voting_pairs = [item
                          for sublist in all_voting_results
                          for item in sublist]
    opposites = {"business": "ordinary", "ordinary": "business"}

    final_results = [vote for _, vote in sorted(
        final_voting_pairs, key=lambda x: x[0])]

    verified = []
    for (i, s, l), v in zip(classified, final_results):
        if v is False:
            l = opposites.get(l, l)
        verified.append((i, s, l))

    end = datetime.now()
    print(f"Total time consumed for voting: {end-start}")

    # STEP 2: Split into two groups, preserving index
    business_items = [(i, s) for i, s, l in verified if (
        l.strip().lower()) == "business"]
    ordinary_items = [(i, s) for i, s, l in verified if (
        l.strip().lower()) == "ordinary"]

    print(
        f"[CLASSIFY AFTER VOTING] Business: {len(business_items)}, Ordinary: {len(ordinary_items)}")

    # Split into translation batches
    business_batches = [business_items[i:i+TRANSLATION_BATCH_SIZE]
                        for i in range(0, len(business_items), TRANSLATION_BATCH_SIZE)]
    ordinary_batches = [ordinary_items[i:i+TRANSLATION_BATCH_SIZE]
                        for i in range(0, len(ordinary_items), TRANSLATION_BATCH_SIZE)]

    print(
        f"\nTotal {len(business_batches)} Business tasks are created, and {len(ordinary_batches)} Ordinary\n")

    business_serialized = serialize_batches(business_batches, "business")
    ordinary_serialized = serialize_batches(ordinary_batches, "ordinary")

    batches_data = {
        "business_batches": business_serialized,
        "ordinary_batches": ordinary_serialized
    }

    batches_file = os.path.join(
        BATCHES_DIR, f"batches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(batches_file, "w", encoding="utf-8") as f:
        json.dump(batches_data, f, ensure_ascii=False, indent=2)

    print(f"Saved batches to {batches_file}")

    translation_progress = {"valid": 0, "partial": 0, "total": len(
        business_batches) + len(ordinary_batches)}
    examples = qdrant_examples(shopDomain, target_lang, user_id)

    # Run translations in parallel
    start = datetime.now()
    translation_tasks = []
    logs = {}

    for idx, batch in enumerate(business_batches):
        translation_tasks.append(_translate_batch(batch, examples, user_id, shopDomain, target_lang, brand_tone, industry, idx+1,
                                                  len(business_batches), type="business", translation_progress=translation_progress, logs=logs))

    for idx, batch in enumerate(ordinary_batches):
        translation_tasks.append(_translate_batch(batch, examples, user_id, shopDomain, target_lang, brand_tone, industry, idx+1,
                                                  len(ordinary_batches), type="ordinary", translation_progress=translation_progress, logs=logs))

    random.shuffle(translation_tasks)

    # all_translation_results = await asyncio.gather(*translation_tasks)

    results = []
    for coro in asyncio.as_completed(translation_tasks):
        res = await coro
        results.append(res)
    all_translation_results = results

    with open(console_file, "a", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    # Flatten already-indexed results
    final_translation_pairs = [pair
                               for batch in all_translation_results
                               if batch is not None
                               for pair in batch]

    # ---------- RECOMBINE ----------
    final_results = [t for _, t in sorted(
        final_translation_pairs, key=lambda x: x[0])]
    for i in range(len(final_results)):
        if final_results[i] == locale:
            print(f"Locale not translated, converting manually")
            final_results[i] = target_lang
    print(f"Total strings retained after processing: {len(final_results)}")
    final_results = reconstruct_from_map(
        final_results, unique_texts, index_map, len(expanded))
    final_results = collapse_strings(final_results, mapping)
    end = datetime.now()
    print(f"Total time consumed for translation: {end-start}")

    comparative_file = os.path.join(LOG_DIR, "comparative.json")
    with open(comparative_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"path": path_str, "orig": orig, "trans": trans} for (
                path, orig_val, path_str), orig, trans in zip(positions, strings_to_translate, final_results)],
            f, ensure_ascii=False, indent=2
        )
    counter = 0
    for s, t in zip(strings_to_translate, final_results):
        if s == t:
            counter += 1
        else:
            continue
    print(
        f"Saved comparison strings to {comparative_file}, total {counter} strings are not translated, i.e. same as original.")

    # ---------- INJECTION ----------
    # # def set_value(d, path, value):
    #     ref = d
    #     for p in path[:-1]:
    #         ref = ref[p]
    #     ref[path[-1]] = value
    # def set_value_with_original(d, path, translated):
    #     ref = d
    #     for p in path[:-1]:
    #         ref = ref[p]
    #     ref[path[-1]] = value

    def set_value_with_original(d, path, translated, path_str):
        ref = d
        for p in path[:-1]:
            ref = ref[p]

        last_key = path[-1]
        original_value = ref[last_key]

        # Keep original
        prefixed_key = f"original{last_key[0].upper()}{last_key[1:]}"
        if prefixed_key not in ref:
            ref[prefixed_key] = original_value

        # Inject translation
        ref[last_key] = translated

        # Inject path_<field>
        path_key = f"path_{last_key}"
        ref[path_key] = path_str

        # Inject AI flag (success if translated != original)
        flag_key = f"aiTranslated_{last_key}"
        ref[flag_key] = (translated.strip() != original_value.strip())
        # Mark priority review if untranslatable
        priority_key = f"priorityReview_{last_key}"
        ref[priority_key] = (translated.strip() == original_value.strip())

    injected_log = []
    counter = 0
    # for i, translated in enumerate(final_results):
    #     path, orig_val, path_str = positions[i]
    #     set_value_with_original(target_data, path, translated)
    #     injected_log.append({"path": path_str, "translated": translated})
    #     counter += 1

    paths_array = []
    for i, translated in enumerate(final_results):
        path, orig_val, path_str = positions[i]
        set_value_with_original(target_data, path, translated, path_str)
        injected_log.append({
            "path": path_str,
            "original": orig_val,
            "translated": translated,
            "aiTranslated": translated.strip() != orig_val.strip()
        })
        paths_array.append(path_str)
        counter += 1
        if "locale" in path_str:
            if orig_val == locale and translated == target_lang:
                continue
            else:
                print(f"mismatching at index {i}")

    print(f"Total {counter} strings are injected")

    # ---- SAVE INJECTED ----
    injected_file = os.path.join(
        LOG_DIR, f"today_injected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(injected_file, "w", encoding="utf-8") as f:
        json.dump(injected_log, f, ensure_ascii=False, indent=2)
    print(f"Saved injected strings to {injected_file}")

    print(
        f" Injected {len(final_results)}/{len(strings_to_translate)} strings")
    save_report()

    # print("Celery task started...")
    # task = store_examples.delay(strings_to_translate, final_results,
    #                             paths_array, shopDomain, target_lang, brand_tone)  # type: ignore
    # print(f"New task ID: {task.id}")
    return target_data
