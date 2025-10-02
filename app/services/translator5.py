import json
import random
import re
import asyncio
import os
import sys
from datetime import datetime
import time
from openai import AsyncOpenAI
import google.generativeai as genai  # Gemini SDK
from ..config import settings


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

REPORT_DIR = os.path.join(LOG_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

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
openai_model_1 = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY_1,
    timeout=100.0
)

openai_model_2 = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY_2,
    timeout=100.0
)

genai.configure(api_key=settings.GEMINI_API_KEY_1)  # type: ignore
gemini_model_1 = genai.GenerativeModel("gemini-2.0-flash-lite")  # type: ignore # gemini-1.5-flash

# genai.configure(api_key=settings.GEMINI_API_KEY_2)  # type: ignore
# gemini_model_2 = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore

# ==== CONFIG ====
BATCH_SIZE = 50
MAX_CONCURRENCY_CLASSIFICATION = 6
semaphore_classification = asyncio.Semaphore(MAX_CONCURRENCY_CLASSIFICATION)
# classification_time = []
MAX_CONCURRENCY_TRANSLATION = 6
semaphore_translation = asyncio.Semaphore(MAX_CONCURRENCY_TRANSLATION)
# translation_time = []

classification_model_cycle = ["openai1", "openai2", "gemini1"]
model_index_classify = 0

translation_model_cycle = ["openai1", "openai2", "gemini1"]
model_index_translation = 0

sys.setrecursionlimit(3000)

# ==== LOG DIR ====
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


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


async def with_retry(fn, *args, provider=None, retries=3, **kwargs):
    for i in range(retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if "Rate limit" in str(e) or "quota" in str(e).lower():
                wait = (2 ** i) + random.random()
                msg = f"⚠ Rate limit for {provider}, retrying in {wait:.2f}s..."
                print(msg)
                TRANSLATION_STATS["rate_limits"]["total"].append(
                    {"provider": provider, "time": datetime.now().isoformat(), "wait": wait})
                TRANSLATION_STATS["rate_limits"][provider].append(
                    {"time": datetime.now().isoformat(), "wait": wait})
                await asyncio.sleep(wait)
            else:
                print(f"⚠ Error in {provider}: {e}, retrying...")
                await asyncio.sleep(2)
    raise Exception(f"Max retries reached for {provider}")


# ===================== CLASSIFICATION FUNCTIONS =====================

async def _classify_openai(strings_batch, batch_num, total_batches, classification_model):
    """
    Classify strings into 'business' or 'ordinary'.
    """

    MAX_CHARS_PER_CLASSIFY = 50_000  # conservative
    total_chars = sum(len(s) for s in strings_batch)
    if total_chars > MAX_CHARS_PER_CLASSIFY and len(strings_batch) > 1:
        # split into two equal parts
        mid = len(strings_batch) // 2
        left = await _classify_openai(strings_batch[:mid], batch_num, total_batches, classification_model)
        right = await _classify_openai(strings_batch[mid:], batch_num, total_batches, classification_model)
        return left + right

    prompt = f"""
    You are a strict text classifier.

    Categories:
    - "business" = official, legal, contractual, financial, invoices, policies, compliance, formal system messages.
    - "ordinary" = product marketing, casual phrases, blogs, general UI text, everyday communication.
    Never invent new categories; only use "business" or "ordinary".

    Rules:
    - Classify each string into exactly ONE category.
    - The number of output labels MUST equal the number of input strings ({len(strings_batch)}).
    - Keep the order of outputs identical to the order of inputs.

    Examples:
    Input: ["Invoice #4533", "Big summer sale!", "Refunds will be processed within 7 days", "Sign In"]
    Output: ["business", "ordinary", "business", "ordinary"]

    Input: ["Terms and Conditions apply", "Export License Required", "Check out our new arrivals", "Best quality leather shoes"]
    Output: ["business", "business", "ordinary", "ordinary"]

    Now classify these {len(strings_batch)} strings:
    {json.dumps(strings_batch, ensure_ascii=False)}

    IMPORTANT:
    Respond with ONLY a valid JSON array of {len(strings_batch)} strings. No extra text.
    """

    resp = await classification_model.chat.completions.create(
        model="gpt-4.1-mini", # gpt-4o-mini # gpt-4.1-mini
        messages=[{"role": "user", "content": prompt}],  # type: ignore
        temperature=0.7,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "classification_labels",
                "schema": {
                    "type": "object",
                    "properties": {
                        "classified_labels": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["business", "ordinary"]},
                            "minItems": len(strings_batch),
                            "maxItems": len(strings_batch)
                        }
                    },
                    "required": ["classified_labels"],
                    "additionalProperties": False
                },
            },
        }  # type: ignore
    )

    labels_text: str = resp.choices[0].message.content  # type: ignore
    # print(f"[DEBUG] Classify batch {batch_num}/{total_batches} → {len(strings_batch)} strings, chars={total_chars}")
    # Ensure only valid labels
    # VALID_LABELS = {"ordinary", "business"}
    try:
        data = json.loads(labels_text)
        if isinstance(data, dict) and "classified_labels" in data:
            return [clean_line(x.lower()) for x in data["classified_labels"]]
        elif isinstance(data, list):
            return [clean_line(x.lower()) for x in data]
        else:
            raise ValueError("Unexpected OpenAI response")
        # labels = data.get("classified_labels", [])
        # labels = [l if l in VALID_LABELS else "ordinary" for l in labels]
        # return labels
    except Exception as e:
        print(f"⚠ Parse fallback: {e}")
        return [line.strip().lower() for line in labels_text.split("\n") if line.strip()]
        # labels = [l if l in VALID_LABELS else "ordinary" for l in labels]
        # return labels
    
async def classify_openai_1(strings_batch, batch_num, total_batches):
    return await _classify_openai(strings_batch, batch_num, total_batches, openai_model_1)

async def classify_openai_2(strings_batch, batch_num, total_batches):
    return await _classify_openai(strings_batch, batch_num, total_batches, openai_model_2)

async def _classify_gemini(strings_batch, batch_num, total_batches, model):
    """
    Classify strings into 'business' or 'ordinary'.
    """

    MAX_CHARS_PER_CLASSIFY = 50_000  # conservative
    total_chars = sum(len(s) for s in strings_batch)
    if total_chars > MAX_CHARS_PER_CLASSIFY and len(strings_batch) > 1:
        # split into two equal parts
        mid = len(strings_batch) // 2
        left = await _classify_openai(strings_batch[:mid], batch_num, total_batches, model)
        right = await _classify_openai(strings_batch[mid:], batch_num, total_batches, model)
        return left + right

    prompt = f"""
    You are a strict text classifier.

    Categories:
    - "business" = official, legal, contractual, financial, invoices, policies, compliance, formal system messages.
    - "ordinary" = product marketing, casual phrases, blogs, general UI text, everyday communication.
    Never invent new categories; only use "business" or "ordinary".

    Rules:
    - Classify each string into exactly ONE category.
    - The number of output labels MUST equal the number of input strings ({len(strings_batch)}).
    - Keep the order of outputs identical to the order of inputs.

    Examples:
    Input: ["Invoice #4533", "Big summer sale!", "Refunds will be processed within 7 days", "Sign In"]
    Output: ["business", "ordinary", "business", "ordinary"]

    Input: ["Terms and Conditions apply", "Export License Required", "Check out our new arrivals", "Best quality leather shoes"]
    Output: ["business", "business", "ordinary", "ordinary"]

    Now classify these {len(strings_batch)} strings:
    {json.dumps(strings_batch, ensure_ascii=False)}

    IMPORTANT:
    Respond with ONLY a valid JSON array of {len(strings_batch)} strings. No extra text.
    """
    
    # resp = await model.generate_content_async(prompt, generation_config={"response_mime_type": "application/json"})
    resp = await asyncio.to_thread(
        model.generate_content,
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    labels_text = resp.text or "[]"
    try:
        data = json.loads(labels_text)
        if isinstance(data, dict) and "classified_labels" in data:
            return [clean_line(x.lower()) for x in data["classified_labels"]]
        elif isinstance(data, list):
            return [clean_line(x.lower()) for x in data]
        else:
            raise ValueError("Unexpected OpenAI response")
        # labels = data.get("classified_labels", [])
        # labels = [l if l in VALID_LABELS else "ordinary" for l in labels]
        # return labels
    except Exception as e:
        print(f"⚠ Parse fallback: {e}")
        return [line.strip().lower() for line in labels_text.split("\n") if line.strip()]
        # labels = [l if l in VALID_LABELS else "ordinary" for l in labels]
        # return labels

async def classify_gemini_1(strings_batch, batch_num, total_batches):
    return await _classify_gemini(strings_batch, batch_num, total_batches, gemini_model_1)


# ===================== BATCH CLASSIFICATION =====================

async def _classify_batch(indexed_strings, batch_num, total_batches, classification_progress=None):
    global model_index_classify
    # global classification_end
    # classification_time.append(datetime.now())
    strings = [s for _, s in indexed_strings]
    classification_model_cycle = ["openai1", "openai2", "gemini1"]
    VALID_LABELS = {"ordinary", "business"}
    raw_ordinary = ["ordinary"] * len(strings)
    async with semaphore_classification:
        print(f"\n[DEBUG] Batch {batch_num}/{total_batches} → {len(strings)} strings ")
        current_model = classification_model_cycle[model_index_classify % len(classification_model_cycle)]
        model_index_classify += 1

        try:
            if current_model == "openai1":
                result = await with_retry(classify_openai_1, strings, batch_num, total_batches)
            elif current_model == "openai2":
                result = await with_retry(classify_openai_2, strings, batch_num, total_batches)
            else: # gemini1
                result = await with_retry(classify_gemini_1, strings, batch_num, total_batches)
            
            labels = []
            for l in result:
                if l in VALID_LABELS:
                    labels.append(l)

        except Exception as e:
            if classification_progress is not None:
                classification_progress["partial"] += 1
                print(f"[CLASSIFICATION PROGRESS: failed] {classification_progress['valid']} valid, {classification_progress['partial']} partial, total {classification_progress['valid']+classification_progress['partial']}/{classification_progress['total']} (batch {batch_num} via {current_model})")
            # classification_end = datetime.now()
            return [(i, l) for (i, _), l in zip(indexed_strings, raw_ordinary)]

        if labels:
            expected = len(strings)
            got = len(labels)

            if expected == got:
                if classification_progress is not None:
                    classification_progress["valid"] += 1
                    print(f"[CLASSIFICATION PROGRESS: valid] {classification_progress['valid']} valid, {classification_progress['partial']} partial, total {classification_progress['valid']+classification_progress['partial']}/{classification_progress['total']} (batch {batch_num} via {current_model})")
                # classification_end = datetime.now()
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
            # classification_end = datetime.now()
            return [(i, l) for (i, _), l in zip(indexed_strings, labels)]
        

# ===================== TRANSLATION FUNCTIONS =====================

async def _translate_openai(strings, target_lang, brand_tone, model):
    prompt = f"""
    You are a professional translator.

    Task:
    Translate the following {len(strings)} strings into {target_lang}.
    - Maintain the brand tone as '{brand_tone}'.
    - If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags unchanged, only translate the inner text.
    - Preserve placeholders (e.g., {{name}}, %s, {{0}}) exactly as they are. Translate surrounding text but do NOT translate or modify the text inside placeholders.
    - Do NOT merge, omit, or add strings.
    - Do not summarize, simplify, or shorten long texts (e.g., Privacy Policies, Terms & Conditions). Translate them fully.
    - Special rule for language codes:
    If a string is a language code such as "en", replace it with the correct code for {target_lang}.
    Example: "en" → "fr" when {target_lang} is French.

    Output requirements:
    - Return ONLY a valid JSON array.
    - The array must contain exactly {len(strings)} items.
    - Order of items must match the input order.
    - Each output item must be a string.

    Input strings:
    {json.dumps(strings, ensure_ascii=False)}

    Output format (strict):
    [
    "translation of string 1",
    "translation of string 2",
    ...
    ]
"""

    resp = await model.chat.completions.create(
        model="gpt-4.1-mini", # gpt-4o-mini # gpt-4.1-mini
        messages=[{"role": "user", "content": prompt}],  # type: ignore
        temperature=0.7,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "translation_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "translations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Translated strings in the same order as input."
                        }
                    },
                    "required": ["translations"],
                    "additionalProperties": False,
                },
            },
        }  # type: ignore
    )

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)  # type: ignore
        if isinstance(data, dict) and "translations" in data:
            return [clean_line(x) for x in data["translations"]]
        elif isinstance(data, list):
            return [clean_line(x) for x in data]
        else:
            raise ValueError("Unexpected OpenAI response")
    except Exception as e:
        print(f"⚠ Parse fallback: {e}")
        # type: ignore
        return [clean_line(line) for line in content.split("\n") if line.strip()]


async def translate_openai_1(strings, target_lang, brand_tone):
    return await _translate_openai(strings, target_lang, brand_tone, openai_model_1)


async def translate_openai_2(strings, target_lang, brand_tone):
    return await _translate_openai(strings, target_lang, brand_tone, openai_model_2)


async def _translate_gemini(strings, target_lang, brand_tone, model):
    prompt = f"""
    You are a professional translator.

    Task:
    Translate the following {len(strings)} strings into {target_lang}.
    - Maintain the brand tone as '{brand_tone}'.
    - If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags unchanged, only translate the inner text.
    - Preserve placeholders (e.g., {{name}}, %s, {{0}}) exactly as they are. Translate surrounding text but do NOT translate or modify the text inside placeholders.
    - Do NOT merge, omit, or add strings.
    - Translate long texts fully (no summarization).
    - Language code rule: if a string is a language code (e.g., "en"), replace it with the correct code for {target_lang}.
    Example: "en" → "fr" when {target_lang} is French.

    Output requirements:
    - Return ONLY valid JSON.
    - JSON must be an array of exactly {len(strings)} strings.
    - Order must match the input order.
    - No comments, no explanations, no extra text.

    Input strings:
    {json.dumps(strings, ensure_ascii=False)}

    Output format (strict):
    [
      "translation of string 1",
      "translation of string 2",
      ...
    ]
    """
    # resp = await model.generate_content_async(prompt, generation_config={"response_mime_type": "application/json"})
    resp = await asyncio.to_thread(
        model.generate_content,
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    text = resp.text or "[]"
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "translations" in data:
            return [clean_line(x) for x in data["translations"]]
        elif isinstance(data, list):
            return [clean_line(x) for x in data]
        else:
            raise ValueError("Unexpected Gemini response")
    except Exception as e:
        print(f"⚠ Parse fallback: {e}")
        return [clean_line(line) for line in text.split("\n") if line.strip()]


async def translate_gemini_1(strings, target_lang, brand_tone):
    return await _translate_gemini(strings, target_lang, brand_tone, gemini_model_1)


# async def translate_gemini_2(strings, target_lang, brand_tone):
#     return await _translate_gemini(strings, target_lang, brand_tone, gemini_model_2)


# ===================== BATCH TRANSLATION =====================

async def _translate_batch(indexed_strings, target_lang, brand_tone, batch_num, type, translation_progress=None):
    global model_index_translation
    # global translation_end
    # translation_time.append(datetime.now())
    strings = [s for _, s in indexed_strings]
    # best_provider = None
    # best_translation = None
    # best_score = float("inf")

    # Define provider order
    translation_model_cycle = ["openai1", "openai2", "gemini1"] if type == "business" else [
        "gemini1", "openai1", "openai2"]
    models_used = []

    async with semaphore_translation:
        print(f"\n[DEBUG] {type.upper()} Batch {batch_num} → {len(strings)} strings ")
        current_model = translation_model_cycle[model_index_translation % len(translation_model_cycle)]
        model_index_translation += 1

        try:
            start = time.time()
            if current_model == "openai1":
                translations = await with_retry(translate_openai_1, strings, target_lang, brand_tone, provider=current_model)
            elif current_model == "openai2":
                translations = await with_retry(translate_openai_2, strings, target_lang, brand_tone, provider=current_model)
            else:
                translations = await with_retry(translate_gemini_1, strings, target_lang, brand_tone, provider=current_model)

            elapsed = time.time() - start
            # rough estimate if API doesn’t return usage
            tokens_used = len(" ".join(strings)) // 4
            TRANSLATION_STATS["tokens"][current_model].append(
                {"tokens": tokens_used, "time": elapsed})
            models_used.append(current_model)

        except Exception as e:
            print(f"⚠ {current_model} failed, falling back: {e}")
            for alt in translation_model_cycle:
                current_model = alt
                if current_model in models_used:
                    continue
                try:
                    start = time.time()
                    if current_model == "openai1":
                        translations = await translate_openai_1(strings, target_lang, brand_tone)
                    elif current_model == "openai2":
                        translations = await translate_openai_2(strings, target_lang, brand_tone)
                    else:
                        translations = await translate_gemini_1(strings, target_lang, brand_tone)

                    models_used.append(current_model)

                    elapsed = time.time() - start
                    # rough estimate if API doesn’t return usage
                    tokens_used = len(" ".join(strings)) // 4
                    TRANSLATION_STATS["tokens"][current_model].append(
                        {"tokens": tokens_used, "time": elapsed})
                    break
                except Exception as e2:
                    print(f"⚠ Fallback {current_model} also failed: {e2}")
            else:
                raise Exception("All providers failed!")
            
        if translations:
            expected = len(strings)
            got = len(translations)

            if expected == got:
                if translation_progress is not None:
                        translation_progress["valid"] += 1
                        print(f"[TRANSLATION PROGRESS: valid] {translation_progress['valid']} valid, {translation_progress['partial']} partial, total {translation_progress['valid']+translation_progress['partial']} ({type} batch via {current_model})")
                # translation_end = datetime.now()
                return [(i, t) for (i, _), t in zip(indexed_strings, translations)]

            # --- FIX: force align translations ---
            elif got < expected:
                # Pad missing with originals
                print(f"Expected {expected}, got {got} -> Padding {expected-got} from original batch")
                translations.extend(strings[got:])
            else: # got > expected
                # Trim extras
                translations = translations[:expected]
                print(f"Expected {expected}, got {got} -> Truncating {got-expected} from translation batch")

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
                print(f"[TRANSLATION PROGRESS: partial] {translation_progress['valid']} valid, {translation_progress['partial']} partial, total {translation_progress['valid']+translation_progress['partial']} ({type} batch via {current_model})")
            # translation_end = datetime.now()
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
async def fast_translate_json(target_data, target_lang, brand_tone):
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
                elif k == "translatableContent" and isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            for field in ["value", "locale"]:
                                if field in item and isinstance(item[field], str) and is_translateable(item[field]):
                                    positions.append(
                                        (path + [k, i, field], item[field],
                                         ".".join(map(str, path + [k, i, field])))
                                    )

                else:
                    collect_strings(v, path + [k], k)

        elif isinstance(d, list):
            for i, item in enumerate(d):
                collect_strings(item, path + [i], parent_key)


    collect_strings(target_data)

    strings_to_translate = [s for _, s, _ in positions]
    print(f"Total strings: {len(strings_to_translate)}")

    counter = 0
    for line in strings_to_translate:
        words = len(line.split(" "))
        counter += words
    
    print(f"Total words in a string: {counter}")

    batches = [list(enumerate(strings_to_translate[i:i+BATCH_SIZE], i))
    for i in range(0, len(strings_to_translate), BATCH_SIZE)]


    classification_progress = {"valid": 0, "partial": 0, "total": len(batches)}
    translation_progress = {"valid": 0, "partial": 0, "total": len(batches)}
    batch_num = translation_progress["valid"] + translation_progress["partial"]


    classification_tasks = [_classify_batch(batch, idx+1, len(batches), classification_progress) for idx, batch in enumerate(batches)]


    business_buffer, ordinary_buffer = [], []
    translation_tasks = []
    start = datetime.now()


    for coro in asyncio.as_completed(classification_tasks):
        results = await coro
        for i, l in results: # type: ignore
            s = strings_to_translate[i]
            if l == "business":
                business_buffer.append((i, s))
            else:
                ordinary_buffer.append((i, s))


        while len(business_buffer) >= BATCH_SIZE:
            batch = business_buffer[:BATCH_SIZE]
            business_buffer = business_buffer[BATCH_SIZE:]
            translation_tasks.append(asyncio.create_task(
            _translate_batch(batch, target_lang, brand_tone, batch_num+1, "business", translation_progress)
            ))


        while len(ordinary_buffer) >= BATCH_SIZE:
            batch = ordinary_buffer[:BATCH_SIZE]
            ordinary_buffer = ordinary_buffer[BATCH_SIZE:]
            translation_tasks.append(asyncio.create_task(
            _translate_batch(batch, target_lang, brand_tone, batch_num+1, "ordinary", translation_progress)
            ))


    if business_buffer:
        batch = business_buffer
        business_buffer = []
        translation_tasks.append(asyncio.create_task(
        _translate_batch(business_buffer, target_lang, brand_tone, batch_num+1, "business", translation_progress)
        ))
    if ordinary_buffer:
        batch = ordinary_buffer
        ordinary_buffer = []
        translation_tasks.append(asyncio.create_task(
        _translate_batch(ordinary_buffer, target_lang, brand_tone, batch_num+1, "ordinary", translation_progress)
        ))


    results = []
    for coro in asyncio.as_completed(translation_tasks):
        batch = await coro
        if batch is not None:
            results.extend(batch)
    # results = await asyncio.gather(*translation_tasks)

    final_translation_pairs = results
    final_results = [t for _, t in sorted(final_translation_pairs, key=lambda x: x[0])]


    print()
    print(f"Total time used for classification and translation: {datetime.now()-start}")
    print()

    comparative_file = os.path.join(LOG_DIR, "comparative.json")
    with open(comparative_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"path": path_str, "orig": orig, "trans": trans} for (
                path, orig_val, path_str), orig, trans in zip(positions, strings_to_translate, final_results)],
            f, ensure_ascii=False, indent=2
        )
    print(f"Saved comparison strings to {comparative_file}")

    # ---------- INJECTION ----------
    # def set_value(d, path, value):
    #     ref = d
    #     for p in path[:-1]:
    #         ref = ref[p]
    #     ref[path[-1]] = value

    def set_value_with_original(d, path, translated):
        ref = d
        for p in path[:-1]:
            ref = ref[p]
        last_key = path[-1]
        original_value = ref[last_key]
        prefixed_key = f"original{last_key[0].upper()}{last_key[1:]}"
        if prefixed_key not in ref:
            ref[prefixed_key] = original_value
        ref[last_key] = translated

    injected_log = []
    counter = 0
    for i, translated in enumerate(final_results):
        path, orig_val, path_str = positions[i]
        set_value_with_original(target_data, path, translated)
        injected_log.append({"path": path_str, "translated": translated})
        counter += 1

    print(f"Total {counter} strings are injected")

    # ---- SAVE INJECTED ----
    injected_file = os.path.join(
        LOG_DIR, f"injected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(injected_file, "w", encoding="utf-8") as f:
        json.dump(injected_log, f, ensure_ascii=False, indent=2)
    print(f"Saved injected strings to {injected_file}")

    print(
        f" Injected {len(final_results)}/{len(strings_to_translate)} strings")
    save_report()
    return target_data
