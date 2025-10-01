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

CONSOLE_DIR = os.path.join(LOG_DIR, "console")
os.makedirs(CONSOLE_DIR, exist_ok=True)
console_file = os.path.join(CONSOLE_DIR, "console.json")

raw_logs = [{"message" : "Logs"}]
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
openai_model_1 = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY_1,
    timeout=150.0
)

openai_model_2 = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY_2,
    timeout=150.0
)

genai.configure(api_key=settings.GEMINI_API_KEY_1)  # type: ignore
gemini_model_1 = genai.GenerativeModel("gemini-2.0-flash-lite")  # type: ignore # gemini-1.5-flash

# genai.configure(api_key=settings.GEMINI_API_KEY_2)  # type: ignore
# gemini_model_2 = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore

# ==== CONFIG ====
CLASSIFICATION_BATCH_SIZE = 50
MAX_CONCURRENCY_CLASSIFICATION = 20
semaphore_classification = asyncio.Semaphore(MAX_CONCURRENCY_CLASSIFICATION)
TRANSLATION_BATCH_SIZE = 30
MAX_CONCURRENCY_TRANSLATION = 20
semaphore_translation = asyncio.Semaphore(MAX_CONCURRENCY_TRANSLATION)

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


async def with_retry(fn, *args, retries=3, **kwargs):
    for i in range(retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            provider = kwargs.get("provider", "unknown")
            batch_num = kwargs.get("batch_num", 0)
            typ = kwargs.get("type", "N/A")
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
                print(f"⚠ Error in, batch {type} {batch_num}, {provider}: {e}, retrying...")
                await asyncio.sleep(2)
    raise Exception(f"Max retries reached for {kwargs.get('provider')}")


# ===================== CLASSIFICATION FUNCTIONS =====================

async def _classify_openai(strings_batch, batch_num, total_batches, classification_model, provider, type):
    """
    Classify strings into 'business' or 'ordinary'.
    """

    MAX_CHARS_PER_CLASSIFY = 50_000  # conservative
    total_chars = sum(len(s) for s in strings_batch)
    if total_chars > MAX_CHARS_PER_CLASSIFY and len(strings_batch) > 1:
        # split into two equal parts
        print(f"{total_chars//4} total tokens, Splitting batch due to large size")
        mid = len(strings_batch) // 2
        left = await _classify_openai(strings_batch[:mid], batch_num, total_batches, classification_model, provider, type)
        right = await _classify_openai(strings_batch[mid:], batch_num, total_batches, classification_model, provider, type)
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
        print(f"⚠ Parse fallback {provider} for {type} batch {batch_num}: {e}")
        return [line.strip().lower() for line in labels_text.split("\n") if line.strip()]
        # labels = [l if l in VALID_LABELS else "ordinary" for l in labels]
        # return labels
    
async def classify_openai_1(strings_batch, batch_num, total_batches, provider, type):
    return await _classify_openai(strings_batch, batch_num, total_batches, openai_model_1, provider, type)

async def classify_openai_2(strings_batch, batch_num, total_batches, provider, type):
    return await _classify_openai(strings_batch, batch_num, total_batches, openai_model_2, provider, type)

async def _classify_gemini(strings_batch, batch_num, total_batches, classification_model, provider, type):
    """
    Classify strings into 'business' or 'ordinary'.
    """

    MAX_CHARS_PER_CLASSIFY = 50_000  # conservative
    total_chars = sum(len(s) for s in strings_batch)
    if total_chars > MAX_CHARS_PER_CLASSIFY and len(strings_batch) > 1:
        # split into two equal parts
        print(f"{total_chars//4} total tokens, Splitting batch due to large size")
        mid = len(strings_batch) // 2
        left = await _classify_gemini(strings_batch[:mid], batch_num, total_batches, classification_model, provider, type)
        right = await _classify_gemini(strings_batch[mid:], batch_num, total_batches, classification_model, provider, type)
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
    # resp = await classification_model.generate_content_async(prompt, generation_config={"response_mime_type": "application/json"})
    resp = await asyncio.to_thread(
        classification_model.generate_content,
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
        print(f"⚠ Parse fallback {provider} for {type} batch {batch_num}: {e}")
        return [line.strip().lower() for line in labels_text.split("\n") if line.strip()]
        # labels = [l if l in VALID_LABELS else "ordinary" for l in labels]
        # return labels

async def classify_gemini_1(strings_batch, batch_num, total_batches, provider, type):
    return await _classify_gemini(strings_batch, batch_num, total_batches, gemini_model_1, provider, type)

# ===================== BATCH CLASSIFICATION =====================

async def _classify_batch(indexed_strings, batch_num, total_batches, classification_progress=None):
    global model_index_classify
    strings = [s for _, s in indexed_strings]
    classification_model_cycle = ["openai1", "openai2", "gemini1"]
    VALID_LABELS = {"ordinary", "business"}
    raw_ordinary = ["ordinary" for i in range(50)]
    async with semaphore_classification:
        current_model = classification_model_cycle[model_index_classify % len(classification_model_cycle)]
        model_index_classify += 1
        print(f"\n[DEBUG] Batch {batch_num}/{total_batches} via {current_model} → {len(strings)} strings ")

        try:
            if current_model == "openai1":
                result = await with_retry(classify_openai_1, strings, batch_num, total_batches, provider=current_model, type=type)
            elif current_model == "openai2":
                result = await with_retry(classify_openai_2, strings, batch_num, total_batches, provider=current_model, type=type)
            else: # gemini1
                result = await with_retry(classify_gemini_1, strings, batch_num, total_batches, provider=current_model, type=type)
            
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
        

# ===================== TRANSLATION FUNCTIONS =====================

async def _translate_openai(strings, target_lang, brand_tone, model, batch_num, type, provider):
    MAX_CHARS_PER_TRANSLATE = 20_000  # conservative
    total_chars = sum(len(s) for s in strings)
    if total_chars > MAX_CHARS_PER_TRANSLATE:
        # split into two equal parts
        if len(strings) > 1:
            print(f"{total_chars} total characters, Splitting {type} batch {batch_num} due to large size, using {provider}")
            mid = len(strings) // 2
            left = await _translate_openai(strings[:mid], target_lang, brand_tone, model, batch_num, type, provider)
            right = await _translate_openai(strings[mid:], target_lang, brand_tone, model, batch_num, type, provider)
            return left + right
        else:
            print(f"{total_chars} total characters, Splitting string of {type} batch {batch_num} due to large size, using {provider}")
            splitted_strings = strings[0].split(".")
            mid = len(strings) // 2
            left = await _translate_openai(splitted_strings[:mid], target_lang, brand_tone, model, batch_num, type, provider)
            right = await _translate_openai(splitted_strings[mid:], target_lang, brand_tone, model, batch_num, type, provider)
            result = left + right
            return ["".join(result)]
    
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
        print(f"⚠ Parse fallback {provider} for {type} batch {batch_num}: {e}")
        # type: ignore
        return [clean_line(line) for line in content.split("\n") if line.strip()]


async def translate_openai_1(strings, target_lang, brand_tone, batch_num, type, provider):
    return await _translate_openai(strings, target_lang, brand_tone, openai_model_1, batch_num, type, provider)


async def translate_openai_2(strings, target_lang, brand_tone, batch_num, type, provider):
    return await _translate_openai(strings, target_lang, brand_tone, openai_model_2, batch_num, type, provider)


async def _translate_gemini(strings, target_lang, brand_tone, model, batch_num, type, provider):
    MAX_CHARS_PER_TRANSLATE = 20_000  # conservative
    total_chars = sum(len(s) for s in strings)
    if total_chars > MAX_CHARS_PER_TRANSLATE:
        # split into two equal parts
        if len(strings) > 1:
            print(f"{total_chars} total characters, Splitting {type} batch {batch_num} due to large size, using {provider}")
            mid = len(strings) // 2
            left = await _translate_gemini(strings[:mid], target_lang, brand_tone, model, batch_num, type, provider)
            right = await _translate_gemini(strings[mid:], target_lang, brand_tone, model, batch_num, type, provider)
            return left + right
        else:
            print(f"{total_chars} total characters, Splitting string of {type} batch {batch_num} due to large size, using {provider}")
            splitted_strings = strings[0].split(".")
            mid = len(strings) // 2
            left = await _translate_gemini(splitted_strings[:mid], target_lang, brand_tone, model, batch_num, type, provider)
            right = await _translate_gemini(splitted_strings[mid:], target_lang, brand_tone, model, batch_num, type, provider)
            result = left + right
            return ["".join(result)]
    
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
        print(f"⚠ Parse fallback {provider} for {type} batch {batch_num}: {e}")
        return [clean_line(line) for line in text.split("\n") if line.strip()]


async def translate_gemini_1(strings, target_lang, brand_tone, batch_num, type, provider):
    return await _translate_gemini(strings, target_lang, brand_tone, gemini_model_1, batch_num, type, provider)


# async def translate_gemini_2(strings, target_lang, brand_tone):
#     return await _translate_gemini(strings, target_lang, brand_tone, gemini_model_2)


# ===================== BATCH TRANSLATION =====================

async def _translate_batch(indexed_strings, target_lang, brand_tone, batch_num, total_batches, type, translation_progress=None, logs={}):
    global model_index_translation
    strings = [s for _, s in indexed_strings]
    logs[f"{type}_{batch_num}"] = {}
    # best_provider = None
    # best_translation = None
    # best_score = float("inf")

    # Define provider order
    translation_model_cycle = ["openai1", "openai2", "gemini1"] if type == "business" else [
        "gemini1", "openai1", "openai2"]
    models_used = []

    async with semaphore_translation:
        current_model = translation_model_cycle[model_index_translation % len(translation_model_cycle)]
        model_index_translation += 1
        logs[f"{type}_{batch_num}"]["print"] = f"\n[DEBUG] {type.upper()} Batch {batch_num}/{total_batches} via {current_model} → {len(strings)} strings"
        print(logs[f"{type}_{batch_num}"]["print"])

        try:
            start = time.time()
            if current_model == "openai1":
                translations = await with_retry(translate_openai_1, strings, target_lang, brand_tone, batch_num, provider=current_model, type=type)
            elif current_model == "openai2":
                translations = await with_retry(translate_openai_2, strings, target_lang, brand_tone, batch_num, provider=current_model, type=type)
            else:
                translations = await with_retry(translate_gemini_1, strings, target_lang, brand_tone, batch_num, provider=current_model, type=type)

            elapsed = time.time() - start
            # rough estimate if API doesn’t return usage
            tokens_used = len(" ".join(strings)) // 4
            TRANSLATION_STATS["tokens"][current_model].append(
                {"tokens": tokens_used, "time": elapsed})
            models_used.append(current_model)

        except Exception as e:
            logs[f"{type}_{batch_num}"]["exc1"] = f"⚠ {current_model} failed, falling back: {e}"
            print(logs[f"{type}_{batch_num}"]["exc1"])
            for alt in translation_model_cycle:
                current_model = alt
                if current_model in models_used:
                    continue
                try:
                    start = time.time()
                    if current_model == "openai1":
                        translations = await translate_openai_1(strings, target_lang, brand_tone, batch_num, type, provider=current_model)
                    elif current_model == "openai2":
                        translations = await translate_openai_2(strings, target_lang, brand_tone, batch_num, type, provider=current_model)
                    else:
                        translations = await translate_gemini_1(strings, target_lang, brand_tone, batch_num, type, provider=current_model)

                    models_used.append(current_model)

                    elapsed = time.time() - start
                    # rough estimate if API doesn’t return usage
                    tokens_used = len(" ".join(strings)) // 4
                    TRANSLATION_STATS["tokens"][current_model].append(
                        {"tokens": tokens_used, "time": elapsed})
                    break
                except Exception as e2:
                    logs[f"{type}_{batch_num}"]["exc2"] = f"⚠ Fallback {current_model} also failed: {e2}"
                    print(logs[f"{type}_{batch_num}"]["exc2"])
            else:
                raise Exception("All providers failed!")
            
        if translations:
            expected = len(strings)
            got = len(translations)

            if expected == got:
                if translation_progress is not None:
                        translation_progress["valid"] += 1
                        logs[f"{type}_{batch_num}"]["valid"] = f"[TRANSLATION PROGRESS: valid] {translation_progress['valid']} valid, {translation_progress['partial']} partial, total {translation_progress['valid']+translation_progress['partial']}/{translation_progress['total']} ({type} batch {batch_num} via {current_model})"
                        print(logs[f"{type}_{batch_num}"]["valid"])
                return [(i, t) for (i, _), t in zip(indexed_strings, translations)]

            # --- FIX: force align translations ---
            elif got < expected:
                # Pad missing with originals
                logs[f"{type}_{batch_num}"]["padding"] = f"Expected {expected}, got {got} -> Padding {expected-got} from original batch"
                print(logs[f"{type}_{batch_num}"]["padding"])
                translations.extend(strings[got:])
            else: # got > expected
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
                logs[f"{type}_{batch_num}"]["partial"] = f"[TRANSLATION PROGRESS: partial] {translation_progress['valid']} valid, {translation_progress['partial']} partial, total {translation_progress['valid']+translation_progress['partial']}/{translation_progress['total']} ({type} batch {batch_num} via {current_model})"
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

    # Splittter for strings using regex
    def split_into_chunks(text, max_len=3000):
        """
        Split text into chunks of <= max_len, trying to split at '.' boundaries.
        """
        sentences = re.split(r'(?<=[.?!])\s+', text)  # split by sentence enders
        chunks, current = [], ""

        for sentence in sentences:
            # if adding sentence exceeds max_len → push current chunk
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
    def expand_strings(strings, max_len=3000):
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
                print(f"Splitting string on index {idx} into {len(chunks)} parts")
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
            print(f"String at index {idx} recombined by joining {count} strings")
        return collapsed
    

    collect_strings(target_data)

    strings_to_translate = [s for _, s, _ in positions]
    print(f"Total strings: {len(strings_to_translate)}")

    counter = 0
    for line in strings_to_translate:
        words = len(line.split(" "))
        counter += words
    
    print(f"Total words in a string: {counter}")

    
    expanded, mapping = expand_strings(strings_to_translate, max_len=5000)
    strings_to_classify = [(i, s) for i, s in enumerate(expanded)]

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

    # ---- TRANSLATE ----
    batches = [strings_to_classify[i:i+CLASSIFICATION_BATCH_SIZE]
               for i in range(0, len(strings_to_classify), CLASSIFICATION_BATCH_SIZE)]
    total_batches = len(batches)

    start = datetime.now()

    classification_progress = {"valid": 0, "partial": 0, "total": total_batches}

    # Run classifications in parallel
    classification_tasks = []
    
    for idx, batch in enumerate(batches):
        classification_tasks.append(_classify_batch(batch, idx+1, total_batches, classification_progress=classification_progress))
    
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
    final_results = [l for _, l in sorted(final_classification_pairs, key=lambda x: x[0])]

    classified = [(i, s, l)
                  for i, (s, l) in enumerate(zip(expanded, final_results))]

    # STEP 2: Split into two groups, preserving index
    business_items = [(i, s) for i, s, l in classified if (l.strip().lower()) == "business"]
    ordinary_items = [(i, s) for i, s, l in classified if (l.strip().lower()) == "ordinary"]

    end = datetime.now()
    print(f"Total time consumed for classification: {end-start}")
    print(
        f"[CLASSIFY] Business: {len(business_items)}, Ordinary: {len(ordinary_items)}")

    # Split into translation batches
    business_batches = [business_items[i:i+TRANSLATION_BATCH_SIZE]
                        for i in range(0, len(business_items), TRANSLATION_BATCH_SIZE)]
    ordinary_batches = [ordinary_items[i:i+TRANSLATION_BATCH_SIZE]
                        for i in range(0, len(ordinary_items), TRANSLATION_BATCH_SIZE)]
    
    print(f"\nTotal {len(business_batches)} Business tasks are created, and {len(ordinary_batches)} Ordinary\n")

    business_serialized = serialize_batches(business_batches, "business")
    ordinary_serialized = serialize_batches(ordinary_batches, "ordinary")

    batches_data = {
        "business_batches": business_serialized,
        "ordinary_batches": ordinary_serialized
    }

    batches_file = os.path.join(LOG_DIR, f"batches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(batches_file, "w", encoding="utf-8") as f:
        json.dump(batches_data, f, ensure_ascii=False, indent=2)

    print(f"Saved batches to {batches_file}")

    translation_progress = {"valid": 0, "partial": 0, "total": len(
        business_batches) + len(ordinary_batches)}

    # Run translations in parallel
    start = datetime.now()
    translation_tasks = []
    logs = {}

    for idx, batch in enumerate(business_batches):
        translation_tasks.append(_translate_batch(batch, target_lang, brand_tone, idx+1,
                     len(business_batches), type="business", translation_progress=translation_progress, logs=logs))

    for idx, batch in enumerate(ordinary_batches):
        translation_tasks.append(_translate_batch(batch, target_lang, brand_tone, idx+1,
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
    final_results = [t for _, t in sorted(final_translation_pairs, key=lambda x: x[0])]
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
