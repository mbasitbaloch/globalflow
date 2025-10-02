import json
import random
import re
import asyncio
import os
import sys
from datetime import datetime
from openai import AsyncOpenAI
import google.generativeai as genai  # Gemini SDK
from ..config import settings
from typing import Optional

# ==== CLIENTS ====
openai_model_classify = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY_1,
    timeout=150.0
)

openai_model_translate = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY_2,
    timeout=150.0
)

genai.configure(api_key=settings.GEMINI_API_KEY_1) # type: ignore
gemini_model_1 = genai.GenerativeModel("gemini-1.5-flash") # type: ignore

# genai.configure(api_key=settings.GEMINI_API_KEY_2) # type: ignore
# gemini_model_2 = genai.GenerativeModel("gemini-1.5-flash") # type: ignore

# ==== CONFIG ====
BATCH_SIZE = 50
MAX_CONCURRENCY = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

model_cycle = ["openai", "gemini1"]
model_index = 0

sys.setrecursionlimit(3000)

# ==== LOG DIR ====
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ===================== HELPERS =====================
def is_translateable(text: str) -> bool:
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
    if text.startswith(("shopify.", "customer.", "customer_", "templates.", "section.", "sections.")):
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
            if "Rate limit" in str(e) or "quota" in str(e).lower():
                wait = (2 ** i) + random.random()
                print(f"⚠ Rate limit, retrying in {wait:.2f}s...")
                await asyncio.sleep(wait)
            else:
                print(f"⚠ Error: {e}, retrying...")
                await asyncio.sleep(2)
    raise Exception("Max retries reached")


# ===================== TRANSLATION FUNCTIONS =====================
async def translate_openai(strings, target_lang, brand_tone):
#     prompt = {
#         "role": "user",
#         "content": f"""
#         Translate the following {len(strings)} strings into {target_lang}.
#         Maintain the brand tone as '{brand_tone}'.
#         ⚠ If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags as-is, only translate inner text.
#         Return ONLY a valid JSON array of translated strings, in the same order, nothing else.
#         Strings:
#         {json.dumps(strings, ensure_ascii=False)}
# """
#     }

    prompt = f"""
    You are a professional translator.

    Task:
    Translate the following {len(strings)} strings into {target_lang}.
    - Maintain the brand tone as '{brand_tone}'.
    - If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags unchanged, only translate the inner text.
    - Preserve placeholders (e.g., {{name}}, %s, {{0}}) exactly as they are.
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

    resp = await openai_model_translate.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user", "content":prompt}], # type: ignore
        temperature=0.0,
        response_format = {
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
        } # type: ignore
    )

    content = resp.choices[0].message.content
    try:
        data = json.loads(content) # type: ignore
        return [clean_line(x) for x in data.get("translations", [])]
    except Exception:
        return [clean_line(line) for line in content.split("\n") if line.strip()] # type: ignore


async def _translate_with_gemini(strings, target_lang, brand_tone, model):
#     prompt = f"""
#     Translate the following {len(strings)} strings into {target_lang}.
#     Maintain the brand tone as '{brand_tone}'.
#     ⚠ If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags as-is, only translate inner text.
#     Return ONLY a valid JSON array of translated strings, same order, nothing else.
#     Strings:
#     {json.dumps(strings, ensure_ascii=False)}
# """

    prompt = f"""
    You are a professional translator.

    Task:
    Translate the following {len(strings)} strings into {target_lang}.
    - Maintain the brand tone as '{brand_tone}'.
    - If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags unchanged, only translate the inner text.
    - Preserve placeholders (e.g., {{name}}, %s, {{0}}) exactly as they are.
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
    resp = await asyncio.to_thread(
        model.generate_content,
        prompt,
        generation_config={"response_mime_type": "application/json"}
        )
    text = resp.text or "[]"
    try:
        return [clean_line(x) for x in json.loads(text)]
    except Exception:
        return [clean_line(line) for line in text.split("\n") if line.strip()]


async def translate_gemini_1(strings, target_lang, brand_tone):
    return await _translate_with_gemini(strings, target_lang, brand_tone, gemini_model_1)


# async def translate_gemini_2(strings, target_lang, brand_tone):
#     return await _translate_with_gemini(strings, target_lang, brand_tone, gemini_model_2)


# ===================== CLASSIFICATION =====================
async def classify_strings(strings_batch, batch_num, total_batches):
    """
    Classify strings into 'business' or 'ordinary'.
    """

    MAX_CHARS_PER_CLASSIFY = 50_000  # conservative
    total_chars = sum(len(s) for s in strings_batch)
    if total_chars > MAX_CHARS_PER_CLASSIFY and len(strings_batch) > 1:
        # split into two equal parts
        mid = len(strings_batch) // 2
        left = await classify_strings(strings_batch[:mid], batch_num, total_batches)
        right = await classify_strings(strings_batch[mid:], batch_num, total_batches)
        return left + right

#     prompt = f"""
# You are a text classifier. 
# Classify each string into exactly one of these categories:
# - "business" (for business, legal, official, invoice, policy, formal communication)
# - "ordinary" (casual, product descriptions, marketing, blogs, general content)

# Return ONLY a valid JSON array of labels in the same order.

# Examples:
# Input: ["Invoice #4533", "Big summer sale!", "Refunds will be processed within 7 days", "Sign In"]
# Output: ["business", "ordinary", "business", "ordinary"]

# Input: ["Terms and Conditions apply", "Export License Required", "Check out our new arrivals", "Best quality leather shoes"]
# Output: ["business", "business", "ordinary", "ordinary"]

# Now classify:
# {json.dumps(strings_batch, ensure_ascii=False)}
# """

    prompt =  f"""
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


    resp = await openai_model_translate.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user", "content":prompt}], # type: ignore
        temperature=0.0,
        response_format = {
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
        } # type: ignore
    )

    labels_text: str = resp.choices[0].message.content # type: ignore
    print(f"[DEBUG] Classify batch {batch_num}/{total_batches} → {len(strings_batch)} strings, chars={total_chars}")
    try:
        data = json.loads(labels_text)
        labels = data.get("classified_labels", [])
        # Ensure only valid labels
        VALID_LABELS = {"ordinary", "business"}
        labels = [l if l in VALID_LABELS else "ordinary" for l in labels]
        return labels
        # return json.loads(labels_text) # type: ignore
    except Exception:
        # fallback: line by line
        return [line.strip().lower() for line in labels_text.split("\n") if line.strip()] # type: ignore

# ===================== BATCH HANDLER =====================
# async def _translate_batch(indexed_strings, target_lang, brand_tone, batch_num, total_batches, type, progress=None, max_retries=3):
#     global model_index

#     for attempt in range(1, max_retries + 1):
#         async with semaphore:
#             if type == "business":
#                 model_index = 0
#             else:
#                 model_index = 1

#             print(f"\n[DEBUG] {type.upper()} Batch {batch_num}/{total_batches} → {len(indexed_strings)} strings "
#                   f"(attempt {attempt}/{max_retries})")

#             current_model = model_cycle[model_index % len(model_cycle)]
#             model_index += 1

#             strings = [s for _, s in indexed_strings]

#             try:
#                 if current_model == "openai":
#                     translations = await with_retry(translate_openai, strings, target_lang, brand_tone)
#                 elif current_model == "gemini1":
#                     translations = await with_retry(translate_gemini_1, strings, target_lang, brand_tone)
#                 else:
#                     translations = await with_retry(translate_gemini_2, strings, target_lang, brand_tone)
#             except Exception as e:
#                 print(f"⚠ {current_model} failed, falling back: {e}")
#                 translations = None
#                 for alt in model_cycle:
#                     if alt == current_model:
#                         continue
#                     try:
#                         if alt == "openai":
#                             translations = await translate_openai(strings, target_lang, brand_tone)
#                         elif alt == "gemini1":
#                             translations = await translate_gemini_1(strings, target_lang, brand_tone)
#                         else:
#                             translations = await translate_gemini_2(strings, target_lang, brand_tone)
#                         break
#                     except Exception as e2:
#                         print(f"⚠ Fallback {alt} also failed: {e2}")

#             # ✅ Validation: correct length
#             if translations and len(translations) == len(strings):
#                 if progress is not None:
#                     progress["done"] += 1
#                     print(f"[PROGRESS] {progress['done']}/{progress['total']} batches done "
#                         f"(latest: {type} batch {batch_num} via {current_model})")
#                 return [(i, t) for (i, _), t in zip(indexed_strings, translations)]
#             else:
#                 print(f"⚠ Validation failed for {type} batch {batch_num} on attempt {attempt}/{max_retries}. "
#                       f"Expected {len(strings)}, got {len(translations) if translations else 'None'}.")

#     # 🚨 After all retries fail
#     raise Exception(f"All retries failed for {type} batch {batch_num}")


async def _translate_batch(indexed_strings, target_lang, brand_tone, batch_num, total_batches, type, progress=None):
    global model_index

    async with semaphore:
        if type == "business":
            model_index = 0
        else:
            model_index = 1
        strings = [s for _, s in indexed_strings]
        print(
            f"\n[DEBUG] {type.upper()} Batch {batch_num}/{total_batches} → {len(strings)} strings ")
        current_model = model_cycle[model_index % len(model_cycle)]
        model_index += 1

        try:
            if current_model == "openai":
                translations = await with_retry(translate_openai, strings, target_lang, brand_tone)
            else:
                translations = await with_retry(translate_gemini_1, strings, target_lang, brand_tone)

        except Exception as e:
            print(f"⚠ {current_model} failed, falling back: {e}")
            for alt in model_cycle:
                if alt == current_model:
                    continue
                try:
                    if alt == "openai":
                        translations = await translate_openai(strings, target_lang, brand_tone)
                    else:
                        translations = await translate_gemini_1(strings, target_lang, brand_tone)
                    break
                except Exception as e2:
                    print(f"⚠ Fallback {alt} also failed: {e2}")
            else:
                raise Exception("All providers failed!")

        if translations and len(translations) == len(strings):
            if progress is not None:
                progress["done"] += 1
                print(
                    f"[PROGRESS: valid] {progress["done"]+progress["invalid"]}/{progress['total']} batches done "
                    f"(latest: {type} batch {batch_num} via {current_model})"
                )
        else:
            if progress is not None:
                progress["invalid"] += 1
                print(
                    f"[PROGRESS: invalid] {progress["done"]+progress["invalid"]}/{progress['total']} batches done "
                    f"(latest: {type} batch {batch_num} via {current_model})"
                )

        print(f"{progress["done"]} are successful and {progress["invalid"]} are unsuccessful!") #type: ignore
        return [(i, t) for (i, _), t in zip(indexed_strings, translations)]




# ===================== MAIN TRANSLATOR =====================
async def fast_translate_json(target_data, target_lang, brand_tone):
    positions = []  # (path, string, path_str)

    # if "fullData" in data and "storeData" in data["fullData"]:
    #     target_data = data["fullData"]["storeData"]
    # else:
    #     print("⚠ No fullData.storeData found")
    #     return data
    # target_data = data

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
                # onlineStoreThemes, menus, links, shopPolicies
                # elif k == "translatableContent" and isinstance(v, list):
                #     for i, item in enumerate(v):
                #         if isinstance(item, dict) and "value" in item:
                #             val = item["value"]
                #             if isinstance(val, str) and is_translateable(val):
                #                 positions.append(
                #                     (path + [k, i, "value"], val, ".".join(map(str, path + [k, i, "value"]))))
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


    # VALID_LABELS = {"business", "ordinary"}

    def normalize_labels(strings: list[str], labels: list[str], default="ordinary") -> list[str]:
        """
        Ensures labels list has exactly the same length as strings.
        If labels are missing, pad with default.
        If too many labels, truncate.
        """
        if not labels:
            # totally broken response → assume all default
            return [default] * len(strings)

        if len(labels) < len(strings):
            # pad missing labels
            labels = labels + [default] * (len(strings) - len(labels))
        elif len(labels) > len(strings):
            # truncate extras
            labels = labels[:len(strings)]

        return labels

    collect_strings(target_data)

    strings_to_translate = [s for _, s, _ in positions]
    print(f"Total strings: {len(strings_to_translate)}")
    # indexed_strings = list(enumerate(strings_to_translate))

    # ---- SAVE EXTRACTED ----
    extracted_log = [{"path": p, "string": s} for _, s, p in positions]
    extracted_file = os.path.join(
        LOG_DIR, f"extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(extracted_file, "w", encoding="utf-8") as f:
        json.dump(extracted_log, f, ensure_ascii=False, indent=2)
    print(f"Saved extracted strings to {extracted_file}")

    # ---- TRANSLATE ----
    batches = [strings_to_translate[i:i+BATCH_SIZE]
               for i in range(0, len(strings_to_translate), BATCH_SIZE)]
    total_batches = len(batches)
    start = datetime.now()
    
    labels = []
    batch_num = 0
    for cb in batches:
        batch_num += 1
        lbls = await classify_strings(cb, batch_num, total_batches)
        lbls = normalize_labels(cb, lbls)
        labels.extend(lbls)

    classified = [(i, s, l) for i, (s, l) in enumerate(zip(strings_to_translate, labels))]
    # classified = [(i, s, l) for (i, s), l in zip(indexed_strings, labels)]

    # STEP 2: Split into two groups, preserving index
    business_items = [(i, s) for i, s, l in classified if l == "business"]
    ordinary_items = [(i, s) for i, s, l in classified if l == "ordinary"]
    
    end = datetime.now()
    print(f"Total time consumed for classification: {end-start}")
    print(f"[CLASSIFY] Business: {len(business_items)}, Ordinary: {len(ordinary_items)}")

    # Split into translation batches
    business_batches = [business_items[i:i+BATCH_SIZE] for i in range(0, len(business_items), BATCH_SIZE)]
    ordinary_batches = [ordinary_items[i:i+BATCH_SIZE] for i in range(0, len(ordinary_items), BATCH_SIZE)]

    progress = {"done": 0, "invalid": 0, "total": len(business_batches) + len(ordinary_batches)}

    # Run translations in parallel
    tasks = []

    for idx, batch in enumerate(business_batches):
        tasks.append(_translate_batch(batch, target_lang, brand_tone, idx+1, len(business_batches), type="business", progress=progress))

    for idx, batch in enumerate(ordinary_batches):
        tasks.append(_translate_batch(batch, target_lang, brand_tone, idx+1, len(ordinary_batches), type="ordinary", progress=progress))
    
    random.shuffle(tasks)

    all_results = await asyncio.gather(*tasks)

    # Flatten already-indexed results
    final_pairs = [pair
                   for batch in all_results if batch is not None
                   for pair in batch]

    # ---------- RECOMBINE ----------
    # final_results = {i: t for i, t in (business_results + ordinary_results)}
    final_results = [t for _, t in sorted(final_pairs, key=lambda x: x[0])]

    comparative_file = os.path.join(LOG_DIR, "comparative.json")
    # print(f"Translation results:\n{final_results}")
    with open(comparative_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"path":path_str,"orig": orig, "trans": trans} for (path, orig_val, path_str), orig, trans in zip(positions, strings_to_translate, final_results)],
            f, ensure_ascii=False, indent=2
        )
    print(f"Saved comparison strings to {comparative_file}")

    # ---------- INJECTION ----------
    def set_value(d, path, value):
        ref = d
        for p in path[:-1]:
            ref = ref[p]
        ref[path[-1]] = value

    injected_log = []
    # for idx, (path, orig_val, path_str) in enumerate(positions):
    #     translated = final_results.get(idx, orig_val)  # fallback: keep original if missing
    #     set_value(target_data, path, translated)
    #     injected_log.append({"path": path_str, "translated": translated})
    #     print(f"[INJECT] {path_str} -> {translated[:60]}") # type: ignore
    for i, translated in enumerate(final_results):
        path, orig_val, path_str = positions[i]
        set_value(target_data, path, translated)
        injected_log.append({"path": path_str, "translated": translated})
        # print(f"[INJECT] {path_str} -> {translated[:60]}")

    # ---- SAVE INJECTED ----
    injected_file = os.path.join(
        LOG_DIR, f"injected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(injected_file, "w", encoding="utf-8") as f:
        json.dump(injected_log, f, ensure_ascii=False, indent=2)
    print(f"Saved injected strings to {injected_file}")

    print(f" Injected {len(final_results)}/{len(strings_to_translate)} strings")
    return target_data
