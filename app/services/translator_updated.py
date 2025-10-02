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
from typing import Optional


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
openai_model_classify = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY_1,
    timeout=150.0
)

openai_model_translate = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY_2,
    timeout=150.0
)

genai.configure(api_key=settings.GEMINI_API_KEY_1)  # type: ignore
gemini_model_1 = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore

genai.configure(api_key=settings.GEMINI_API_KEY_2)  # type: ignore
gemini_model_2 = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore

# ==== CONFIG ====
BATCH_SIZE = 50
MAX_CONCURRENCY = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

model_cycle = ["openai1", "openai2", "gemini1"]
model_index = 0

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
        model="gpt-4.1-mini",
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
    return await _translate_openai(strings, target_lang, brand_tone, openai_model_translate)


async def translate_openai_2(strings, target_lang, brand_tone):
    return await _translate_openai(strings, target_lang, brand_tone, openai_model_classify)


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


async def translate_gemini_2(strings, target_lang, brand_tone):
    return await _translate_gemini(strings, target_lang, brand_tone, gemini_model_2)


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

    resp = await openai_model_translate.chat.completions.create(
        model="gpt-4.1-mini",
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
    print(
        f"[DEBUG] Classify batch {batch_num}/{total_batches} → {len(strings_batch)} strings, chars={total_chars}")
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
        # type: ignore
        return [line.strip().lower() for line in labels_text.split("\n") if line.strip()]

# ===================== BATCH HANDLER =====================


async def _translate_batch(indexed_strings, target_lang, brand_tone, batch_num, total_batches, type, progress=None):
    strings = [s for _, s in indexed_strings]
    best_provider = None
    best_translation = None
    best_score = float("inf")

    # Define provider order
    providers = ["openai1", "openai2", "gemini1"] if type == "business" else [
        "gemini1", "openai1", "openai2", "gemini2"]

    async with semaphore:
        print(
            f"\n[DEBUG] {type.upper()} Batch {batch_num}/{total_batches} → {len(strings)} strings ")

        for provider in providers:
            try:
                start = time.time()

                if provider == "openai1":
                    translations = await with_retry(translate_openai_1, strings, target_lang, brand_tone, provider=provider)
                elif provider == "openai2":
                    translations = await with_retry(translate_openai_2, strings, target_lang, brand_tone, provider=provider)
                elif provider == "gemini1":
                    translations = await with_retry(translate_gemini_1, strings, target_lang, brand_tone, provider=provider)
                else:
                    translations = await with_retry(translate_gemini_2, strings, target_lang, brand_tone, provider=provider)

                elapsed = time.time() - start
                # rough estimate if API doesn’t return usage
                tokens_used = len(" ".join(strings)) // 4
                TRANSLATION_STATS["tokens"][provider].append(
                    {"tokens": tokens_used, "time": elapsed})

                if translations:
                    expected = len(strings)
                    got = len(translations)

                    # --- FIX: force align translations ---
                    if got < expected:
                        # Pad missing with originals
                        translations.extend(strings[got:])
                    elif got > expected:
                        # Trim extras
                        translations = translations[:expected]

                    # Record mismatch stats
                    if expected != got:
                        TRANSLATION_STATS["mismatches"]["batches"].append({
                            "batch": batch_num, "provider": provider,
                            "expected": expected, "got": got, "adjusted_to": len(translations),
                            "mismatched": abs(expected - got)
                        })
                        TRANSLATION_STATS["mismatches"]["total_mismatched"] += abs(
                            expected - got)

                    score = abs(got - expected)
                    if score < best_score or (score == best_score and got > len(best_translation or [])):
                        best_score = score
                        best_provider = provider
                        best_translation = translations

                # Now translations is always aligned to `strings`
                if translations and len(translations) == len(strings):
                    if progress is not None:
                        progress["done"] += 1
                        print(
                            f"[PROGRESS: valid] {progress['done']}/{progress['total']} (batch {batch_num} via {provider})")
                    return [(i, t) for (i, _), t in zip(indexed_strings, translations)]

                else:
                    print(
                        f"⚠ {provider} returned {len(translations) if translations else 'None'} items, expected {len(strings)}")
                    continue

            except Exception as e:
                print(f"⚠ {provider} failed for batch {batch_num}: {e}")
                TRANSLATION_STATS["fallbacks"][provider].append(
                    {"batch": batch_num, "reason": str(e)})
                continue

        #  All failed, fallback
        if best_translation:
            if len(best_translation) < len(strings):
                best_translation.extend(strings[len(best_translation):])
                print(
                    f"⚠ Fallback used for less {len(strings) - len(best_translation)} strings")
            else:
                best_translation = best_translation[:len(strings)]
                print(
                    f"⚠ Fallback used for extra {len(best_translation) - len(strings)} strings")
            if progress is not None:
                progress["done"] += 1
                print(
                    f"[PROGRESS: partial] batch {batch_num} using fallback {best_provider}")
            return [(i, t if t else s) for (i, s), t in zip(indexed_strings, best_translation)]
        else:
            if progress is not None:
                progress["done"] += 1
                print(
                    f"[PROGRESS: failed] batch {batch_num}, returning originals")
            return [(i, s) for (i, s) in indexed_strings]

# ===================== SAVE REPORT =====================


def save_report():
    # --- Compute Totals ---
    totals = {
        "fallbacks": {k: len(v) for k, v in TRANSLATION_STATS["fallbacks"].items()},
        "rate_limits": {k: len(v) for k, v in TRANSLATION_STATS["rate_limits"].items()},
        "tokens": {k: sum(x["tokens"] for x in v) for k, v in TRANSLATION_STATS["tokens"].items()},
        "mismatches": {
            "batches": len(TRANSLATION_STATS["mismatches"]["batches"]),
            "strings": TRANSLATION_STATS["mismatches"]["total_mismatched"]
        }
    }

    # add grand totals
    totals["fallbacks"]["total"] = sum(totals["fallbacks"].values())
    totals["rate_limits"]["total"] = sum(
        v for k, v in totals["rate_limits"].items() if k != "total"
    )
    totals["tokens"]["total"] = sum(totals["tokens"].values())

    # --- Print to logs ---
    print("\n========== TRANSLATION TOTALS ==========")
    print("Fallbacks:", totals["fallbacks"])
    print("Rate limits:", totals["rate_limits"])
    print("Tokens:", totals["tokens"])
    print("Mismatches:", totals["mismatches"])
    print("========================================\n")

    # --- Attach to report JSON ---
    TRANSLATION_STATS["totals"] = totals

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

    classified = [(i, s, l)
                  for i, (s, l) in enumerate(zip(strings_to_translate, labels))]

    # STEP 2: Split into two groups, preserving index
    business_items = [(i, s) for i, s, l in classified if l == "business"]
    ordinary_items = [(i, s) for i, s, l in classified if l == "ordinary"]

    end = datetime.now()
    print(f"Total time consumed for classification: {end-start}")
    print(
        f"[CLASSIFY] Business: {len(business_items)}, Ordinary: {len(ordinary_items)}")

    # Split into translation batches
    business_batches = [business_items[i:i+BATCH_SIZE]
                        for i in range(0, len(business_items), BATCH_SIZE)]
    ordinary_batches = [ordinary_items[i:i+BATCH_SIZE]
                        for i in range(0, len(ordinary_items), BATCH_SIZE)]

    progress = {"done": 0, "total": len(
        business_batches) + len(ordinary_batches)}

    # Run translations in parallel
    tasks = []

    for idx, batch in enumerate(business_batches):
        tasks.append(_translate_batch(batch, target_lang, brand_tone, idx+1,
                     len(business_batches), type="business", progress=progress))

    for idx, batch in enumerate(ordinary_batches):
        tasks.append(_translate_batch(batch, target_lang, brand_tone, idx+1,
                     len(ordinary_batches), type="ordinary", progress=progress))

    random.shuffle(tasks)

    all_results = await asyncio.gather(*tasks)

    # Flatten already-indexed results
    final_pairs = [pair
                   for batch in all_results if batch is not None
                   for pair in batch]

    # ---------- RECOMBINE ----------
    final_results = [t for _, t in sorted(final_pairs, key=lambda x: x[0])]

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
