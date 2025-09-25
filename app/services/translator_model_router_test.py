import json
import random
import re
import asyncio
import os
import sys
from datetime import datetime
from openai import AsyncOpenAI
import google.generativeai as genai  # Gemini SDK

# ==== CLIENTS ====
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=120.0
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))
gemini_model_1 = genai.GenerativeModel("gemini-1.5-flash")

genai.configure(api_key=os.getenv("GEMINI_API_KEY_2"))
gemini_model_2 = genai.GenerativeModel("gemini-1.5-flash")

# ==== CONFIG ====
BATCH_SIZE = 50
MAX_CONCURRENCY = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

model_cycle = ["gemini1", "gemini2", "openai"]
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
    prompt = {
        "role": "user",
        "content": f"""
Translate the following {len(strings)} strings into {target_lang}.
Maintain the brand tone as '{brand_tone}'.
⚠ If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags as-is, only translate inner text.
Return ONLY a valid JSON array of translated strings, in the same order, nothing else.
Strings:
{json.dumps(strings, ensure_ascii=False)}
"""
    }

    resp = await openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[prompt],
        temperature=0.7,
    )

    content = resp.choices[0].message.content
    try:
        return [clean_line(x) for x in json.loads(content)]
    except Exception:
        return [clean_line(line) for line in content.split("\n") if line.strip()]


async def _translate_with_gemini(strings, target_lang, brand_tone, model):
    prompt = f"""
Translate the following {len(strings)} strings into {target_lang}.
Maintain the brand tone as '{brand_tone}'.
⚠ If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags as-is, only translate inner text.
Return ONLY a valid JSON array of translated strings, same order, nothing else.
Strings:
{json.dumps(strings, ensure_ascii=False)}
"""
    resp = model.generate_content(prompt)
    text = resp.text or "[]"
    try:
        return [clean_line(x) for x in json.loads(text)]
    except Exception:
        return [clean_line(line) for line in text.split("\n") if line.strip()]


async def translate_gemini_1(strings, target_lang, brand_tone):
    return await _translate_with_gemini(strings, target_lang, brand_tone, gemini_model_1)


async def translate_gemini_2(strings, target_lang, brand_tone):
    return await _translate_with_gemini(strings, target_lang, brand_tone, gemini_model_2)


# ===================== BATCH HANDLER =====================
async def _translate_batch(strings, target_lang, brand_tone, batch_num, total_batches):
    global model_index
    async with semaphore:
        print(
            f"\n[DEBUG] Batch {batch_num}/{total_batches} → {len(strings)} strings")
        current_model = model_cycle[model_index % len(model_cycle)]
        model_index += 1

        try:
            if current_model == "openai":
                result = await with_retry(translate_openai, strings, target_lang, brand_tone)
            elif current_model == "gemini1":
                result = await with_retry(translate_gemini_1, strings, target_lang, brand_tone)
            else:
                result = await with_retry(translate_gemini_2, strings, target_lang, brand_tone)
        except Exception as e:
            print(f"⚠ {current_model} failed, falling back: {e}")
            for alt in model_cycle:
                if alt == current_model:
                    continue
                try:
                    if alt == "openai":
                        result = await translate_openai(strings, target_lang, brand_tone)
                    elif alt == "gemini1":
                        result = await translate_gemini_1(strings, target_lang, brand_tone)
                    else:
                        result = await translate_gemini_2(strings, target_lang, brand_tone)
                    break
                except Exception as e2:
                    print(f"⚠ Fallback {alt} also failed: {e2}")
            else:
                raise Exception("All providers failed!")

        print(
            f"[✔] Completed batch {batch_num}/{total_batches} via {current_model}")
        return result

# ===================== MAIN TRANSLATOR =====================


# async def fast_translate_json(data, target_lang, brand_tone):
#     positions = []  # (index, path, string, path_str)

#     target_data = data

#     # ---------- COLLECT STRINGS ----------
#     def collect_strings(d, path=None, parent_key=None):
#         if path is None:
#             path = []

#         if isinstance(d, dict):
#             for k, v in d.items():
#                 if parent_key == "products" and k in ["title", "descriptionHtml", "productType", "vendor", "status"]:
#                     if isinstance(v, str) and is_translateable(v):
#                         positions.append(
#                             (len(positions), path + [k], v, ".".join(map(str, path + [k]))))
#                 elif parent_key == "images" and k == "altText":
#                     if isinstance(v, str) and is_translateable(v):
#                         positions.append(
#                             (len(positions), path + [k], v, ".".join(map(str, path + [k]))))
#                 elif parent_key == "variants" and k == "title":
#                     if isinstance(v, str) and is_translateable(v):
#                         positions.append(
#                             (len(positions), path + [k], v, ".".join(map(str, path + [k]))))
#                 elif parent_key == "collections" and k in ["title", "descriptionHtml", "handle"]:
#                     if isinstance(v, str) and is_translateable(v):
#                         positions.append(
#                             (len(positions), path + [k], v, ".".join(map(str, path + [k]))))
#                 elif parent_key == "blogs" and k in ["title", "handle"]:
#                     if isinstance(v, str) and is_translateable(v):
#                         positions.append(
#                             (len(positions), path + [k], v, ".".join(map(str, path + [k]))))
#                 elif parent_key == "shopPolicies" and k in ["value", "locale"]:
#                     if isinstance(v, str) and is_translateable(v):
#                         positions.append(
#                             (len(positions), path + [k], v, ".".join(map(str, path + [k]))))
#                 elif k == "translatableContent" and isinstance(v, list):
#                     for i, item in enumerate(v):
#                         if isinstance(item, dict):
#                             for field in ["value", "locale"]:
#                                 if field in item and isinstance(item[field], str) and is_translateable(item[field]):
#                                     positions.append(
#                                         (len(positions), path + [k, i, field], item[field],
#                                          ".".join(map(str, path + [k, i, field]))))
#                             collect_strings(item, path + [k, i], k)
#                 elif k in ["title", "body", "value", "altText", "description", "name"]:
#                     if isinstance(v, str) and is_translateable(v):
#                         positions.append(
#                             (len(positions), path + [k], v, ".".join(map(str, path + [k]))))
#                 else:
#                     collect_strings(v, path + [k], k)

#         elif isinstance(d, list):
#             for i, item in enumerate(d):
#                 collect_strings(item, path + [i], parent_key)

#     collect_strings(target_data)

#     # Save extracted
#     extracted_log = [{"index": i, "path": p, "string": s}
#                      for i, p, s, _ in positions]
#     with open(os.path.join(LOG_DIR, f"extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
#         json.dump(extracted_log, f, ensure_ascii=False, indent=2)

#     # ---------- CLASSIFICATION ----------
#     def classify_string(s: str) -> str:
#         print(f"[?] Classifying: {s[:30]}...")
#         keywords = ["policy", "terms", "conditions",
#                     "refund", "privacy", "legal"]
#         if any(kw.lower() in s.lower() for kw in keywords):
#             return "business"
#         return "ordinary"

#     business_jobs = [(i, s) for i, _, s,
#                      _ in positions if classify_string(s) == "business"]
#     ordinary_jobs = [(i, s) for i, _, s,
#                      _ in positions if classify_string(s) == "ordinary"]

#     translations = {}

#     # ---------- TRANSLATE BUSINESS STRINGS (OpenAI) ----------
#     print("translating business string")
#     if business_jobs:
#         biz_strings = [s for _, s in business_jobs]
#         biz_batches = [biz_strings[i:i+BATCH_SIZE]
#                        for i in range(0, len(biz_strings), BATCH_SIZE)]
#         results = []
#         for idx, batch in enumerate(biz_batches):
#             out = await _translate_batch(batch, target_lang, brand_tone, idx+1, len(biz_batches))
#             results.extend(out)
#         for (i, _), t in zip(business_jobs, results):
#             translations[i] = t

#     # ---------- TRANSLATE ORDINARY STRINGS (Gemini) ----------
#     print("translating ordinary string")
#     if ordinary_jobs:
#         ord_strings = [s for _, s in ordinary_jobs]
#         ord_batches = [ord_strings[i:i+BATCH_SIZE]
#                        for i in range(0, len(ord_strings), BATCH_SIZE)]
#         results = []
#         for idx, batch in enumerate(ord_batches):
#             out = await with_retry(translate_gemini_1, batch, target_lang, brand_tone)
#             results.extend(out)
#         for (i, _), t in zip(ordinary_jobs, results):
#             translations[i] = t

#     # ---------- INJECTION ----------
#     def set_value_with_original(d, path, translated):
#         print("injecting strings")
#         ref = d
#         for p in path[:-1]:
#             ref = ref[p]

#         last_key = path[-1]
#         original_value = ref[last_key]

#         prefixed_key = f"original{last_key[0].upper()}{last_key[1:]}"
#         if prefixed_key not in ref:
#             ref[prefixed_key] = original_value

#         ref[last_key] = translated

#     injected_log = []
#     for i, path, orig_val, path_str in positions:
#         # fallback: keep original if missing
#         translated = translations.get(i, orig_val)
#         set_value_with_original(target_data, path, translated)
#         injected_log.append(
#             {"path": path_str, "original": orig_val, "translated": translated})

#     injected_file = os.path.join(
#         LOG_DIR, f"injected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
#     with open(injected_file, "w", encoding="utf-8") as f:
#         json.dump(injected_log, f, ensure_ascii=False, indent=2)

#     return target_data


async def fast_translate_json(data, target_lang, brand_tone):
    """
    Modernized, safe, order-preserving translation pipeline:
    - Collect all translatable positions (same as before)
    - Classify strings into 'business' vs 'ordinary' (cheap local heuristic)
    - Create batches preserving original indices
    - Translate business => OpenAI only
      Translate ordinary => Gemini then fallback to OpenAI
    - On any provider failure for a batch, inject originals (no crash)
    - Inject translations back into the same JSON structure, adding originalXxx fields
    - Save extracted/injected logs like before
    """
    positions = []  # (path, string, path_str)

    # keep entire structure (user + summary + fullData etc) — per your request
    target_data = data

    # ---------- COLLECTION RULES (keep yours) ----------
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
                            (path + [k], v, ".".join(map(str, path + [k]))))

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

                # SHOP POLICIES and translatableContent
                elif parent_key == "shopPolicies" and k in ["value", "locale"]:
                    if isinstance(v, str) and is_translateable(v):
                        positions.append(
                            (path + [k], v, ".".join(map(str, path + [k]))))

                elif k == "translatableContent" and isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            # check common fields inside translatableContent items
                            for field in item:
                                if field in ["value", "locale", "key"] and isinstance(item[field], str) and is_translateable(item[field]):
                                    positions.append(
                                        (path + [k, i, field], item[field], ".".join(map(str, path + [k, i, field]))))
                            # also recursively check deeper if present
                            collect_strings(item, path + [k, i], k)

                # other common translatable names
                elif k in ["title", "body", "value", "altText", "description", "name"]:
                    if isinstance(v, str) and is_translateable(v):
                        positions.append(
                            (path + [k], v, ".".join(map(str, path + [k]))))
                else:
                    collect_strings(v, path + [k], k)

        elif isinstance(d, list):
            for i, item in enumerate(d):
                collect_strings(item, path + [i], parent_key)

    collect_strings(target_data)

    strings = [s for _, s, _ in positions]
    print(f"Total strings collected: {len(strings)}")

    # Save extracted (same as before)
    extracted_log = [{"path": p, "string": s} for _, s, p in positions]
    extracted_file = os.path.join(
        LOG_DIR, f"extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(extracted_file, "w", encoding="utf-8") as f:
        json.dump(extracted_log, f, ensure_ascii=False, indent=2)
    print(f"Saved extracted strings to {extracted_file}")

    # ----------------- Classify strings (cheap local heuristic) -----------------
    # You can replace this with a model call if you prefer (but that costs calls).
    biz_keywords = {
        "privacy", "terms", "refund", "shipping", "refund policy", "terms of service",
        "cookie", "disclaimer", "warranty", "return", "legal", "privacy policy", "shipping policy"
    }

    def is_business_string(s: str) -> bool:
        low = s.lower()
        # keyword match OR likely long/structured legal-sounding content
        if any(k in low for k in biz_keywords):
            return True
        if len(low) > 800:  # very long => likely policy / legal / important
            return True
        # simple heuristic for dates/last-updated blocks
        if "last updated" in low or "effective date" in low:
            return True
        return False

    labels = [is_business_string(s) for s in strings]  # True => business

    # ----------------- Build index-preserving batches -----------------
    def make_indexed_items(idx_list):
        return [(i, strings[i]) for i in idx_list]

    biz_indices = [i for i, lab in enumerate(labels) if lab]
    ord_indices = [i for i, lab in enumerate(labels) if not lab]

    biz_items = make_indexed_items(biz_indices)
    ord_items = make_indexed_items(ord_indices)

    def chunk_items(items, size):
        return [items[i:i+size] for i in range(0, len(items), size)]

    biz_batches = chunk_items(biz_items, BATCH_SIZE)
    ord_batches = chunk_items(ord_items, BATCH_SIZE)

    print(
        f"Business batches: {len(biz_batches)}  Ordinary batches: {len(ord_batches)}")

    # ----------------- Batch translate helper -----------------
    async def translate_batch_items(items, target_lang, brand_tone, batch_num, total_batches, prefer_openai=False):
        """
        items: list of (orig_index, string)
        returns: list of (orig_index, translated_string) (same order as items)
        prefer_openai: if True -> use OpenAI only (for business)
        """
        async with semaphore:
            strings_to_send = [s for _, s in items]
            if not strings_to_send:
                return []

            # choose flow
            if prefer_openai:
                # OpenAI only flow (for critical/business strings)
                try:
                    translated = await with_retry(translate_openai, strings_to_send, target_lang, brand_tone)
                except Exception as e:
                    print(
                        f"❌ OpenAI failed for business batch {batch_num}/{total_batches}: {e}")
                    translated = strings_to_send  # fallback to originals
            else:
                # ordinary: try Gemini1 -> Gemini2 -> OpenAI
                translated = None
                try:
                    translated = await with_retry(translate_gemini_1, strings_to_send, target_lang, brand_tone)
                except Exception as e1:
                    print(f"⚠ gemini1 failed: {e1}, trying gemini2...")
                    try:
                        translated = await with_retry(translate_gemini_2, strings_to_send, target_lang, brand_tone)
                    except Exception as e2:
                        print(
                            f"⚠ gemini2 failed: {e2}, trying OpenAI fallback...")
                        try:
                            translated = await with_retry(translate_openai, strings_to_send, target_lang, brand_tone)
                        except Exception as e3:
                            print(
                                f"❌ All providers failed for ordinary batch {batch_num}/{total_batches}: {e3}")
                            translated = strings_to_send  # fallback originals

            # enforce same length
            if len(translated) != len(strings_to_send):
                print(
                    f"⚠ Length mismatch in translation result for batch {batch_num}/{total_batches}. Padding/truncating.")
                if len(translated) < len(strings_to_send):
                    translated.extend(strings_to_send[len(translated):])
                else:
                    translated = translated[:len(strings_to_send)]

            # return with original indices
            return [(items[i][0], translated[i]) for i in range(len(items))]

    # ----------------- Fire translation tasks in parallel (but limited by semaphore) ----------
    tasks = []
    # business batches -> prefer_openai True
    for idx, batch in enumerate(biz_batches):
        tasks.append(translate_batch_items(batch, target_lang,
                     brand_tone, idx+1, len(biz_batches), prefer_openai=True))
    # ordinary batches
    for idx, batch in enumerate(ord_batches):
        tasks.append(translate_batch_items(batch, target_lang,
                     brand_tone, idx+1, len(ord_batches), prefer_openai=False))

    # gather results (some batches may take longer; concurrency limited by semaphore)
    all_results = []
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for r in results:
            all_results.extend(r)

    # Build mapping index -> translated string
    translations_map = {idx: val for idx, val in all_results}

    # ---------- INJECTION: set translated strings back and also keep original with prefix ----------
    def set_value_with_original(d, path, translated):
        ref = d
        for p in path[:-1]:
            ref = ref[p]
        last_key = path[-1]
        # original value
        try:
            original_value = ref[last_key]
        except Exception:
            original_value = None

        # if last_key is a string field, add prefixed original
        if isinstance(last_key, str) and original_value is not None:
            prefixed_key = f"original{last_key[0].upper()}{last_key[1:]}" if len(
                last_key) > 0 else f"original_{last_key}"
            # avoid overwriting if already exists
            if prefixed_key not in ref:
                ref[prefixed_key] = original_value

        # set translated
        ref[last_key] = translated

    injected_log = []
    for i, (path, orig_val, path_str) in enumerate(positions):
        new_val = translations_map.get(
            i, orig_val)  # if missing, keep original
        set_value_with_original(target_data, path, new_val)
        injected_log.append(
            {"path": path_str, "original": orig_val, "translated": new_val})
        if i % 200 == 0:
            # occasional progress log
            print(f"[INJECT PROGRESS] {i+1}/{len(positions)}")

    # ---- SAVE INJECTED ----
    injected_file = os.path.join(
        LOG_DIR, f"injected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(injected_file, "w", encoding="utf-8") as f:
        json.dump(injected_log, f, ensure_ascii=False, indent=2)
    print(f"Saved injected strings to {injected_file}")

    print(f"✅ Injected {len(injected_log)}/{len(strings)} strings")
    return target_data
