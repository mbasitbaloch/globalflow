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
    api_key=os.getenv("OPENAI_API_KEY_1"),
    timeout=120.0
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))
gemini_model_1 = genai.GenerativeModel("gemini-2.5-flash-lite")

# genai.configure(api_key=os.getenv("GEMINI_API_KEY_2"))
# gemini_model_2 = genai.GenerativeModel("gemini-2.5-flash-lite")

# ==== CONFIG ====
BATCH_SIZE = 50
MAX_CONCURRENCY = 6
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
    # resp = model.generate_content(prompt)
    resp = await model.generate_content_async(prompt)

    text = resp.text or "[]"
    try:
        return [clean_line(x) for x in json.loads(text)]
    except Exception:
        return [clean_line(line) for line in text.split("\n") if line.strip()]


async def translate_gemini_1(strings, target_lang, brand_tone):
    return await _translate_with_gemini(strings, target_lang, brand_tone, gemini_model_1)


# async def translate_gemini_2(strings, target_lang, brand_tone):
#     return await _translate_with_gemini(strings, target_lang, brand_tone, gemini_model_2)


# ===================== BATCH REPORTING =====================
def save_batch_report(batch_num, model_name, sent, returned, unchanged_strings, run_dir):
    report = {
        "batch_num": batch_num,
        "model": model_name,
        "sent_count": len(sent),
        "returned_count": len(returned),
        "unchanged_count": len(unchanged_strings),
        "unchanged_strings": unchanged_strings
    }

    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    file_path = os.path.join(reports_dir, f"batch_{batch_num}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"[REPORT] Saved batch_{batch_num}.json → unchanged={len(unchanged_strings)}")


# ===================== BATCH HANDLER =====================
async def _translate_batch(strings, target_lang, brand_tone, batch_num, total_batches, run_dir):
    global model_index
    async with semaphore:
        print(
            f"\n[DEBUG] Batch {batch_num}/{total_batches} → {len(strings)} strings")
        current_model = model_cycle[model_index % len(model_cycle)]
        model_index += 1

        try:
            if current_model == "openai":
                result = await with_retry(translate_openai, strings, target_lang, brand_tone)
            else:
                result = await with_retry(translate_gemini_1, strings, target_lang, brand_tone)
            # else:
            #     result = await with_retry(translate_gemini_2, strings, target_lang, brand_tone)
        except Exception as e:
            print(f"⚠ {current_model} failed, falling back: {e}")
            for alt in model_cycle:
                if alt == current_model:
                    continue
                try:
                    if alt == "openai":
                        result = await translate_openai(strings, target_lang, brand_tone)
                    else:
                        result = await translate_gemini_1(strings, target_lang, brand_tone)
                    # else:
                    #     result = await translate_gemini_2(strings, target_lang, brand_tone)
                    break
                except Exception as e2:
                    print(f"⚠ Fallback {alt} also failed: {e2}")
            else:
                raise Exception("All providers failed!")

        returned_count = len(result)
        unchanged_strings = [s for s, t in zip(strings, result) if s == t]

        # Console summary
        print(f"[BATCH {batch_num}] sent={len(strings)}, returned={returned_count}, unchanged={len(unchanged_strings)} via {current_model}")

        # Save JSON report
        save_batch_report(batch_num, current_model, strings,
                          result, unchanged_strings, run_dir)

        (
            f"[✔] Completed batch {batch_num}/{total_batches} via {current_model}")
        return result

# ===================== MAIN TRANSLATOR =====================


async def fast_translate_json(data, target_lang, brand_tone):
    positions = []  # (path, string, path_str)

    # if "fullData" in data and "storeData" in data["fullData"]:
    #     target_data = data["fullData"]["storeData"]
    # else:
    #     print("⚠ No fullData.storeData found")
    #     return data
    target_data = data

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

    collect_strings(target_data)

    strings_to_translate = [s for _, s, _ in positions]
    print(f"Total strings: {len(strings_to_translate)}")

    # ---- RUN DIR ----
    run_dir = os.path.join(
        LOG_DIR, f"partial_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    # ---- SAVE EXTRACTED ----
    extracted_log = [{"path": p, "string": s} for _, s, p in positions]
    extracted_file = os.path.join(
        LOG_DIR, f"latest_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(extracted_file, "w", encoding="utf-8") as f:
        json.dump(extracted_log, f, ensure_ascii=False, indent=2)
    print(f"Saved extracted strings to {extracted_file}")

    # ---- TRANSLATE ----
    batches = [strings_to_translate[i:i+BATCH_SIZE]
               for i in range(0, len(strings_to_translate), BATCH_SIZE)]
    total_batches = len(batches)

    tasks = [_translate_batch(batch, target_lang, brand_tone, idx+1, total_batches, run_dir)
             for idx, batch in enumerate(batches)]
    batch_results = await asyncio.gather(*tasks)

    # Replaces the above gather to process as completed to avoid waiting for all to finish
    # results = []
    # for coro in asyncio.as_completed(tasks):
    #     res = await coro
    #     results.append(res)

    # batch_results = results

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

        # Add original string with "original" prefix
        prefixed_key = f"original{last_key[0].upper()}{last_key[1:]}"
        if prefixed_key not in ref:  # avoid overwriting if already exists
            ref[prefixed_key] = original_value

        # Replace main field with translated string
        ref[last_key] = translated

    injected_log = []
    string_index = 0
    for batch, result in zip(batches, batch_results):
        # enforce same length
        if len(result) != len(batch):
            print(
                f"⚠ Length mismatch: batch={len(batch)}, result={len(result)}")
            # pad missing translations with originals
            if len(result) < len(batch):
                result.extend(batch[len(result):])
            else:
                result = result[:len(batch)]

        for orig, translated in zip(batch, result):
            path, orig_val, path_str = positions[string_index]

            set_value_with_original(target_data, path, translated)

            injected_log.append({
                "path": path_str,
                "original": orig_val,
                "translated": translated
            })
            # print(f"[INJECT] {path_str} -> {translated[:60]}")
            string_index += 1

    # ---- SAVE INJECTED ----
    injected_file = os.path.join(
        LOG_DIR, f"injected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(injected_file, "w", encoding="utf-8") as f:
        json.dump(injected_log, f, ensure_ascii=False, indent=2)
    print(f"Saved injected strings to {injected_file}")

    print(f" Injected {string_index}/{len(strings_to_translate)} strings")
    return target_data
