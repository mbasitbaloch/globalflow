import json
import random
import re
import asyncio
import os
import uuid
from openai import AsyncOpenAI
# import google.generativeai as genai  # Gemini SDK
from google import genai
# from google.genai import types


openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=120.0
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==== CONFIG ====
BATCH_SIZE = 100
MAX_CONCURRENCY = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# round-robin
model_cycle = ["openai", "gemini"]
model_index = 0

# ========== FILTER HELPERS ==========


def is_translateable(text: str) -> bool:
    """Skip IDs, hashes, timestamps, numbers, placeholders, empty strings"""
    if not text or not text.strip():
        return False
    if text.isdigit():
        return False
    if re.match(r"^\d+(\.\d+)?$", text):  # numbers like 129.00
        return False
    if re.match(r"^\d{4}-\d{2}-\d{2}T", text):  # ISO timestamps
        return False
    if re.match(r"^[a-f0-9]{32,64}$", text):  # hashes
        return False
    if text.startswith("gid://"):
        return False
    if re.match(r"^\{\{.*\}\}$", text):  # {{placeholders}}
        return False
    if "@" in text and "." in text:  # emails
        return False
    if re.match(r"^https?://", text):  # URLs
        return False
    if text.startswith("shopify."):
        return False
    if text.startswith("customer."):
        return False
    if text.startswith("customer_"):
        return False
    if text.startswith("templates."):
        return False
    return True


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


# ==== TRANSLATION FUNCTIONS ====
async def translate_openai(strings, target_lang, brand_tone):
    prompt = f"""
    Translate the following {len(strings)} strings into {target_lang}.
    Maintain the brand tone as '{brand_tone}'.
    ⚠ If a string contains HTML tags (<p>, <div>, <br>, etc.), keep the tags exactly as they are, 
    and only translate the inner text.
    Return ONLY translations line by line, same order:
    """
    for i, s in enumerate(strings, 1):
        prompt += f"{i}. {s}\n"

    resp = await openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content = resp.choices[0].message.content
    if content is None:
        return []
    return content.strip().split("\n")


async def translate_gemini(strings, target_lang, brand_tone):
    prompt = f"""
    Translate the following {len(strings)} strings into {target_lang}.
    Maintain the brand tone as '{brand_tone}'.
    ⚠ If a string contains HTML tags (<p>, <div>, <br>, etc.), keep the tags exactly as they are,
    and only translate the inner text.
    Return ONLY translations line by line, same order:
    """
    for i, s in enumerate(strings, 1):
        prompt += f"{i}. {s}\n"

    # resp = gemini_model.generate_content(prompt)

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    lines = resp.text.strip().split("\n") if resp.text is not None else []

    return [line.strip() for line in lines if line.strip()]


async def _translate_batch(strings, target_lang, brand_tone, batch_num, total_batches):
    global model_index
    async with semaphore:
        print(
            f"\n[DEBUG] Batch {batch_num}/{total_batches} using {len(strings)} strings")
        for i, s in enumerate(strings[:5], 1):
            print(f"   {i}. {s[:120]}")

        current_model = model_cycle[model_index % len(model_cycle)]
        model_index += 1

        try:
            if current_model == "openai":
                result = await with_retry(translate_openai, strings, target_lang, brand_tone)
            else:
                result = await with_retry(translate_gemini, strings, target_lang, brand_tone)
        except Exception as e:
            print(f"⚠ {current_model} failed, falling back: {e}")
            if current_model == "openai":
                result = await translate_gemini(strings, target_lang, brand_tone)
            else:
                result = await translate_openai(strings, target_lang, brand_tone)

        print(
            f"[✔] Completed batch {batch_num}/{total_batches} via {current_model}")
        return result


# ==== MAIN JSON TRANSLATOR ====
async def fast_translate_json(data, target_lang, brand_tone):
    string_map = {}
    positions = []

    # Restrict to only fullData.storeData
    if "fullData" in data and "storeData" in data["fullData"]:
        target_data = data["fullData"]["storeData"]
    else:
        print("⚠ No fullData.storeData found, skipping translation.")
        return data
    print("data into fullData is:", target_data)

    # Save translated JSON to file
    file_name = f"fullData{uuid.uuid4().hex}.json"
    file_path = os.path.join("fullData", file_name)
    os.makedirs("fullData", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(target_data, f, ensure_ascii=False, indent=2)

    # Step 1: collect translateable strings with their positions

    def collect_strings(d, path=None):
        if path is None:
            path = []
        if isinstance(d, str):
            if is_translateable(d):
                positions.append((path, d))
        elif isinstance(d, dict):
            for k, v in d.items():
                collect_strings(v, path + [k])
        elif isinstance(d, list):
            for i, item in enumerate(d):
                collect_strings(item, path + [i])

    collect_strings(target_data)

    unique_strings = list({s for _, s in positions})
    print(f"Total unique translateable strings: {len(unique_strings)}")

    # Step 2: batch unique strings → translate
    batches = [unique_strings[i:i+BATCH_SIZE]
               for i in range(0, len(unique_strings), BATCH_SIZE)]
    total_batches = len(batches)

    tasks = [
        _translate_batch(batch, target_lang, brand_tone, idx+1, total_batches)
        for idx, batch in enumerate(batches)
    ]
    batch_results = await asyncio.gather(*tasks)
    translated_lines = [line.strip()
                        for batch in batch_results for line in batch if line.strip()]

    # map unique originals → translated
    for orig, trans in zip(unique_strings, translated_lines):
        string_map[orig] = trans

    # Step 3: inject back into JSON (using positions)
    def set_value(d, path, value):
        ref = d
        for p in path[:-1]:
            ref = ref[p]
        ref[path[-1]] = value

    for path, orig in positions:
        if orig in string_map:
            set_value(target_data, path, string_map[orig])

    print(
        f"✅ Translation completed: {len(string_map)}/{len(unique_strings)} unique strings translated")
    # return data
    return target_data