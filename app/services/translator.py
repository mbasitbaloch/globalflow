import json
import random
import re
import asyncio
import os
import uuid
from openai import AsyncOpenAI
import google.generativeai as genai  # Gemini SDK
# from google import genai
# from google.genai import types


openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=120.0
)

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


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

    resp = gemini_model.generate_content(prompt)

    # resp = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents=prompt,
    # )

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


# ------updated code of translator.py on 9-4-2025-------
# import random
# import re
# import asyncio
# import os
# from openai import AsyncOpenAI
# import google.generativeai as genai  # Gemini SDK

# openai_client = AsyncOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     timeout=120.0
# )

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# # ==== CONFIG ====
# BATCH_SIZE = 100
# MAX_CONCURRENCY = 3
# semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# # keep track of which model to use (round robin)
# model_cycle = ["openai", "gemini"]
# model_index = 0


# # ========== FILTER HELPERS ==========
# def is_translateable(text: str) -> bool:
#     """Skip IDs, hashes, timestamps, numbers, placeholders, empty strings"""
#     if not text or not text.strip():
#         return False
#     if text.isdigit():
#         return False
#     if re.match(r"^\d+(\.\d+)?$", text):  # numbers like 129.00, 0.00
#         return False
#     if re.match(r"^\d{4}-\d{2}-\d{2}T", text):  # ISO timestamps
#         return False
#     if re.match(r"^[a-f0-9]{32,64}$", text):  # hashes
#         return False
#     if text.startswith("gid://"):
#         return False
#     if re.match(r"^\{\{.*\}\}$", text):  # {{placeholders}}
#         return False
#     if "@" in text and "." in text:  # emails
#         return False
#     if re.match(r"^https?://", text):  # URLs
#         return False
#     return True


# async def with_retry(fn, *args, retries=3, **kwargs):
#     for i in range(retries):
#         try:
#             return await fn(*args, **kwargs)
#         except Exception as e:
#             if "Rate limit" in str(e) or "quota" in str(e).lower():
#                 wait = (2 ** i) + random.random()
#                 print(f"⚠ Rate limit, retrying in {wait:.2f}s...")
#                 await asyncio.sleep(wait)
#             else:
#                 print(f"⚠ Error: {e}, retrying...")
#                 await asyncio.sleep(2)
#     raise Exception("Max retries reached")


# async def translate_openai(strings, target_lang, brand_tone):
#     prompt = f"""
#     Translate the following {len(strings)} strings into {target_lang}.
#     Maintain the brand tone as '{brand_tone}'.
#     Return ONLY translations line by line, same order:
#     """
#     for i, s in enumerate(strings, 1):
#         prompt += f"{i}. {s}\n"

#     resp = await openai_client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=1,
#     )
#     return resp.choices[0].message.content.strip().split("\n")


# async def translate_gemini(strings, target_lang, brand_tone):
#     prompt = f"""
#     Translate the following {len(strings)} strings into {target_lang}.
#     Maintain the brand tone as '{brand_tone}'.
#     Return ONLY translations line by line, same order:
#     """
#     for i, s in enumerate(strings, 1):
#         prompt += f"{i}. {s}\n"

#     resp = gemini_model.generate_content(prompt)
#     lines = resp.text.strip().split("\n")
#     return [line.strip() for line in lines if line.strip()]


# async def _translate_batch(strings, target_lang, brand_tone, batch_num, total_batches):
#     global model_index
#     async with semaphore:
#         print(
#             f"\n[DEBUG] Batch {batch_num}/{total_batches} using {len(strings)} strings")
#         for i, s in enumerate(strings[:5], 1):
#             print(f"   {i}. {s[:120]}")
#         current_model = model_cycle[model_index % len(model_cycle)]
#         model_index += 1

#         try:
#             if current_model == "openai":
#                 result = await with_retry(translate_openai, strings, target_lang, brand_tone)
#             else:
#                 result = await with_retry(translate_gemini, strings, target_lang, brand_tone)
#         except Exception as e:
#             print(f"⚠ {current_model} failed, falling back: {e}")
#             if current_model == "openai":
#                 result = await translate_gemini(strings, target_lang, brand_tone)
#             else:
#                 result = await translate_openai(strings, target_lang, brand_tone)

#         print(
#             f"[✔] Completed batch {batch_num}/{total_batches} via {current_model}")
#         return result


# async def fast_translate_json(data, target_lang, brand_tone):
#     strings = []
#     positions = []

#     # Step 1: collect only translateable strings + their positions
#     def collect_strings(d, path=None):
#         if path is None:
#             path = []
#         if isinstance(d, str):
#             if is_translateable(d):
#                 strings.append(d)
#                 positions.append(path)
#         elif isinstance(d, dict):
#             for k, v in d.items():
#                 collect_strings(v, path + [k])
#         elif isinstance(d, list):
#             for i, item in enumerate(d):
#                 collect_strings(item, path + [i])

#     collect_strings(data)
#     print(f"Total translateable strings: {len(strings)}")

#     # Step 2: batch + send for translation
#     batches = [strings[i:i+BATCH_SIZE]
#                for i in range(0, len(strings), BATCH_SIZE)]
#     total_batches = len(batches)

#     tasks = [
#         _translate_batch(batch, target_lang, brand_tone, idx+1, total_batches)
#         for idx, batch in enumerate(batches)
#     ]
#     batch_results = await asyncio.gather(*tasks)
#     translated_lines = [line.strip()
#                         for batch in batch_results for line in batch if line.strip()]

#     # Step 3: inject translations back into JSON
#     it = iter(translated_lines)

#     def set_value(d, path, value):
#         ref = d
#         for p in path[:-1]:
#             ref = ref[p]
#         ref[path[-1]] = value

#     for path in positions:
#         set_value(data, path, next(it, ""))

#     print(
#         f"✅ Translation completed: {len(translated_lines)}/{len(strings)} strings translated")
#     return data


# import os
# import json
# import asyncio
# from typing import Any, List, Tuple, Dict
# from openai import AsyncOpenAI

# # OPTIONAL (better token sizing). If not installed, we fallback to char-based sizing.
# try:
#     import tiktoken
#     _enc = tiktoken.get_encoding("cl100k_base")

#     def count_tokens(txt: str) -> int:
#         return len(_enc.encode(txt))
# except Exception:
#     def count_tokens(txt: str) -> int:
#         # rough fallback
#         return max(1, len(txt) // 4)

# client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # === CONFIG ===
# MODEL = "gpt-4o-mini"          # fast & cheap; switch to "gpt-4o" if needed
# MAX_TOKENS_PER_BATCH = 12000    # ~prompt size budget per request
# MAX_CONCURRENCY_CAP = 8         # hard cap; we’ll compute dynamic <= this
# SAFETY_SLEEP = 0.05             # tiny jitter between starting tasks


# def _flatten(data: Any, path: Tuple = ()) -> List[Tuple[Tuple, str]]:
#     """Collect all strings with their JSON path."""
#     out = []
#     if isinstance(data, str):
#         out.append((path, data))
#     elif isinstance(data, dict):
#         for k, v in data.items():
#             out.extend(_flatten(v, path + (("k", k),)))
#     elif isinstance(data, list):
#         for i, v in enumerate(data):
#             out.extend(_flatten(v, path + (("i", i),)))
#     return out


# def _set_by_path(root: Any, path: Tuple, value: str):
#     """Write translated string back into original JSON structure."""
#     cur = root
#     for t, key in path[:-1]:
#         if t == "k":
#             cur = cur[key]
#         else:
#             cur = cur[key]
#     t, key = path[-1]
#     if t == "k":
#         cur[key] = value
#     else:
#         cur[key] = value


# def _chunk_by_tokens(items: List[str], max_tokens: int) -> List[List[str]]:
#     """Greedy pack strings into batches under token budget."""
#     batches, cur, cur_tok = [], [], 0
#     # small per-item overhead for prompt JSON/quotes
#     def item_cost(s): return count_tokens(s) + 6
#     for s in items:
#         c = item_cost(s)
#         if cur and (cur_tok + c) > max_tokens:
#             batches.append(cur)
#             cur, cur_tok = [], 0
#         cur.append(s)
#         cur_tok += c
#     if cur:
#         batches.append(cur)
#     return batches


# def _dedup(strings_with_paths: List[Tuple[Tuple, str]]):
#     """Map duplicate strings to a single translation."""
#     uniq: Dict[str, List[int]] = {}
#     uniq_list: List[str] = []
#     for idx, (_, s) in enumerate(strings_with_paths):
#         key = s.strip()
#         if key not in uniq:
#             uniq[key] = []
#             uniq_list.append(key)
#         uniq[key].append(idx)
#     return uniq_list, uniq


# async def _translate_batch(batch: List[str], lang: str, tone: str) -> List[str]:
#     """
#     Ask model to translate an array of strings. Returns list in same order.
#     JSON-enforced output: {"t": ["...", "..."]}.
#     """
#     system = (
#         "You are a professional localization engine. "
#         "Translate text faithfully, preserve numbers, URLs, SKUs, HTML tags, "
#         "double curly placeholders like {{var}} or {0}, and markdown. "
#         "Do not add explanations or change order. "
#         "Always respond in strict JSON format."
#     )

#     user = {
#         "role": "user",
#         "content": json.dumps({
#             "task": "translate_array",
#             "note": "Return output as JSON with key 't'",
#             "target_lang": lang,
#             "brand_tone": tone,
#             "strings": batch
#         }, ensure_ascii=False)
#     }

#     resp = await client.chat.completions.create(
#         model=MODEL,
#         messages=[
#             {"role": "system", "content": system},
#             user
#         ],
#         temperature=0.2,
#         response_format={"type": "json_object"}  # force JSON
#     )

#     content = resp.choices[0].message.content
#     obj = json.loads(content)
#     out = obj.get("t") or obj.get("translations") or obj.get("data")
#     if not isinstance(out, list):
#         return batch
#     if len(out) != len(batch):
#         out = (out + batch)[:len(batch)]
#     return [s if isinstance(s, str) else str(s) for s in out]


# async def fast_translate_json(data: Any, target_lang: str, brand_tone: str) -> Any:
#     """
#     High-speed path:
#       1) Flatten & dedup
#       2) Chunk unique strings by token budget
#       3) Translate chunks in parallel (async)
#       4) Reuse translations for duplicates
#       5) Rebuild original JSON (same shape)
#     """
#     # 1) flatten
#     strings_with_paths = _flatten(data)
#     if not strings_with_paths:
#         return data

#     # 2) dedup
#     uniq_strings, index_map = _dedup(strings_with_paths)

#     # 3) chunk
#     batches = _chunk_by_tokens(uniq_strings, MAX_TOKENS_PER_BATCH)

#     # 4) dynamic concurrency to avoid TPM bursts
#     # rough tokens per batch -> concurrency so sum ~< 200k TPM
#     est_batch_tokens = sum(count_tokens(s)
#                            for s in (batches[0] if batches else [])) + 200
#     if est_batch_tokens <= 0:
#         est_batch_tokens = 2000
#     # if your org TPM ~200k for this model:
#     max_parallel = max(1, min(MAX_CONCURRENCY_CAP, 200000 //
#                        max(1000, est_batch_tokens)))

#     sem = asyncio.Semaphore(max_parallel)

#     async def worker(batch):
#         async with sem:
#             await asyncio.sleep(SAFETY_SLEEP)  # tiny stagger
#             return await _translate_batch(batch, target_lang, brand_tone)

#     tasks = [asyncio.create_task(worker(b)) for b in batches]
#     batch_results = await asyncio.gather(*tasks)

#     # 5) stitch unique translations back
#     uniq_translated: Dict[str, str] = {}
#     idx = 0
#     for b, out in zip(batches, batch_results):
#         for s, t in zip(b, out):
#             uniq_translated[s] = t
#             idx += 1

#     # 6) rebuild original JSON
#     result = json.loads(json.dumps(data))  # deep copy
#     for i, (path, original) in enumerate(strings_with_paths):
#         key = original.strip()
#         translated = uniq_translated.get(key, original)
#         _set_by_path(result, path, translated)

#     return result
