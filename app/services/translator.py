import random
import asyncio
from openai import AsyncOpenAI
import os


client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=200.0
)


# Adjustable settings
BATCH_SIZE = 300  # 300 strings per batch

MAX_CONCURRENCY = 6  # 20 parallel calls

semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


async def with_retry(fn, *args, retries=5, **kwargs):
    for i in range(retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if "Rate limit" in str(e):
                wait = (2 ** i) + random.random()
                print(f"⚠ Rate limit, retrying in {wait:.2f}s...")
                await asyncio.sleep(wait)
            else:
                raise
    raise Exception("Max retries reached")


async def _translate_batch(strings, target_lang, brand_tone, batch_num, total_batches):
    async with semaphore:
        prompt = f"""
        Translate the following {len(strings)} strings into {target_lang}.
        Maintain the brand tone as '{brand_tone}'.
        Return ONLY translations line by line, same order:
        """
        for i, s in enumerate(strings, 1):
            prompt += f"{i}. {s}\n"

        resp = await with_retry(
            client.chat.completions.create,
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        print(f"[✔] Completed batch {batch_num}/{total_batches}")
        return resp.choices[0].message.content.strip().split("\n")


async def fast_translate_json(data, target_lang, brand_tone):
    """Super-fast concurrent translation of JSON"""

    # 1. Collect strings
    strings = []

    def collect_strings(d):
        if isinstance(d, str):
            strings.append(d)
        elif isinstance(d, dict):
            for v in d.values():
                collect_strings(v)
        elif isinstance(d, list):
            for item in d:
                collect_strings(item)

    collect_strings(data)

    print(f"Total strings to translate: {len(strings)}")

    # 2. Make batches
    batches = [strings[i:i+BATCH_SIZE]
               for i in range(0, len(strings), BATCH_SIZE)]
    total_batches = len(batches)

    # 3. Launch parallel tasks
    tasks = [
        _translate_batch(batch, target_lang, brand_tone, idx+1, total_batches)
        for idx, batch in enumerate(batches)
    ]
    batch_results = await asyncio.gather(*tasks)

    # 4. Flatten results
    translated_lines = [line.strip()
                        for batch in batch_results for line in batch if line.strip()]
    it = iter(translated_lines)

    # 5. Replace strings in original JSON
    def replace_strings(d):
        if isinstance(d, str):
            return next(it, d)
        elif isinstance(d, dict):
            return {k: replace_strings(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [replace_strings(item) for item in d]
        return d

    final_data = replace_strings(data)

    # ✅ Final summary
    print(
        f"✅ Translation completed: {len(translated_lines)}/{len(strings)} strings translated")

    return final_data


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
