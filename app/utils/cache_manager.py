import json
import os
import hashlib
import copy  # For deep copy on cache loads (prevents mutations)
from redis import Redis
from app.config import settings
import uuid

# Initialize Redis client
try:
    redis_client = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    print("Redis connected successfully.")
except Exception as e:
    print(f"Redis connection failed: {e}. Falling back to no-cache mode.")
    redis_client = None  # Disable cache if down


def _make_cache_key(shop_domain: str, target_lang: str, brand_tone: str, text: str) -> str:
    raw = f"{shop_domain}:{target_lang}:{brand_tone}:{text.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()

# UPDATED: Helper for per-string keys (now supports global reuse)


def _make_cache_key_flex(target_lang: str, brand_tone: str, text: str, include_domain: bool = False, shop_domain: str = "") -> str:
    raw = f"{shop_domain}:{target_lang}:{brand_tone}:{text.strip().lower()}" if include_domain else f"{target_lang}:{brand_tone}:{text.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()

# NEW: Helper for full JSON keys (simple, non-hashed for readability)


def _make_full_cache_key(shop_domain: str, target_lang: str, brand_tone: str, prefix: str = "translate") -> str:
    return f"globalflow:{prefix}:{shop_domain}:{target_lang}:{brand_tone}"

# Existing per-string functions (updated to use flex key for global option)


def get_translation_from_cache(shop_domain: str, target_lang: str, brand_tone: str, text: str, include_domain: bool = True):
    if not redis_client:
        return None
    key = _make_cache_key_flex(target_lang, brand_tone, text,
                               include_domain=include_domain, shop_domain=shop_domain)
    cached_value = redis_client.get(key)
    if cached_value:
        print(f"Cache hit: {text[:40]}...")
        return json.loads(cached_value) # type: ignore
    return None


def set_translation_in_cache(shop_domain: str, target_lang: str, brand_tone: str, text: str, translated_text: str, include_domain: bool = True):
    if not redis_client:
        return
    key = _make_cache_key_flex(target_lang, brand_tone, text,
                               include_domain=include_domain, shop_domain=shop_domain)
    redis_client.setex(key, 60 * 60 * 24 * 30,
                       json.dumps(translated_text))  # 30-day TTL
    print(f"Cached: {text[:40]}...")


def delete_translation_cache(shop_domain: str, target_lang: str, brand_tone: str, text: str, include_domain: bool = True):
    if not redis_client:
        return
    key = _make_cache_key_flex(target_lang, brand_tone, text,
                               include_domain=include_domain, shop_domain=shop_domain)
    redis_client.delete(key)
    print(f"Cache cleared: {text[:40]}...")

# NEW: Hash Helper


def compute_raw_hash(raw_data: dict) -> str:
    """Compute SHA256 hash of sorted JSON for change detection."""
    sorted_json = json.dumps(raw_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(sorted_json.encode()).hexdigest()

# NEW: Full JSON Caching Functions (hash-aware)


def get_full_translation_from_cache(shop_domain: str, target_lang: str, brand_tone: str, current_raw_hash: str) -> dict | None:
    """
    Load full translated JSON if raw_hash matches. Returns deep copy.
    """
    if not redis_client:
        return None
    key = _make_full_cache_key(shop_domain, target_lang, brand_tone)
    cached_value = redis_client.get(key)
    if cached_value:
        try:
            cache_data = json.loads(cached_value) # type: ignore
            if cache_data.get("raw_hash") == current_raw_hash:
                print(f"Full JSON cache HIT (hash match) for {shop_domain}:{target_lang}:{brand_tone}")
                file_name = f"Today_translated_{uuid.uuid4().hex}.json"
                file_path = os.path.join("tmp", file_name)
                os.makedirs("tmp", exist_ok=True)

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(cache_data["translated"], f, ensure_ascii=False, indent=2)
                print("Translated JSON saved to file:", file_path)
                return copy.deepcopy(cache_data["translated"])
            else:
                print(
                    f"Full JSON cache MISS (hash mismatch: {cache_data.get('raw_hash')} != {current_raw_hash})")
                return None
        except Exception as e:
            print(f"Cache parse error: {e}")
    return None


def set_full_translation_in_cache(shop_domain: str, target_lang: str, brand_tone: str, translated_data: dict, raw_hash: str, ttl: int = 3600):  # 1h TTL
    """
    Store full translated JSON + raw_hash with TTL.
    """
    if not redis_client:
        return
    key = _make_full_cache_key(shop_domain, target_lang, brand_tone)
    cache_obj = {
        "translated": translated_data,
        "raw_hash": raw_hash
    }
    serialized = json.dumps(cache_obj, ensure_ascii=False)
    redis_client.setex(key, ttl, serialized)
    print(
        f"Full JSON cached (with hash {raw_hash[:8]}...) for {shop_domain}:{target_lang}:{brand_tone} (TTL: {ttl}s)")


def invalidate_full_translation_cache(shop_domain: str, target_lang: str, brand_tone: str):
    """
    Invalidate full JSON cache (e.g., after updates).
    """
    if not redis_client:
        return
    key = _make_full_cache_key(shop_domain, target_lang, brand_tone)
    deleted = redis_client.delete(key)
    if deleted:
        print(
            f"Full JSON cache INVALIDATED for {shop_domain}:{target_lang}:{brand_tone}")
    # Optional: Invalidate raw fetch too (though we're not caching raw)
    raw_key = _make_full_cache_key(shop_domain, target_lang, brand_tone, "raw")
    redis_client.delete(raw_key)


def get_cached_classification(target_lang, brand_tone, text):
    key = _make_cache_key_flex(
        target_lang, brand_tone, text, include_domain=False)
    cached = redis_client.get(f"classify:{key}") # type: ignore
    return json.loads(cached) if cached else None # type: ignore


def set_cached_classification(target_lang, brand_tone, text, label):
    key = _make_cache_key_flex(
        target_lang, brand_tone, text, include_domain=False)
    redis_client.setex(f"classify:{key}", 60 * # type: ignore
                       60*24*7, json.dumps(label))  # 7d TTL

# Similar for vote: key on f"{text}:{initial_label}"


# cache_key = f"{text}:{initial_label}"
def get_cached_vote(target_lang, brand_tone, cache_key):
    flex_key = _make_cache_key_flex(
        target_lang, brand_tone, cache_key, include_domain=False)
    cached = redis_client.get(f"vote:{flex_key}") # type: ignore
    return json.loads(cached) if cached else None # type: ignore


def set_cached_vote(target_lang, brand_tone, cache_key, vote):
    flex_key = _make_cache_key_flex(
        target_lang, brand_tone, cache_key, include_domain=False)
    redis_client.setex(f"vote:{flex_key}", 60*60*24*7, json.dumps(vote)) # type: ignore


def get_cached_full_process(target_lang, brand_tone, text):
    key = _make_cache_key_flex(
        target_lang, brand_tone, text, include_domain=False)
    cached = redis_client.get(f"fullprocess:{key}") # type: ignore
    return json.loads(cached) if cached else None # type: ignore


# {"voted_label": , "translation": }
def set_cached_full_process(target_lang, brand_tone, text, process_data):
    key = _make_cache_key_flex(
        target_lang, brand_tone, text, include_domain=False)
    redis_client.setex( # type: ignore
        f"fullprocess:{key}", 60*60*24*30, json.dumps(process_data))  # 30d