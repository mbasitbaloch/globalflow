# app/services/suggestion_engine.py
import re
import json
import uuid
from typing import List, Dict, Any, Tuple
from datetime import datetime
from openai import AsyncOpenAI
import google.generativeai as genai
from ..config import settings
from ..utils.cache_manager import (
    get_suggestions_from_cache,
    set_suggestions_in_cache,
    get_stylepack_from_cache,
    set_stylepack_in_cache
)
from ..database import SessionLocal
from ..models.suggestion import Suggestion, SuggestionAudit
import hashlib
import logging
import asyncio

logger = logging.getLogger("suggestion_engine")

# init clients
openai_async = AsyncOpenAI(api_key=settings.OPENAI_API_KEY_1, timeout=60)
genai.configure(api_key=settings.GEMINI_API_KEY_1)

# ----------------- helpers -----------------


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def default_style_pack(language_pair: str, domain: str, country: str) -> dict:
    # sensible defaults; actual tenants should override
    return {
        "language_pair": language_pair,
        "domain": domain,
        "country": country,
        "tone": "neutral",
        "sentence_length_range": [8, 25],
        "idiom_preferences": [],
        "do_not_change": [],  # glossary terms (filled per tenant)
        "formatting": {"dates": "locale", "numbers": "locale"},
    }

# ----------------- load tenant style pack -----------------


def load_style_pack(tenant_id: str, language_pair: str, domain: str, country: str) -> dict:
    cached = get_stylepack_from_cache(
        tenant_id, language_pair, domain, country)
    if cached:
        return cached
    # Attempt: load from DB or disk; fallback to default
    # If you have tenant style packs in DB, replace this with DB fetch
    style_pack = default_style_pack(language_pair, domain, country)
    set_stylepack_in_cache(tenant_id, language_pair,
                           domain, country, style_pack)
    return style_pack

# ----------------- compliance / glossary gates -----------------


def find_forbidden_matches(
    text: str, glossary: List[str], compliance_patterns: List[str]
) -> Dict[str, List[str]]:
    """
    Return dict with keys 'glossary' and 'compliance' containing matched tokens.
    Handles invalid regex patterns gracefully and logs only once globally.
    """
    matches = {"glossary": [], "compliance": []}
    bad_patterns = []

    for g in glossary or []:
        if not g:
            continue
        try:
            if re.search(re.escape(g), text, flags=re.IGNORECASE):
                matches["glossary"].append(g)
        except re.error as e:
            bad_patterns.append(g)
            logger.warning(f"Invalid glossary token skipped: {g} ({e})")

    for pat in compliance_patterns or []:
        if not pat:
            continue
        try:
            if re.search(pat, text, flags=re.IGNORECASE):
                matches["compliance"].append(pat)
        except re.error as e:
            bad_patterns.append(pat)
            logger.warning(f"Invalid regex pattern skipped: {pat} ({e})")

    if bad_patterns:
        logger.debug(f"Ignored invalid patterns: {bad_patterns}")

    return matches


# ----------------- Update translation JSON -----------------
async def update_translation_json(translation_obj, path: str, new_value: str):
    """
    Traverses translation JSON by path and updates only that string.
    """
    data = translation_obj.translated_text_json
    if isinstance(data, str):
        data = json.loads(data)

    keys = path.split(".")
    ref = data

    try:
        for k in keys[:-1]:
            ref = ref[int(k)] if k.isdigit() else ref[k]

        last_key = keys[-1]
        ref[last_key] = new_value
        logger.info(f"Updated JSON path: {path} → {new_value}")
        return data

    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.error(f" Invalid JSON path `{path}`: {e}")
        raise ValueError(f"Invalid JSON path: {path}") from e

# ----------------- meaning-shift classifier -----------------


async def meaning_shift_risk(original: str, candidate: str, doc_type: str) -> Tuple[str, float]:
    """
    Returns (risk_level, confidence_score)
    risk_level: "low"|"medium"|"high"
    confidence_score: 0..1 (how confident that risk is low)
    We use an LLM call to judge whether candidate changes meaning.
    """
    prompt = (
        "You are a classifier. Given ORIGINAL and CANDIDATE texts, answer JSON with keys:\n"
        "'risk' one of ['low','medium','high'] and 'confidence' 0..1.\n"
        "Risk indicates likelihood the candidate changes legally/semantically important meaning.\n"
        f"doc_type: {doc_type}\n\n"
        f"ORIGINAL: {original}\n\n"
        f"CANDIDATE: {candidate}\n\n"
        "Respond only with valid JSON like: {{\"risk\":\"low\",\"confidence\":0.92}}"
    )
    try:
        resp = await openai_async.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        content = resp.choices[0].message.content
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*", "", content).strip("`").strip()
        j = json.loads(content)
        risk = j.get("risk", "medium")
        conf = float(j.get("confidence", 0.5))
        if risk not in ("low", "medium", "high"):
            risk = "medium"
        return risk, max(0.0, min(1.0, conf))
    except Exception as e:
        # On failure: be conservative for Legal
        logger.exception("meaning_shift_risk failed: %s", e)
        if doc_type and doc_type.lower().startswith("legal"):
            return "high", 0.2
        return "medium", 0.5

# ----------------- generate suggestions using constrained LLM prompt -----------------


async def generate_candidate_suggestions(segment_text: str,
                                         style_pack: dict,
                                         language_pair: str,
                                         target_country: str,
                                         n: int = 3
                                         ) -> List[Dict[str, Any]]:
    """
    Returns a list of candidate suggestions (type, after, rationale, confidence).
    Adds localization awareness based on target_language and target_country.
    """
    # We will ask the model to produce a JSON array of suggestion objects,
    # each with type ∈ {grammar,fluency,style,idiom}, after, rationale, confidence (0..1).
    prompt = f"""
You are a constrained text editor and linguistic expert.

Task:
Propose up to {n} non-factual-edit suggestions (grammar, fluency, style, idiom) for the given text.


Localization Rule:
- The text is written in **{language_pair}** as used in **{target_country}**.
- Make sure all suggestions align with how this language is naturally written and spoken in that country.
- Adjust spelling, idioms, tone, and phrasing to match the country's local variant.
  Examples:
  - English (United States): "color", "customization"
  - English (United Kingdom): "colour", "customisation"
  - French (France) vs French (Canada): adapt expressions and tone accordingly
  - Arabic (Egypt) vs Arabic (Saudi Arabia): use local vocabulary
- Keep the meaning unchanged; only improve fluency or localization accuracy.


Rules:
- Do NOT invent new facts or numbers.
- Keep glossary terms stable.
- Do NOT change tokens that look like SKUs, URLs, emails, or numbers.
- Each suggestion object must have: type, after, rationale, confidence (0..1).
- Provide only JSON: an array of objects.

Style pack (brief): {json.dumps(style_pack, ensure_ascii=False)}
Source: {segment_text}
"""
    try:
        resp = await openai_async.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*", "", content).strip("`").strip()
        data = json.loads(content)
        # Validate basic shape
        out = []
        for item in data:
            if "type" in item and "after" in item:
                out.append({
                    "id": str(uuid.uuid4()),
                    "type": item.get("type"),
                    "after": item.get("after"),
                    "rationale": item.get("rationale", ""),
                    "confidence": float(item.get("confidence", 0.5))
                })
        return out
    except Exception as e:
        logger.exception("generate_candidate_suggestions failed: %s", e)
        # fallback heuristic: simple grammar swaps — minimal
        return [{
            "id": str(uuid.uuid4()),
            "type": "fluency",
            "after": segment_text,
            "rationale": "fallback — no changes",
            "confidence": 0.4
        }]

# ----------------- main public API function -----------------


async def produce_suggestions(
    tenant_id: str,
    doc_type: str,
    domain: str,
    country: str,
    language_pair: str,
    preserve_legal_meaning: bool,
    segments: List[Dict[str, Any]],
    glossary: List[str] = None,
    compliance_patterns: List[str] = None
) -> Dict[str, Any]:
    """
    For each segment {id,text} produce suggestions per the spec and return suggestions + scores.
    This function:
    - loads stylepack
    - checks cache for existing suggestions
    - generates candidates
    - computes blocked flags, risk using classifier, and final structured suggestion objects
    - caches result
    """
    style_pack = load_style_pack(tenant_id, language_pair, domain, country)
    glossary = glossary or style_pack.get("do_not_change", []) or []
    compliance_patterns = compliance_patterns or []

    aggregated_suggestions = []
    # track scores (very simple aggregated)
    scores = {"grammar": 0.0, "fluency": 0.0, "style": 0.0}
    score_counts = {"grammar": 0, "fluency": 0, "style": 0}

    # parallel generation for speed
    tasks = []
    for seg in segments:
        seg_path = seg["path"]
        text = seg["text"]
        cached = get_suggestions_from_cache(
            tenant_id, language_pair, domain, text)
        if cached:
            logger.info(f"[cache-hit] {seg_path}")
            aggregated_suggestions.append({
                "path": seg_path,
                "original": text,
                "suggestions": cached["suggestions"],
                "scores": cached.get("scores", {})
            })
            continue
        # else schedule generation
        tasks.append((seg_path, text))

    async def _process_one(path, text):
        logger.info(f"[generate] Creating candidates for: {path}")
        # candidates = await generate_candidate_suggestions(text, style_pack, n=4)
        candidates = await generate_candidate_suggestions(
            segment_text=text,
            style_pack=style_pack,
            language_pair=language_pair,
            target_country=country,
            n=5
        )
        final = []
        for c in candidates:
            # blocked checks
            blocked_reasons = []
            matches = find_forbidden_matches(
                c["after"], glossary, compliance_patterns)
            if matches["glossary"]:
                blocked_reasons.append("glossary")
            if matches["compliance"]:
                blocked_reasons.append("compliance")
            # numbers / sku / urls detection
            if re.search(r"\b\d{2,}\b", c["after"]):
                # numbers present — block if they differ from original numbers
                # coarse check: if number set different between orig & after mark blocked
                orig_nums = set(re.findall(r"\b\d+\b", text))
                new_nums = set(re.findall(r"\b\d+\b", c["after"]))
                if orig_nums != new_nums:
                    blocked_reasons.append("numbers")

            risk, risk_conf = await meaning_shift_risk(text, c["after"], doc_type)
            blocked = len(blocked_reasons) > 0

            suggestion_obj = {
                "suggestion_id": c.get("id") or _sha1(c.get("after", "")[:64]),
                "type": c.get("type"),
                "before": text,
                "after": c.get("after"),
                "rationale": c.get("rationale", ""),
                "confidence": float(c.get("confidence", 0.5)),
                "blocked": blocked,
                "blocks": blocked_reasons,
                "risk": risk,
                "risk_confidence": float(risk_conf)
            }

            final.append(suggestion_obj)

            # accumulate scores simply by confidence mapped to type
            t = suggestion_obj["type"]
            if t in scores:
                scores[t] += suggestion_obj["confidence"] * 100
                score_counts[t] += 1

        # default scoring if none
        local_scores = {}
        for k in ("grammar", "fluency", "style"):
            if score_counts[k] > 0:
                local_scores[k] = int(scores[k] / max(1, score_counts[k]))
            else:
                local_scores[k] = 0

        # cache per segment to avoid repeated LLM calls
        payload = {"suggestions": final,
                   "scores": local_scores,
                   "generated_at": datetime.utcnow().isoformat()
                   }
        set_suggestions_in_cache(
            tenant_id, language_pair, domain, text, payload)
        return {"path": path, "original": text, "suggestions": final, "scores": local_scores}

    # # spawn tasks
    # proc = [asyncio.create_task(_process_one(seg_id, text))
    #         for seg_id, text in tasks]
    # if proc:
    #     done = await asyncio.gather(*proc, return_exceptions=False)
    #     aggregated_suggestions.extend(done)

    if tasks:
        results = await asyncio.gather(*[asyncio.create_task(_process_one(path, text)) for path, text in tasks])
        aggregated_suggestions.extend(results)

    # overall aggregated scores average
    overall_scores = {}
    for k in ("grammar", "fluency", "style"):
        if score_counts[k] > 0:
            overall_scores[k] = int((scores[k] / score_counts[k]))
        else:
            overall_scores[k] = 0
    logger.info(
        f"[{tenant_id}] Processed {len(aggregated_suggestions)} segments total")
    logger.info(f"Scores summary: {overall_scores}")

    return {"suggestions": aggregated_suggestions, "scores": overall_scores}
