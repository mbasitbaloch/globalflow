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
import aiohttp
from ..validator.countryValidator import validate_language_and_country

logger = logging.getLogger("suggestion_engine")

# Initialize clients
openai_async = AsyncOpenAI(api_key=settings.OPENAI_API_KEY_1, timeout=60)
genai.configure(api_key=settings.GEMINI_API_KEY_1)

# ----------------- Helpers -----------------


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def default_style_pack(language_pair: str, domain: str, country: str) -> dict:
    return {
        "language_pair": language_pair,
        "domain": domain,
        "country": country,
        "tone": "neutral",
        "sentence_length_range": [8, 25],
        "idiom_preferences": [],
        "do_not_change": [],
        "formatting": {"dates": "locale", "numbers": "locale"},
    }

# ----------------- Load Tenant Style Pack -----------------


def load_style_pack(tenant_id: str, language_pair: str, domain: str, country: str) -> dict:
    cached = get_stylepack_from_cache(
        tenant_id, language_pair, domain, country)
    if cached:
        return cached
    style_pack = default_style_pack(language_pair, domain, country)
    set_stylepack_in_cache(tenant_id, language_pair,
                           domain, country, style_pack)
    return style_pack

# ----------------- Compliance / Glossary Gates -----------------


def find_forbidden_matches(
    text: str, glossary: List[str], compliance_patterns: List[str]
) -> Dict[str, List[str]]:
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

# ----------------- Update Translation JSON -----------------


async def update_translation_json(translation_obj, path: str, new_value: str):
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
        logger.error(f"Invalid JSON path `{path}`: {e}")
        raise ValueError(f"Invalid JSON path: {path}") from e

# ----------------- Meaning-Shift Classifier -----------------


async def meaning_shift_risk(original: str, candidate: str, doc_type: str) -> Tuple[str, float]:
    prompt = (
        "You are a classifier. Given ORIGINAL and CANDIDATE texts, return JSON with keys:\n"
        "'risk' (one of ['low', 'medium', 'high']) and 'confidence' (0 to 1).\n"
        "Risk indicates likelihood the candidate changes legally/semantically important meaning.\n"
        f"doc_type: {doc_type}\n\n"
        f"ORIGINAL: {original}\n\n"
        f"CANDIDATE: {candidate}\n\n"
        "Respond ONLY with valid JSON like: {\"risk\": \"low\", \"confidence\": 0.92}"
    )
    try:
        resp = await openai_async.chat.completions.create(
            model="gpt-4.1",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*", "", content).strip("`").strip()
        j = json.loads(content)
        risk = j.get("risk", "medium")
        conf = float(j.get("confidence", 0.5))
        if risk not in ("low", "medium", "high"):
            risk = "medium"
        return risk, max(0.0, min(1.0, conf))
    except Exception as e:
        logger.exception("meaning_shift_risk failed: %s", e)
        return ("high" if doc_type.lower().startswith("legal") else "medium", 0.5)

# ----------------- Generate Suggestions with Retry Logic -----------------


async def generate_candidate_suggestions(
    segment_text: str,
    style_pack: dict,
    language_pair: str,
    target_country: str,
    doc_type: str,
    n: int = 4,
    retries: int = 3
) -> List[Dict[str, Any]]:
    if not segment_text.strip():
        logger.error("Empty segment text provided")
        return [{
            "suggestion_id": str(uuid.uuid4()),
            "type": "fluency",
            "before": segment_text,
            "after": segment_text,
            "rationale": "No suggestions due to empty input text",
            "confidence": 0.4,
            "blocked": True,
            "blocks": ["empty_input"],
            "risk": "low",
            "risk_confidence": 1.0
        }]

    required_types = ["grammar", "fluency", "style", "idioms"]
    system_prompt = (

        "You are a linguistic and editorial assistant. Your job is to provide exactly four distinct improvements, one for each of: grammar, fluency, style, and idiomatic usage. "
        "Return a JSON array of exactly 4 objects, each with keys: type, after, rationale, confidence(0 to 1). "
        "Types must be: grammar, fluency, style, idioms. "
        "For grammar, correct syntax, agreement, or punctuation errors. "
        "For fluency, improve sentence flow and readability. "
        "For style, adjust tone and phrasing to match the document type ({doc_type}) and target country ({target_country}). "
        "For idioms, replace idiomatic phrases with clearer or more natural alternatives; if no idioms are present, use standard legal phrasing for legal documents. "
        "Each suggestion must address the entire input text. "
        "Ensure suggestions are distinct, meaningful, and preserve the exact meaning, especially for legal/contractual terms. "
        "For each suggestion, provide a detailed rationale explaining: 1) what specific text was changed, 2) how it was changed, and 3) why the change improves the text. "
        "Output ONLY a valid JSON array with no markdown or extra text."
    ).format(doc_type=doc_type, target_country=target_country)

    # user_prompt = f"""

    #     Text: {segment_text}
    #     Generate exactly 4 improvement suggestions, one for each of: grammar, fluency, style, idioms.
    #     - Language pair: {language_pair}
    #     - Country: {target_country}
    #     - Document type: {doc_type}
    #     - Preserve the exact meaning, especially legal/contractual terms if applicable.
    #     - Keep placeholders, numbers, and glossary terms unchanged.
    #     - For grammar, correct syntax, agreement, or punctuation errors.
    #     - For fluency, enhance sentence flow and coherence.
    #     - For style, adjust tone (e.g., formal for legal documents) and improve clarity.
    #     - For idioms, replace idiomatic phrases with standard or clearer phrases.
    #     Output ONLY a valid JSON array like:
    #     [
    #     {{"type": "grammar", "after": "Corrected text", "rationale": "Fixed verb agreement", "confidence": 0.95}},
    #     {{"type": "fluency", "after": "Improved text", "rationale": "Enhanced flow", "confidence": 0.92}},
    #     {{"type": "style", "after": "Revised text", "rationale": "Adjusted tone", "confidence": 0.90}},
    #     {{"type": "idioms", "after": "Clearer text", "rationale": "Replaced idioms", "confidence": 0.94}}
    #     ]

    # """

    user_prompt = f"""

        Text: {segment_text}
        Generate exactly 4 improvement suggestions, one for each of: grammar, fluency, style, idioms.
        - Language pair: {language_pair}
        - Country: {target_country}
        - Document type: {doc_type}
        - Preserve the exact meaning, especially legal/contractual terms.
        - Keep placeholders, numbers, and glossary terms unchanged.
        - For grammar, correct syntax, agreement, or punctuation errors, and explain each correction in the rationale.
        - For fluency, enhance sentence flow and coherence, detailing how the structure was improved.
        - For style, adjust tone (e.g., formal for legal documents) and improve clarity, specifying the phrasing changes.
        - For idioms, replace idiomatic phrases with standard or clearer phrases; if no idioms, use standard legal phrasing for legal documents, and explain the substitutions.
        - For each suggestion, provide a detailed rationale that lists: 1) what specific text was changed, 2) how it was changed (e.g., replaced 'X' with 'Y'), and 3) why the change improves the text (e.g., enhances clarity, ensures legal precision).

        Output ONLY a valid JSON array like:

        [
        {{"type": "grammar", "after": "Corrected text", "rationale": "Changed 'X' to 'Y' to fix subject-verb agreement; added semicolon after 'Z' for clarity.", "confidence": 0.95}},
        {{"type": "fluency", "after": "Improved text", "rationale": "Split sentence at 'X' into two sentences to improve readability; replaced 'Y' with 'Z' for smoother flow.", "confidence": 0.92}},
        {{"type": "style", "after": "Revised text", "rationale": "Replaced 'X' with 'Y' to adopt a formal tone suitable for legal documents; removed 'Z' to avoid redundancy.", "confidence": 0.90}},
        {{"type": "idioms", "after": "Clearer text", "rationale": "Replaced idiomatic 'X' with 'Y' for clarity; used 'Z' for standard legal phrasing.", "confidence": 0.94}}
        ]
"""

    fallback_prompt = f"""
            Text: {segment_text}
            Generate exactly 4 improvement suggestions, one for each of: grammar, fluency, style, idioms.
            - Language pair: {language_pair}
            - Country: {target_country}
            - Document type: {doc_type}
            - Preserve the exact meaning, especially legal/contractual terms.
            - For grammar, fix any errors in syntax or punctuation, and list each correction.
            - For fluency, make the text easier to read, explaining structural changes.
            - For style, ensure the tone is formal for legal documents, detailing phrasing changes.
            - For idioms, use standard legal phrasing if no idioms are present, and explain substitutions.
            - For each suggestion, explain what was changed, how it was changed, and why it improves the text.
            Output ONLY a valid JSON array with 4 objects, each with type, after, rationale, and confidence.
        """

    def clean_and_parse(raw: str) -> List[Dict[str, Any]]:
        raw = raw.strip()
        raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip("` \n\t")

        # Fix common JSON issues
        raw = re.sub(r",\s*([\]}])", r"\1", raw)  # Remove trailing commas
        # Fix empty arrays with commas
        raw = re.sub(r"\[\s*,\s*\]", "[]", raw)
        if raw.startswith("{") and not raw.endswith("}"):
            raw = raw + "}"  # Close incomplete objects

        if raw.startswith("[") and not raw.endswith("]"):
            raw = raw + "]"  # Close incomplete arrays
        raw = re.sub(r'("[^"]*")\s*:', r'\1:', raw)  # Ensure proper quoting

        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "suggestions" in data:
                return data["suggestions"] if isinstance(data["suggestions"], list) else []
            return data if isinstance(data, list) else []
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {raw[:200]} - Error: {e}")
            print(f"Failed to parse JSON: {raw[:200]} - Error: {e}")
            return []

    async def try_openai(attempt: int, max_tokens: int = 3000, use_fallback: bool = False) -> List[Dict[str, Any]]:
        try:
            resp = await openai_async.chat.completions.create(
                model="gpt-4.1",
                # response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=max_tokens,
                timeout=30
            )
            content = resp.choices[0].message.content or "[]"
            parsed = clean_and_parse(content)
            if not parsed:
                logger.warning(
                    f"[attempt {attempt}] Empty or invalid JSON: {content[:200]}")
                print(
                    f"[attempt {attempt}] Empty or invalid JSON: {content[:200]}")
                return []
            return parsed
        except Exception as e:
            logger.error(f"[attempt {attempt}] OpenAI failed: {e}")
            print(f"[attempt {attempt}] OpenAI failed: {e}")
            return []

    # async def try_gemini() -> List[Dict[str, Any]]:
    #     try:
    #         model = genai.GenerativeModel("gemini-1.5-pro")
    #         response = await model.generate_content_async(
    #             system_prompt + "\n\n" + user_prompt,
    #             generation_config={"response_mime_type": "application/json"}
    #         )
    #         content = response.text or "[]"
    #         parsed = clean_and_parse(content)
    #         if not parsed:
    #             logger.warning(
    #                 f"Gemini fallback produced invalid JSON: {content[:200]}")
    #             print(
    #                 f"Gemini fallback produced invalid JSON: {content[:200]}")
    #             return []
    #         return parsed
    #     except Exception as e:
    #         logger.error(f"Gemini fallback failed: {e}")
    #         return []

    # Try OpenAI with retries

    for attempt in range(retries):
        parsed = await try_openai(attempt + 1, max_tokens=2000 + attempt * 1000)
        if parsed and len(parsed) >= len(required_types):
            break

    # Normalize results
    normalized, seen = [], set()
    for s in parsed:
        stype = str(s.get("type", "")).lower().strip()
        if stype in required_types and stype not in seen:
            s["suggestion_id"] = str(uuid.uuid4())
            s["before"] = segment_text
            s.setdefault("rationale", f"{stype} improvement")
            s.setdefault("confidence", 0.7)
            s.setdefault("blocked", False)
            s.setdefault("blocks", [])
            s.setdefault("risk", "low")
            s.setdefault("risk_confidence", 1.0)
            normalized.append(s)
            seen.add(stype)

    # Ensure all required types
    for t in required_types:
        if t not in seen:
            normalized.append({
                "suggestion_id": str(uuid.uuid4()),
                "type": t,
                "before": segment_text,
                "after": segment_text,
                "rationale": f"fallback — no {t} change",
                "confidence": 0.4,
                "blocked": True,
                "blocks": ["fallback"],
                "risk": "low",
                "risk_confidence": 1.0
            })

    return sorted(normalized, key=lambda s: required_types.index(s["type"]))

# ----------------- Main Public API Function -----------------


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

    style_pack = load_style_pack(tenant_id, language_pair, domain, country)
    glossary = glossary or style_pack.get("do_not_change", []) or []
    compliance_patterns = compliance_patterns or []
    aggregated_suggestions = []
    scores = {"grammar": 0.0, "fluency": 0.0, "style": 0.0, "idioms": 0.0}
    score_counts = {"grammar": 0, "fluency": 0, "style": 0,
                    "idioms": 0}  # Fixed typo: " fluency" -> "fluency"

    async def _process_one(path: str, text: str):
        logger.info(f"[generate] Creating candidates for: {path}")
        # cached = get_suggestions_from_cache(
        #     tenant_id, language_pair, domain, text)
        # if cached:
        #     logger.info(f"[cache-hit] {path}")
        #     return {"path": path, "original": text, "suggestions": cached["suggestions"], "scores": cached.get("scores", {})}

        # Validate language and country
        valid, validation_msg = validate_language_and_country(
            language_pair,
            country,
            text  # Pass the text string directly
        )
        if not valid:
            logger.error(f"Validation failed for {path}: {validation_msg}")
            # Should not reach here due to early validation
            raise ValueError(validation_msg)

        candidates = await generate_candidate_suggestions(
            segment_text=text,
            style_pack=style_pack,
            language_pair=language_pair,
            target_country=country,
            doc_type=doc_type,
            n=4
        )
        final = []
        local_scores = {"grammar": 0.0, "fluency": 0.0,
                        "style": 0.0, "idioms": 0.0}
        local_counts = {"grammar": 0, "fluency": 0, "style": 0, "idioms": 0}
        for c in candidates:
            blocked_reasons = []
            matches = find_forbidden_matches(
                c["after"], glossary, compliance_patterns)
            if matches["glossary"]:
                blocked_reasons.append("glossary")
            if matches["compliance"]:
                blocked_reasons.append("compliance")
            orig_nums = set(re.findall(r"\b\d+\b", text))
            new_nums = set(re.findall(r"\b\d+\b", c["after"]))
            if orig_nums != new_nums:
                blocked_reasons.append("numbers")
            risk, risk_conf = await meaning_shift_risk(text, c["after"], doc_type)
            blocked = len(blocked_reasons) > 0
            suggestion_obj = {
                "suggestion_id": c.get("suggestion_id", _sha1(c.get("after", "")[:64])),
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
            t = suggestion_obj["type"]
            if t in scores:
                scores[t] += suggestion_obj["confidence"] * 100
                score_counts[t] += 1

        # Calculate local scores for this segment
        segment_scores = {
            k: int(local_scores[k] / max(1, local_counts[k])
                   ) if local_counts[k] > 0 else 0
            for k in local_scores
        }
        payload = {

            "suggestions": final,
            "scores": segment_scores,
            "generated_at": datetime.utcnow().isoformat()
        }
        set_suggestions_in_cache(
            tenant_id, language_pair, domain, text, payload)
        # , "scores": segment_scores
        return {"path": path, "original": text, "suggestions": final}

    tasks = [(seg["path"], seg["text"])
             for seg in segments if seg.get("text", "").strip()]
    if not tasks:
        logger.error("No valid segments provided")
        print("No valid segments provided")
        return {"suggestions": [], "scores": {"grammar": 0, "fluency": 0, "style": 0, "idioms": 0}}

    results = await asyncio.gather(*[asyncio.create_task(_process_one(path, text)) for path, text in tasks])
    aggregated_suggestions.extend(results)

    # Calculate overall scores across all segments
    overall_scores = {
        k: int(scores[k] / max(1, score_counts[k])
               ) if score_counts[k] > 0 else 0
        for k in scores
    }
    logger.info(
        f"[{tenant_id}] Processed {len(aggregated_suggestions)} segments total")
    logger.info(f"Scores summary: {overall_scores}")
    print(f"[{tenant_id}] Processed {len(aggregated_suggestions)} segments total")
    print(f"Scores summary: {overall_scores}")

    return {"suggestions": aggregated_suggestions, "scores": overall_scores}
