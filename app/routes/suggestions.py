from datetime import datetime
import re
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from ..services.suggestion_engine import produce_suggestions, load_style_pack, update_translation_json
from ..database import SessionLocal
from app.routes.ingest import get_db
from ..models.suggestion import Suggestion, SuggestionAudit
from ..models.suggestionRequest import GenerateRequest, ApplyRequest
from ..models.models import Translation
from sqlalchemy.orm import Session
from ..utils.cache_manager import invalidate_suggestions_for_tenant
from ..config import settings
import uuid
import json
from ..config import settings
import logging


logger = logging.getLogger("suggestions_api")


router = APIRouter(prefix="/suggestions", tags=["suggestions"])


@router.post("/generate")
async def suggestions_generate(req: GenerateRequest, db: Session = Depends(get_db)):
    """
    Expect body:
    {
    "tenant_id": "68cabc6b4eddddad0891642b",
    "translation_id":40,
    "doc_type": "Legal",
    "domain": "globalflow-ai-esp.myshopify.com",
    "country": "CA-CA", 
    "language_pair": "en-fr",
    "preserve_legal_meaning": true,
    "segments": [
        { "path": "fullData.storeData.products.0.title", "text": "Ventilateur portable sans pales à suspendre au cou – Rechargeable, silencieux et mains libres pour l'été" } //france french
    ],
    "glossary": ["portable", "Rechargeable"],
    "compliance_patterns": ["sans pales à suspendre"]
    }
    """
   # 1 Validate all required fields exist and are non-empty
    required_fields = [
        "tenant_id", "translation_id", "doc_type",
        "domain", "country", "language_pair",
        "preserve_legal_meaning", "segments"
    ]

    missing = [f for f in required_fields if getattr(req, f, None) is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required field: {', '.join(missing)}"
        )

    empty_fields = [f for f in required_fields if not str(
        getattr(req, f)).strip()]
    if empty_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Field cannot be empty: {', '.join(empty_fields)}"
        )

    # 2 Validate domain format (must end with a valid TLD)
    shop_domain = req.domain.strip().lower()
    domain_pattern = re.compile(
        # ensures .com, .ca, .org, etc.
        r"^(?!-)([A-Za-z0-9-]+\.)+[A-Za-z]{2,}$"
    )
    if not domain_pattern.match(shop_domain):
        raise HTTPException(
            status_code=400,
            detail="Invalid domain format. Please provide a valid domain like 'example.myshopify.com' or 'example.ca'."
        )

    # 3 Validate language pair (e.g., en-fr, fr-en)
    lang_pair = req.language_pair.strip().lower()
    if not re.match(r"^[a-z]{2}-[a-z]{2}$", lang_pair):
        raise HTTPException(
            status_code=400,
            detail="Invalid language_pair format. Use ISO codes like 'en-fr', 'fr-en', 'en-es'."
        )

    # 4 Validate segments
    if not req.segments or len(req.segments) == 0:
        raise HTTPException(
            status_code=400,
            detail="Segments list cannot be empty."
        )

    for idx, seg in enumerate(req.segments):
        if not seg.path.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Segment #{idx + 1}: 'path' cannot be empty."
            )
        if not seg.text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Segment #{idx + 1}: 'text' cannot be empty."
            )
    # Logging
    logger.info(
        f"[generate] Tenant={req.tenant_id} Domain={req.domain} language_pair={req.language_pair} country={req.country}"
    )
    logger.info("[generate] Loading style pack and checking cache...")

    # call engine
    out = await produce_suggestions(
        tenant_id=req.tenant_id,
        doc_type=req.doc_type,
        domain=req.domain,
        country=req.country,
        language_pair=req.language_pair,
        preserve_legal_meaning=req.preserve_legal_meaning,
        segments=[s.dict() for s in req.segments],
        glossary=req.glossary,
        compliance_patterns=req.compliance_patterns
    )
    logger.info("[generate] Suggestions generated successfully.")
    logger.info("[generate] Storing results in PostgreSQL...")
    # store suggestions in DB (for audit & indexing)
    for seg in out["suggestions"]:
        s_model = Suggestion(
            tenant_id=req.tenant_id,
            translation_id=req.translation_id,
            doc_type=req.doc_type,
            domain=req.domain,
            language_pair=req.language_pair,
            path=seg["path"],
            original_text=seg["original"],
            suggestions=seg["suggestions"]
        )
        db.add(s_model)
    db.commit()
    logger.info("[generate] Suggestions stored successfully in PostgreSQL.")

    return {"status": "Success", "data": out}


@router.post("/apply")
async def suggestions_apply(req: ApplyRequest, db: Session = Depends(get_db)):
    """
    Accept or reject a suggestion. This persists an audit entry and (for accept) writes the change into Suggestion table.
    Frontend should itself update displayed text (we store audit + updated suggestion record).
    expected body

    {
    "tenant_id": "68cabc6b4eddddad0891642b",
    "translation_id": 40,
    "path": "fullData.storeData.products.0.title",
    "suggestion_id": "54",
    "user_id": "68cabc6b4eddddad0891642b",
    "action": "reject",
    "before": "Ventilateur portable sans pales à suspendre au cou – Rechargeable, silencieux et mains libres pour l'été", 
    "after": "Ventilateur portable sans pales à porter autour du cou – Rechargeable, silencieux et mains libres pour l'été",
    "metadata": {
        "doc_type": "Legal",
        "language_pair": "en-fr"
        }
    }
    """

    logger.info(
        f"[apply] Applying suggestion {req.suggestion_id} | Action={req.action}")
    print(f"action is {req.action}")

    if req.action not in ("accept", "reject"):
        raise HTTPException(status_code=400, detail="invalid action")
    # audit write
    audit = SuggestionAudit(
        suggestion_id=req.suggestion_id,
        translation_id=req.translation_id,
        tenant_id=req.tenant_id,
        user_id=req.user_id,
        path=req.path,
        suggestion_hash=str(uuid.uuid4()),
        action=req.action,
        suggestion_type=None,
        before=req.before,
        after=req.after,
        metadata=req.metadata
    )
    db.add(audit)
    logger.info("[apply] Audit record added.")
    print("[apply] Audit record added.")

    if req.action == "accept":
        try:
            logger.info("[apply] Fetching translation record...")
            translation = db.query(Translation).filter_by(
                id=req.translation_id).first()
            if not translation:
                raise HTTPException(
                    status_code=404, detail="Translation not found")

            logger.info("[apply] Updating JSON path in translation data...")
            updated_json = await update_translation_json(translation, req.path, req.after)
            translation.translated_text_json = updated_json
            translation.translated_text_raw = json.dumps(
                updated_json, ensure_ascii=False)
            translation.updated_at = datetime.utcnow()

            db.add(translation)
            db.commit()
            logger.info(
                "[apply] Translation updated successfully in PostgreSQL.")

        except Exception as e:
            logger.exception(f"[apply] Failed to update translation: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to update translation: {e}")

    db.commit()
    if req.action == "reject":
        logger.info(
            "[apply] Suggestion rejected, no translation update needed.")
        print("[apply] Suggestion rejected, no translation update needed.")
        print(f"[apply] Action recorded successfully for action: {req.action}")
        return {"status": "Rejected", "message": "No changes applied, action recorded in audit log"}
    else:
        logger.info(
            f"[apply] Action recorded successfully for action: {req.action}")
        print(f"[apply] Action recorded successfully for action: {req.action}")
        return {"status": "Success", "message": "Changes are applied and action recorded in audit log"}


@router.get("/style-pack")
def get_style_pack(tenant_id: str, domain: str, country: str, language_pair: str):
    # simply return the loaded style pack (from cache or defaults)
    pack = load_style_pack(tenant_id, language_pair, domain, country)
    return {"status": "ok", "style_pack": pack}


@router.post("/invalidate")
def invalidate_tenant_cache(tenant_id: str):
    invalidate_suggestions_for_tenant(tenant_id)
    return {"status": "ok", "message": "cache invalidated"}
