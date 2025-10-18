from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from ..services.suggestion_engine import produce_suggestions, load_style_pack, update_translation_json
from ..database import SessionLocal
from app.routes.ingest import get_db
from ..models.suggestion import Suggestion, SuggestionAudit
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


class Segment(BaseModel):
    path: str
    text: str


class GenerateRequest(BaseModel):
    tenant_id: str
    translation_id: int
    doc_type: str
    domain: str
    country: str
    language_pair: str
    preserve_legal_meaning: Optional[bool] = True
    segments: List[Segment]
    glossary: Optional[List[str]] = None
    compliance_patterns: Optional[List[str]] = None


class ApplyRequest(BaseModel):
    tenant_id: str
    translation_id: int
    path: str
    suggestion_id: str
    user_id: Optional[str] = None
    action: str  # "accept"|"reject"
    before: Optional[str] = None
    after: Optional[str] = None
    metadata: Optional[dict] = None


@router.post("/generate")
async def suggestions_generate(req: GenerateRequest, db: Session = Depends(get_db)):
    # validate segments
    if not req.segments or len(req.segments) == 0:
        raise HTTPException(status_code=400, detail="segments required")

    logger.info(f"[generate] Tenant={req.tenant_id} Domain={req.domain}")
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

    return {"status": "ok", "data": out}


@router.post("/apply")
async def suggestions_apply(req: ApplyRequest, db: Session = Depends(get_db)):
    """
    Accept or reject a suggestion. This persists an audit entry and (for accept) writes the change into Suggestion table.
    Frontend should itself update displayed text (we store audit + updated suggestion record).
    """
    logger.info(
        f"[apply] Applying suggestion {req.suggestion_id} | Action={req.action}")

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
    logger.info("[apply] Action recorded successfully.")
    return {"status": "ok", "message": "Changes are applied and action recorded"}


@router.get("/style-pack")
def get_style_pack(tenant_id: str, domain: str, country: str, language_pair: str):
    # simply return the loaded style pack (from cache or defaults)
    pack = load_style_pack(tenant_id, language_pair, domain, country)
    return {"status": "ok", "style_pack": pack}


@router.post("/invalidate")
def invalidate_tenant_cache(tenant_id: str):
    invalidate_suggestions_for_tenant(tenant_id)
    return {"status": "ok", "message": "cache invalidated"}
