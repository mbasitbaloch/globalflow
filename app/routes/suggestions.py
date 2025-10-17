from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from ..services.suggestion_engine import produce_suggestions, load_style_pack
from ..database import SessionLocal
from app.routes.ingest import get_db
from ..models.suggestion import Suggestion, SuggestionAudit
from sqlalchemy.orm import Session
from ..utils.cache_manager import invalidate_suggestions_for_tenant
import uuid
import json
from ..config import settings

router = APIRouter(prefix="/suggestions", tags=["suggestions"])


class Segment(BaseModel):
    id: str
    text: str


class GenerateRequest(BaseModel):
    tenant_id: str
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
    segment_id: str
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

    # store suggestions in DB (for audit & indexing)
    for seg in out["suggestions"]:
        s_model = Suggestion(
            tenant_id=req.tenant_id,
            doc_type=req.doc_type,
            domain=req.domain,
            language_pair=req.language_pair,
            segment_id=seg["segment_id"],
            original_text=seg["original"],
            suggestions=seg["suggestions"]
        )
        db.add(s_model)
    db.commit()

    return {"status": "ok", "data": out}


@router.post("/apply")
def suggestions_apply(req: ApplyRequest, db: Session = Depends(get_db)):
    """
    Accept or reject a suggestion. This persists an audit entry and (for accept) writes the change into Suggestion table.
    Frontend should itself update displayed text (we store audit + updated suggestion record).
    """
    if req.action not in ("accept", "reject"):
        raise HTTPException(status_code=400, detail="invalid action")
    # audit write
    audit = SuggestionAudit(
        suggestion_id=None,
        tenant_id=req.tenant_id,
        user_id=req.user_id,
        segment_id=req.segment_id,
        suggestion_hash=str(uuid.uuid4()),
        action=req.action,
        suggestion_type=None,
        before=req.before,
        after=req.after,
        metadata=req.metadata
    )
    db.add(audit)

    # if accept -> update persisted suggestion record if exists
    if req.action == "accept":
        # Find most recent Suggestion for tenant+segment
        existing = db.query(Suggestion).filter_by(tenant_id=req.tenant_id,
                                                  segment_id=req.segment_id).order_by(Suggestion.created_at.desc()).first()
        if existing:
            # update suggestions array: mark suggestion_id accepted where possible
            try:
                suggestions = existing.suggestions or []
                # we will append an audit field; front-end still receives new text directly
                db_sugg = {
                    "accepted_suggestion_id": req.suggestion_id,
                    "accepted_by": req.user_id,
                    "accepted_at": datetime.utcnow().isoformat(),
                    "before": req.before,
                    "after": req.after
                }
                # attach to metadata area inside suggestions record
                existing.suggestions = {
                    "applied": db_sugg, "previous": suggestions}
                db.add(existing)
            except Exception as e:
                pass

    db.commit()
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
