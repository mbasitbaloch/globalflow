# from app.database import get_db
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import re
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from ..services.suggestion_engine import produce_suggestions, load_style_pack, update_translation_json
from ..database import SessionLocal
from app.routes.ingest import get_db
from ..models.suggestionRequest import GenerateRequest, ApplyRequest
from ..models.models import Translation
from sqlalchemy.orm import Session
from ..utils.cache_manager import invalidate_suggestions_for_tenant
from ..config import settings
import uuid
import json
import logging
from ..validator.countryValidator import validate_language
from ..models.suggestion import SuggestionAudit, SuggestionSegment, SuggestionOption
from uuid import UUID as UUIDType

logger = logging.getLogger("suggestions_api")


router = APIRouter(prefix="/suggestions", tags=["suggestions"])

print("Entered in suggestions.py")


@router.post("/generate")
async def suggestions_generate(req: GenerateRequest, db: Session = Depends(get_db)):
    # (validation code you already have omitted here for brevity)
    saved_segments = []

    try:
        out = await produce_suggestions(
            tenant_id=req.tenant_id,
            doc_type=req.doc_type,
            domain=req.domain,
            country=req.target_country,
            target_language=req.target_language,
            source_language=req.source_language,
            preserve_legal_meaning=req.preserve_legal_meaning,
            segments=[s.dict() for s in req.segments],
            glossary=req.glossary,
            compliance_patterns=req.compliance_patterns
        )

        # each item in out["suggestions"] is one segment
        for seg in out["suggestions"]:
            # create segment row
            seg_row = SuggestionSegment(
                tenant_id=req.tenant_id,
                translation_id=req.translation_id,
                doc_type=req.doc_type,
                domain=req.domain,
                target_language=req.target_language,
                source_language=req.source_language,
                path=seg["path"],
                original_text=seg["original"],
            )
            db.add(seg_row)
            db.flush()  # now seg_row.id is available

            option_rows_response = []
            # create option rows for each suggestion option
            for opt in seg.get("suggestions", []):
                opt_row = SuggestionOption(
                    id=uuid.UUID(opt.get("suggestion_id")) if opt.get(
                        "suggestion_id") else uuid.uuid4(),
                    segment_id=seg_row.id,
                    suggestion_type=opt.get(
                        "type") or opt.get("suggestion_type"),
                    before=opt.get("before"),
                    after=opt.get("after"),
                    confidence=opt.get("confidence"),
                    blocked=opt.get("blocked", False),
                    risk=opt.get("risk"),
                    metadata=opt.get("meta")
                )
                db.add(opt_row)
                db.flush()  # get opt_row.id
                option_rows_response.append({
                    "id": opt_row.id,
                    "suggestion_type": opt_row.suggestion_type,
                    "before": opt_row.before,
                    "after": opt_row.after,
                    "confidence": opt_row.confidence,
                    "blocked": opt_row.blocked,
                    "risk": opt_row.risk,
                    "metadata": opt_row.meta
                })

            saved_segments.append({
                "id": seg_row.id,
                "path": seg_row.path,
                "original": seg_row.original_text,
                "options": option_rows_response
            })

        db.commit()

        return {
            "status": "Success",
            "data": {
                "suggestions": saved_segments
            }
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Failed to generate suggestions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply")
async def suggestions_apply(req: ApplyRequest, db: Session = Depends(get_db)):
    # Basic validations (action and before/after)
    if req.action not in ("accept", "reject"):
        raise HTTPException(status_code=400, detail="invalid action")

    if req.before == req.after:
        raise HTTPException(
            status_code=400, detail="Before and After cannot be the same")

    # Validate segment existence
    segment = db.query(SuggestionSegment).filter_by(
        id=req.suggestion_id, tenant_id=req.tenant_id).first()
    if not segment:
        raise HTTPException(
            status_code=404, detail="Suggestion segment not found")

    # Validate option exists and belongs to segment (option_id may be UUID string or actual UUID)
    try:
        option_uuid = UUIDType(req.metadata.get(
            "option_id")) if req.metadata and req.metadata.get("option_id") else None
    except Exception:
        option_uuid = None

    option = None
    if option_uuid:
        option = db.query(SuggestionOption).filter_by(
            id=option_uuid, segment_id=segment.id).first()
    else:
        # If frontend provides suggestion_id inside top-level field 'suggestion_id' (legacy),
        # try to find a single default option (not ideal). Better: require option_id in metadata.
        pass

    if not option and req.action == "accept":
        raise HTTPException(
            status_code=404, detail="Selected suggestion option not found for this segment")

    # Create audit
    audit = SuggestionAudit(
        tenant_id=req.tenant_id,
        translation_id=req.translation_id,
        segment_id=segment.id,
        option_id=option.id if option else None,
        user_id=req.user_id,
        path=req.path,
        suggestion_hash=str(uuid.uuid4()),
        action=req.action,
        before=req.before,
        after=req.after,
        metadata=req.metadata
    )
    db.add(audit)

    # If accept -> update translation JSON (use option.after or req.after)
    if req.action == "accept":
        try:
            translation = db.query(Translation).filter_by(
                id=req.translation_id).first()
            if not translation:
                raise HTTPException(
                    status_code=404, detail="Translation not found")

            # prefer option.after if an option was validated, otherwise use req.after
            value_to_apply = option.after if option else req.after

            updated_json = await update_translation_json(translation, req.path, value_to_apply)
            translation.translated_text_json = updated_json
            translation.translated_text_raw = json.dumps(
                updated_json, ensure_ascii=False)
            translation.updated_at = datetime.utcnow()

            db.add(translation)
            db.commit()

        except HTTPException:
            db.rollback()
            raise
        except Exception as e:
            db.rollback()
            logger.exception("Failed to apply suggestion: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Failed to update translation: {e}")

    # commit audit (and any other changes) if not already committed
    db.commit()

    if req.action == "reject":
        return {"status": "Rejected", "message": "No changes applied, action recorded in audit log"}
    else:
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
