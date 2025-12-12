# schemas/suggestions.py
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID


class SuggestionOptionOut(BaseModel):
    option_id: UUID = Field(..., alias="id")
    suggestion_type: str
    before: str
    after: str
    confidence: Optional[float] = None
    blocked: Optional[bool] = False
    risk: Optional[str] = None
    metadata: Optional[dict] = None

    class Config:
        orm_mode = True
        allow_population_by_field_name = True


class SuggestionSegmentOut(BaseModel):
    segment_id: int = Field(..., alias="id")
    path: str
    original: str
    options: List[SuggestionOptionOut]

    class Config:
        orm_mode = True
        allow_population_by_field_name = True


class GenerateResponse(BaseModel):
    status: str
    data: dict
