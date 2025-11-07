from pydantic import BaseModel, field_validator, ConfigDict, Field
from typing import List, Optional, Dict
import re


class Segment(BaseModel):
    path: str = Field(..., description="JSON path to the text segment")
    text: str = Field(..., description="Text content to analyze")


class GenerateRequest(BaseModel):
    tenant_id: str
    translation_id: int
    doc_type: str
    domain: str
    country: str
    language_pair: str
    preserve_legal_meaning: bool
    segments: List[Segment]
    glossary: Optional[List[str]] = []
    compliance_patterns: Optional[List[str]] = []

    @field_validator("domain")
    def validate_domain(cls, v):
        pattern = r"^(?!-)([A-Za-z0-9-]+\.)+[A-Za-z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid domain format. Example: example.myshopify.com")
        return v

    @field_validator("language_pair")
    def validate_language_pair(cls, v):
        if not re.match(r"^[a-z]{2}-[a-z]{2}$", v.lower()):
            raise ValueError(
                "Invalid language_pair format. Use short codes like 'en-fr'.")
        return v

    @field_validator("segments")
    def validate_segments(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one segment is required.")
        for i, seg in enumerate(v):
            if not seg.path.strip():
                raise ValueError(f"Segment #{i+1} has empty path.")
            if not seg.text.strip():
                raise ValueError(f"Segment #{i+1} has empty text.")
        return v


class ApplyRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant unique identifier")
    translation_id: int = Field(..., description="Translation record ID")
    path: str = Field(..., description="JSON path where the change applies")
    suggestion_id: str = Field(..., description="Suggestion record ID")
    user_id: str = Field(..., description="User performing the action")
    action: str = Field(..., description="Action must be 'accept' or 'reject'")
    before: Optional[str] = Field(
        None, description="Original string before change")
    after: Optional[str] = Field(
        None, description="Updated string after change")
    metadata: Optional[Dict] = Field(None, description="Additional info")

    model_config = ConfigDict(
        str_strip_whitespace=True,  # replaces anystr_strip_whitespace
        extra="forbid"  # prevent extra/unknown fields
    )

    @field_validator("*", mode="before")
    def no_empty_fields(cls, v, info):
        """Reject empty or null values for required fields."""
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(f"{info.field_name} cannot be empty or null")
        return v

    @field_validator("action")
    def validate_action(cls, v):
        """Ensure action is either 'accept' or 'reject'."""
        if v not in ("accept", "reject"):
            raise ValueError("Action must be 'accept' or 'reject'")
        return v
