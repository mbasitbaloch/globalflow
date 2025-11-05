import re
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional


class UpdateRequest(BaseModel):
    translation_id: int = Field(..., description="Translation record ID")
    shopDomain: str = Field(
        ..., description="Shop domain name (e.g. globalflow-ai-esp.myshopify.com)")
    targetLanguage: str = Field(
        ..., description="Language code of the target translation (e.g. 'fr')")
    targetcountry: str = Field(...,
                               description="Target country (e.g. 'france')")
    path: str = Field(..., description="JSON path to the string being updated")
    newValue: str = Field(..., description="The updated translated string")
    originalValue: str = Field(
        "", description="The original translation before update")
    expertEdit: Optional[bool] = Field(
        False, description="Whether the update was made by an expert")
    customerEdit: Optional[bool] = Field(
        False, description="Whether the update was made by a customer")
    transAccept: bool = Field(
        False, description="Whether the translation was accepted")
    transEdit: Optional[bool] = Field(
        False, description="Whether the translation was edited")

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid"
    )

    @field_validator("shopDomain")
    def validate_shop_domain(cls, v):
        v = v.strip()
        # Check must contain at least one dot, not start or end with a dot, and must look like a domain
        domain_pattern = re.compile(
            r"^(?!\-)(?:[a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,}$")
        if not domain_pattern.match(v):
            raise ValueError(
                "Invalid shopDomain format. Must be a valid domain like 'example.myshopify.com'.")
        return v

    @field_validator("targetLanguage", "targetcountry", "path", "newValue")
    def not_empty(cls, v, info):
        if not v or not str(v).strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return v

    @field_validator("newValue")
    def check_min_length(cls, v):
        if len(v.strip()) < 2:
            raise ValueError(
                "newValue is too short — must be a meaningful string.")
        return v
