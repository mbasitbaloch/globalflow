from pydantic import BaseModel


class ShopifyTranslateRequest(BaseModel):
    shopDomain: str
    accessToken: str
    targetLanguage: str
    brandTone: str


# class TranslationResponse(BaseModel):
#     source: str
#     target: str
#     target_lang: str
