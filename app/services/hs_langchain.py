from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from qdrant_client.http import models
from app.config import settings
from dataclasses import dataclass
from langchain_core.output_parsers import StrOutputParser
from typing import List
import json


def fewshotTranslation(llm, qdrant, query):
    scroll_results, _ = qdrant.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="shopDomain",
                    match=models.MatchValue(value=query.shopDomain)
                ),
                models.FieldCondition(
                    key="targetLanguage",
                    match=models.MatchValue(value=query.targetLanguage)
                ),
                models.FieldCondition(
                    key="region",
                    match=models.MatchValue(value=query.region)
                )
            ]
        ),
        with_payload=True,
        limit=1
    )

    original_text = None
    translated_text = None
    for point in scroll_results:
        original_text = json.dumps(point.payload.get("original_text"), indent=4, ensure_ascii=False) if point.payload is not None else None
        translated_text = json.dumps(point.payload.get("translated_text"), indent=4, ensure_ascii=False) if point.payload is not None else None

    original_text = original_text.replace("{", "{{").replace("}", "}}") if original_text is not None else ""
    translated_text = translated_text.replace("{", "{{").replace("}", "}}") if translated_text is not None else ""

    example = {
        "original": original_text[:len(original_text)//4],
        "translated": translated_text[:len(translated_text)//4],
    }

    print("Example is created!")

    example_template = """
    Original: {original}
    Translated: {translated}
    """

    print("Example template is created!")

    example_prompt = PromptTemplate(
        input_variables=["original", "translated"],
        template=example_template,
    )

    print("Example prompt is created!")

    fewshot_prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=[example],
        prefix="""
        Translate the following strings into {targetLanguage}, adapted for the {region} region.
        Maintain the brand tone as {brandTone}.
        ⚠ If a string contains HTML tags (<p>, <div>, <br>, etc.), keep the tags exactly as they are,
        and only translate the inner text.
        Ensure cultural and linguistic nuances are appropriate for {region}.
        Return ONLY translations line by line, in the same order:
        """,
        suffix="Source: {input}\nTranslated:",
        input_variables=["input", "targetLanguage", "brandTone", "region"],
    )

    print(f"Fewshot prompt template is created!\n{fewshot_prompt}")
    print("Fewshot prompt is created!")

    print("Expected variables:", fewshot_prompt.input_variables)


    chain = fewshot_prompt | llm |StrOutputParser()

    print("Chain is created!")

        # 🔹 Handle list input
    input_text = query.input
    if isinstance(input_text, list):
        # Join into newline-separated block for the model
        input_text = "\n".join(input_text)

    response = chain.invoke({
        "input": query.input,
        "targetLanguage": query.targetLanguage,
        "brandTone": query.brandTone,
        "region": query.region
    })

    print("Response is created!")
    
    try:
        translations = json.loads(response)
    except Exception:
        translations = response.split("\n")

    return translations

def fewshotCategorization(llm, text):
    example = [
        {"text": "Blue cotton T-Shirt", "label": "ordinary"},
        {"text": "Add to Cart", "label": "ordinary"},
        {"text": "Best quality leather shoes", "label": "ordinary"},
        {"text": "Customer Privacy Policy", "label": "business"},
        {"text": "Refunds will be processed within 7 days", "label": "business"},
        {"text": "Continue Shopping", "label": "ordinary"},
        {"text": "Wholesale Pricing Available", "label": "business"},
        {"text": "Track Your Order", "label": "ordinary"},
        {"text": "Terms & Conditions apply", "label": "business"},
        {"text": "New Summer Collection", "label": "ordinary"},
        {"text": "Business Invoice Download", "label": "business"},
        {"text": "Proceed to Checkout", "label": "ordinary"},
        {"text": "This product is covered by a 1-year warranty", "label": "business"},
        {"text": "Flash Sale: Up to 50% off", "label": "ordinary"},
        {"text": "Compliance with EU regulations", "label": "business"},
        {"text": "Sign In", "label": "ordinary"},
        {"text": "Corporate Account Registration", "label": "business"},
        {"text": "View Cart", "label": "ordinary"},
        {"text": "Export License Required", "label": "business"},
        {"text": "Shop Now", "label": "ordinary"},
        ]

    example_template = """
    Text: {text}
    Category: {label}
    """

    example_prompt = PromptTemplate(
        input_variables=["text", "label"],
        template=example_template,
    )

    fewshot_prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=example,
        prefix="""You are a text classifier. 
        Classify each input string into one of two categories:
        
        - "business" → official, professional, legal, formal, or business-related.
        - "ordinary" → casual, personal, everyday language.
        
        Here are some examples:""",
        suffix="\nText: {text}\nCategory:",
        input_variables=["text"]
    )

    chain = fewshot_prompt | llm

    response = chain.invoke({
        "text": text,
    })

    return response.content.strip().lower()


def stringCategorize(llm, _strings_to_translate):
    text_to_translate = {"business": [], "ordinary": []}
    for s in _strings_to_translate:
        response = fewshotCategorization(llm, s)
        if response == "business":
            text_to_translate["business"].append(s)
        else:
            text_to_translate["ordinary"].append(s)
    return text_to_translate


@dataclass
class TranslationQuery:
    shopDomain: str
    input: List[str]
    targetLanguage: str
    brandTone: str
    region: str
