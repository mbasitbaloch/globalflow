from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from qdrant_client.http import models
from app.config import settings
import json


def fewshotPrompt(llm, qdrant, query):
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
   
    # Final JSON response structure
    # result_json = {
    #     "points": original_text,
    #     "next_page": translated_text
    # }

    # # Ensure directory exists
    # os.makedirs("result", exist_ok=True)

    # # Save result to file
    # with open("result/result.json", "w", encoding="utf-8") as f:
    #     json.dump(result_json, f, indent=4, ensure_ascii=False)

    # print("Result saved in result/result.json")

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

    few_shot_prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=[example],
        prefix="""
        Translate the following strings into {targetLanguage}.
        Maintain the brand tone as {brandTone}.
        ⚠ If a string contains HTML tags (<p>, <div>, <br>, etc.), keep the tags exactly as they are, 
        and only translate the inner text.
        Return ONLY translations line by line, same order:
        """,
        suffix="Source: {input}\nTranslated:",
        input_variables=["input", "targetLanguage", "brandTone"],
    )

    print("Fewshot prompt is created!")

    print("Expected variables:", few_shot_prompt.input_variables)


    chain = few_shot_prompt | llm

    print("Chain is created!")

    response = chain.invoke({
        "input": query.input,
        "targetLanguage": query.targetLanguage,
        "brandTone": query.brandTone,
    })

    print("Response is created!")



    return response


from dataclasses import dataclass

@dataclass
class TranslationQuery:
    shopDomain: str
    input: str
    targetLanguage: str
    brandTone: str

# Example usage
# query = TranslationQuery(
#     shopDomain="globalflow-ai-esp.myshopify.com",
#     input="Hello, how are you?",
#     targetLanguage="French",
#     brandTone="Friendly"
# )

# result = fewshotPrompt(query)
# print("Translation:", result)