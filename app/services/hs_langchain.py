from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from app.config import settings
from dataclasses import dataclass
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from typing import List
import json
import re


async def promptClassification(llm, strings_batch):
    prompt_template = PromptTemplate.from_template("""
    You are a strict text classifier.

    Categories:
    - "business" = official, legal, contractual, financial, invoices, policies, compliance, formal system messages.
    - "ordinary" = product marketing, casual phrases, blogs, general UI text, everyday communication.
    Never invent new categories; only use "business" or "ordinary".

    Rules:
    - Classify each string into exactly ONE category.
    - The number of output labels MUST equal the number of input strings ({num_strings}).
    - Keep the order of outputs identical to the order of inputs.

    Examples:
    Input: ["Invoice #4533", "Big summer sale!", "Refunds will be processed within 7 days", "Sign In"]
    Output: ["business", "ordinary", "business", "ordinary"]

    Input: ["Terms and Conditions apply", "Export License Required", "Check out our new arrivals", "Best quality leather shoes"]
    Output: ["business", "business", "ordinary", "ordinary"]

    Now classify these {num_strings} strings:
    {strings_batch}

    IMPORTANT:
    Respond with ONLY a valid JSON array of {num_strings} strings. No extra text.
    """
    )

    chain = prompt_template | llm | JsonOutputParser()
    response = await chain.ainvoke({
        "strings_batch": json.dumps(strings_batch, ensure_ascii=False),
        "num_strings": len(strings_batch)
    })

    return [x.strip().lower() for x in response]



async def fewshotTranslation(examples, llm, query, SafeJsonParser):
    example_template = """
    Original: {original}
    Translated: {translated}
    """

    # print("Example template is created!")

    example_prompt = PromptTemplate(
        input_variables=["original", "translated"],
        template=example_template,
    )

    # print("Example prompt is created!")

    fewshot_prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        prefix="""
        You are a professional translator.

        Task:
        Translate the following {num_strings} strings into {targetLanguage}.
        - Maintain the brand tone as '{brandTone}'.
        - Adapt translations to the industrial domain '{industry}'.
          Use terminology, phrasing, and style that are natural and widely used in this domain.
        - If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags unchanged, only translate the inner text.
        - Preserve placeholders (e.g., {{name}}, %s, {{0}}) exactly as they are. Translate surrounding text but do NOT translate or modify the text inside placeholders.
        - Do NOT merge, omit, or add strings.
        - Translate long texts fully (no summarization).
        - Language code rule: if a string is a language code (e.g., "en"), replace it with the correct code for {targetLanguage}.
        Example: "en" → "fr" when {targetLanguage} is French.

        Output requirements:
        - Return ONLY valid JSON.
        - JSON must be an array of exactly {num_strings} strings.
        - Order must match the input order.
        - No comments, no explanations, no extra text.

        Input strings:
        {input}

        Output format (strict):
        [
        "translation of string 1",
        "translation of string 2",
        ...
        ]
        """,
        suffix="Source:\n{input}\nTranslated:",
        input_variables=["input", "targetLanguage", "brandTone", "industry", "num_strings"],
    )

    # print(f"Fewshot prompt template is created!\n{fewshot_prompt}")
    # print("Fewshot prompt is created!")

    # print("Expected variables:", fewshot_prompt.input_variables)


    chain = fewshot_prompt | llm | SafeJsonParser()

    # print("Chain is created!")
    input_text = query.input

    response = await chain.ainvoke({
        "input": json.dumps(input_text, ensure_ascii=False),
        "targetLanguage": query.targetLanguage,
        "brandTone": query.brandTone,
        "industry": query.industry,
        "num_strings": len(input_text),
    })

    # print("Response is created!")

    return response


@dataclass
class TranslationQuery:
    input: List[str]
    user_id: str
    shopDomain: str
    targetLanguage: str
    brandTone: str
    industry: str
    num_strings: int

class SafeJsonParser(BaseOutputParser):
    def parse(self, text: str):
        text = text.strip()
        text = re.sub(r"^```(?:json|json5|javascript)?\s*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()
        return json.loads(text)
