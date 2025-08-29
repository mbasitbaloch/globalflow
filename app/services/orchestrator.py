from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from .rag import retrieve_examples


def translate_text(text, tenant, target_lang):
    # RAG: get few-shot examples
    examples = retrieve_examples(text, tenant.id)

    example_prompt = PromptTemplate(
        input_variables=["source", "target"],
        template="Source: {source}\nTarget: {target}"
    )

    few_shot = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=f"You are a professional translator for {tenant.name}. Tone: {tenant.glossary}",
        suffix="Now translate:\nSource: {input}\nTarget:",
        input_variables=["input"],
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    final_prompt = few_shot.format(input=text)
    result = llm.predict(final_prompt)
    return result
