from openai import OpenAI
from ..config import settings
import json

client = OpenAI(api_key=settings.OPENAI_API_KEY_1)

source_language = "English"
target_language = "French"

system_prompt = f"""
You are a professional translator. Your task is to accurately translate any given text from the {source_language} into {target_language}. 
- Preserve the meaning, tone, and style of the original text.
- Do not add, remove, or alter information.
- Only provide the translation without extra explanation.
"""

source_text = ""
target_text = ""

with open("train.jsonl", "w", encoding="utf-8") as f:
    for _ in range(100):
        example = {"messages" : [{"role": "system","content": system_prompt},{"role": "user","content": source_text},{"role": "assistant","content": target_text}]}
        json.dump(example, f)
        f.write("\n")

training_file = client.files.create(
    file=open("train.jsonl", "rb"),
    purpose="fine-tune"
)
print("Uploaded file ID:", training_file.id)

fine_tune = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo"
)
print("Fine-tune job created:", fine_tune.id)

# STEP 3: Wait & check status
status = client.fine_tuning.jobs.retrieve(fine_tune.id)
print("Status:", status.status)