from fastapi import APIRouter, UploadFile, Form, Depends
from sqlalchemy.orm import Session
from ..database import SessionLocal
from .. import models
from ..workers.tasks import generate_embedding
from ..services.rag import embed_and_store
from ..services import translator
from fastapi import BackgroundTasks

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# @router.post("/ingest/html")
# async def ingest_html(file: UploadFile, target_lang: str = Form(...), db: Session = Depends(get_db),  background_tasks: BackgroundTasks = None):
#     content = await file.read()
#     text = content.decode("utf-8")
#     print("File received:", file.filename)
#     print("Target lang:", target_lang)

#     translated_text = translator.translate_text(text, target_lang)

#     record = models.Translation(
#         source_text=text, target_text=translated_text, target_lang=target_lang, content_type="html"
#     )
#     db.add(record)
#     db.commit()
#     db.refresh(record)

#     # embed_and_store(text, record.id)
#     print(record.target_text)

#     background_tasks.add_task(embed_and_store, text, record.id)

#     # generate_embedding.delay(text, record.id)
#     return {"id": record.id, "status": "ingested", "text": text, "target_lang": target_lang}
