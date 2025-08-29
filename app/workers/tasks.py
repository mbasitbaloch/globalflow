from celery import Celery
from ..services.rag import embed_and_store
from ..config import settings

celery = Celery("tasks", broker=settings.REDIS_URL)


@celery.task
def generate_embedding(text: str, translation_id: int):
    embed_and_store(text, translation_id)
