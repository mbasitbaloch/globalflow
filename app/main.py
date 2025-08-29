from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import translate, ingest
from .database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="GlobalFlow.ai - Phase 1")


# Allow CORS for all origins (you can restrict to frontend domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(translate.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
