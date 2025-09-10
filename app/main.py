from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import translate, ingest
from .routes import hs_users
from .database import Base, engine
import sentry_sdk

sentry_sdk.init(
    dsn="https://9aa1cba62037b2a28e85948ac2ec2049@o4509993463578624.ingest.de.sentry.io/4509993483698256",
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
    # Enable sending logs to Sentry
    enable_logs=True,
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
    traces_sample_rate=1.0,
    # Set profile_session_sample_rate to 1.0 to profile 100%
    # of profile sessions.
    profile_session_sample_rate=1.0,
    # Set profile_lifecycle to "trace" to automatically
    # run the profiler on when there is an active transaction
    profile_lifecycle="trace",
)

app = FastAPI(title="GlobalFlow.ai - Phase 1")

Base.metadata.create_all(bind=engine)

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
app.include_router(hs_users.router, prefix="/api")
