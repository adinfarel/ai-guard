"""
src/ai_guard/worker/celery_app.py

Celery application for AI-Guard background jobs.

Redis is used as broker and result backend.
"""

import os

from celery import Celery

REDIS_URL = os.getenv("AI_GUARD_REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ai_guard_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.ai_guard.worker.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_time_limit=60 * 30,
    task_soft_time_limit=60 * 25,
    result_expires=60 * 60,
    timezone="UTC",
    enable_utc=True
)