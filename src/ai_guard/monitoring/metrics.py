"""
src/ai_guard/monitoring/metrics.py

Prometheus metrics for AI-Guard.

This module defines application-level metrics for monitoring gateway behavior.
"""

from celery.exceptions import CPendingDeprecationWarning
from prometheus_client import Counter, Histogram, Gauge

GUARD_REQUESTS_TOTAL = Counter(
    "ai_guard_guard_request_total",
    "Total number of /guard requests."
)

ALLOWED_REQUESTS_TOTAL = Counter(
    "ai_guard_allowed_requests_total",
    "Total number of allowed /guard requests.",
)

BLOCKED_REQUESTS_TOTAL = Counter(
    "ai_guard_blocked_requests_total",
    "Total number of blocked /guard requests.",
    ["blocked_by"],
)

GUARD_REQUEST_LATENCY_SECONDS = Histogram(
    "ai_guard_guard_request_latency_seconds",
    "Latency of /guard requests in seconds.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

NLP_SCORE_GAUGE = Gauge(
    "ai_guard_last_nlp_score",
    "Last observed NLP firewall risk score.",
)

DRIFT_TASKS_SUBMITTED_TOTAL = Counter(
    "ai_guard_drift_tasks_submitted_total",
    "Total number of submitted drift check tasks.",
)

NETWORK_SAMPLE_CHECKS_TOTAL = Counter(
    "ai_guard_network_sample_checks_total",
    "Total number of internal network sample checks.",
)

NETWORK_SAMPLE_BLOCKED_TOTAL = Counter(
    "ai_guard_network_sample_blocked_total",
    "Total number of internal network samples classified as DDoS.",
)