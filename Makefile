setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	pytest tests/

gateway:
	uvicorn src.ai_guard.gateway.app:app --reload --host 0.0.0.0 --port 8000