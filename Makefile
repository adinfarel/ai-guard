setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

gateway:
	uvicorn src.ai_guard.gateway.app:app --reload --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

test:
	python -m pytest tests/ -q