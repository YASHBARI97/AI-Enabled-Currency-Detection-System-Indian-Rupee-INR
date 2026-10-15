.PHONY: install test api web web-build
install:
	pip install -r requirements.txt && cd frontend && npm install
test:
	pytest -q && cd frontend && npm run typecheck && npm test
api:
	uvicorn app.api:app --reload --port 8000
web:
	cd frontend && npm run dev
web-build:
	cd frontend && npm run build && npm start
