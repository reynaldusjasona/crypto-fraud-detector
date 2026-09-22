.PHONY: install collect train api dashboard test lint clean

install:
	pip install -r requirements.txt

collect:
	python -m src.collector.etherscan
	python -m src.collector.dataset

train:
	python -m src.models.train

api:
	uvicorn src.api.main:app --reload --port 8000

dashboard:
	streamlit run src/dashboard/app.py

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
