.PHONY: install test synthetic run dashboard

install:
	pip install -r requirements.txt && pip install -e .

test:
	python -m pytest -q

synthetic:            # offline smoke test, fake data under data/synthetic/
	PYTHONPATH=src python -m trading_ai.cli all --synthetic

run:                  # real data (needs internet)
	PYTHONPATH=src python -m trading_ai.cli all

dashboard:
	streamlit run dashboard/app.py
