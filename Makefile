CONDA_ENV := evolver-studio
RUN := conda run -n $(CONDA_ENV)

.PHONY: env lint format test run sync-resources

env:
	conda env create -n $(CONDA_ENV) -f environment.yml || conda env update -n $(CONDA_ENV) -f environment.yml

lint:
	$(RUN) ruff check .

format:
	$(RUN) ruff format .

test:
	$(RUN) pytest tests/ -x

run:
	$(RUN) streamlit run app.py

sync-resources:
	$(RUN) python scripts/sync_resources.py
