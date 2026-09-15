.PHONY: help install pipeline app test clean

help:
	@echo "make install   - install Python dependencies"
	@echo "make pipeline  - run the full data + modelling pipeline"
	@echo "make app       - launch the Streamlit customer explorer"
	@echo "make test      - run the unit tests"
	@echo "make clean     - delete generated data/processed and models artifacts"

install:
	pip install -r requirements.txt

pipeline:
	python scripts/run_pipeline.py

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

clean:
	rm -f data/processed/*.csv data/processed/*.csv.gz models/*.joblib models/*.json
