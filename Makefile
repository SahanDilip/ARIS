# ARIS Step 1. Every target is one command and leaves something runnable.
PY ?= python

.PHONY: help install test smoke step1 step1-hard step1-realistic serve ollama clean

help:
	@echo "make install          - install dependencies"
	@echo "make test             - run the test suite (includes the information-barrier test)"
	@echo "make smoke            - 40 s sanity run, no LLM needed"
	@echo "make step1            - Step 1 report, base config (easy testbed)"
	@echo "make step1-realistic  - Step 1 report, RECOMMENDED config"
	@echo "make step1-hard       - Step 1 report, hardened config"
	@echo "make ollama           - check which local models are installed"
	@echo "make serve            - start the black-box platform on :8000"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

smoke:
	$(PY) -m scripts.step1_report --conf realistic --tiny --no-llm

step1:
	$(PY) -m scripts.step1_report --conf base

step1-realistic:
	$(PY) -m scripts.step1_report --conf realistic

step1-hard:
	$(PY) -m scripts.step1_report --conf hard

ollama:
	$(PY) -m aris.llm.bench

serve:
	$(PY) -m aris.platform.server --oracle O2 --port 8000

clean:
	rm -rf outputs .pytest_cache **/__pycache__
