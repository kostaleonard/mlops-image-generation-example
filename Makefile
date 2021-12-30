all: help

help:
	@echo "To install required packages, run 'make install' from a clean 'python:3.9' (or higher) conda environment."

install:
	pip install -r requirements.txt

pylint:
	pylint imagegen
	pylint tests

pytest:
	pytest -m "not slowtest and not awstest and not wandbtest" --cov=imagegen tests
	coverage xml

pytest_slow:
	pytest -m "slowtest" --cov=imagegen tests
	coverage xml

pytest_aws:
	pytest -m "awstest" --cov=imagegen tests
	coverage xml

pytest_wandb:
	pytest -m "wandbtest" --cov=imagegen tests
	coverage xml

pytest_full:
	pytest --cov=imagegen tests
	coverage xml
