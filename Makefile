all: help

help:
	@echo "To install required packages, run 'make install' from a clean 'python:3.9' (or higher) conda environment."

install:
	pip install -r requirements.txt

pylint:
	pylint imagegen
	pylint tests

pytest:
	pytest --cov=imagegen tests
	coverage xml
