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

docker_jupyter:
	jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root

docker_build:
	docker build . -t kostaleonard/mlops-image-gen

docker_run:
	docker run --gpus all --name mlops-image-gen -itd -p 8888:8888 kostaleonard/mlops-image-gen

docker_rm:
	docker rm mlops-image-gen

docker_push:
	docker push kostaleonard/mlops-image-gen

docker_pull:
	docker pull kostaleonard/mlops-image-gen

run_train:
	PYTHONPATH=. nohup python3 imagegen/train_model.py & | tee train_log.txt