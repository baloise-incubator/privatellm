init:
	poetry install

format:
	poetry run ruff format .
	poetry run ruff . --fix

format-check:
	poetry run ruff format . --check

lint:
	poetry run ruff .

mypy:
	poetry run mypy --install-types --non-interactive .


test:
	poetry run pytest -vv -s --typeguard-packages=privatellm

coverage:
	poetry run coverage run -m pytest
	poetry run coverage html
	poetry run coverage report

checks: format-check lint mypy test

image:
	DOCKER_BUILDKIT=1 docker build --progress=plain -t texttitan:latest .

docs:
	mkdocs serve

update:
	poetry lock

.PHONY: init format format-check lint mypy test coverage checks image docs update
