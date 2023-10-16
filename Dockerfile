ARG BASE_IMAGE=ubuntu:22.04
FROM $BASE_IMAGE AS base

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install the venv
FROM base as installer
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-poetry python3-venv build-essential

COPY pyproject.toml poetry.lock ./
RUN which python
RUN poetry install
COPY privatellm privatellm

# assemble runtime image
FROM base
RUN apt-get update && apt-get install -y \
    python3 --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY --from=installer $VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=installer /app/privatellm/*.py /app/privatellm/

EXPOSE 8000

CMD ["uvicorn", "privatellm.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
