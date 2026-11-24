# RAPIDS base image (cuDF, cuML, cuGraph, dask-cudf preinstalled). Pin to a release you have tested.
FROM rapidsai/base:25.10-cuda12.8-py3.12

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . matplotlib

ENV ANTHROPIC_API_KEY=""
ENTRYPOINT ["rapids-analyst"]
