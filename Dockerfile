FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY neuroforge ./neuroforge
COPY configs ./configs
COPY docker ./docker
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 neuroforge \
    && mkdir -p /app/logs /app/models /app/data \
    && chown -R neuroforge:neuroforge /app
EXPOSE 8000
RUN chmod +x docker/entrypoint.sh
USER 10001:10001
ENTRYPOINT ["./docker/entrypoint.sh"]
