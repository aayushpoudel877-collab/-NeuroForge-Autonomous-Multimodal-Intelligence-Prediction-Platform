FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY neuroforge ./neuroforge
COPY configs ./configs
COPY docker ./docker
RUN pip install --no-cache-dir .
EXPOSE 8000
RUN chmod +x docker/entrypoint.sh
ENTRYPOINT ["./docker/entrypoint.sh"]
