# Docker image for HuggingFace Spaces (SDK: docker, app_port: 7860)
FROM python:3.10-slim

# HF Spaces run containers as a non-root user; create one with a writable home
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the application code
COPY --chown=user:user . .

# Bake the embedding + NLI models into the image (fast, offline cold starts)
RUN python -m src.prefetch_models

# Gradio UI (public) runs on 7860; FastAPI backend runs internally on 8000
EXPOSE 7860

CMD ["bash", "start.sh"]
