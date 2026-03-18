# ─────────────────────────────────────────────────────────
# Hallucination Detection v1 — Attention Entropy
# Multi-stage Docker build: test → production
# ─────────────────────────────────────────────────────────

# Stage 1: Base with dependencies
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ─────────────────────────────────────────────────────────
# Stage 2: Test runner
FROM base AS test

COPY . .

RUN python -m pytest test_attention_analyzer.py -v --tb=short -x
RUN python attention_analyzer.py
RUN python hypothesis_test.py
RUN python confidence_calibrator.py

# ─────────────────────────────────────────────────────────
# Stage 3: Production image
FROM base AS production

COPY attention_analyzer.py .
COPY hypothesis_test.py .
COPY confidence_calibrator.py .
COPY utils.py .
COPY README.md .
COPY claude1.md .

RUN useradd --create-home appuser
USER appuser

ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/hub

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from hypothesis_test import HallucinationHypothesisTest; print('OK')"

ENTRYPOINT ["python"]
CMD ["attention_analyzer.py"]

# ─────────────────────────────────────────────────────────
# Usage:
#
#   docker build -t hallucination-detection .
#   docker run hallucination-detection
#   docker run hallucination-detection hypothesis_test.py
#   docker run -it --entrypoint bash hallucination-detection
# ─────────────────────────────────────────────────────────
