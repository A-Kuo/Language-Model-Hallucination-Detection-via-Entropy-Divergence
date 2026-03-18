# ─────────────────────────────────────────────────────────
# Hallucination Detection via Attention Entropy
# Multi-stage Docker build: test → production
# ─────────────────────────────────────────────────────────

# Stage 1: Base with dependencies
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ─────────────────────────────────────────────────────────
# Stage 2: Test runner
FROM base AS test

COPY . .

# Run test suite (fail build if tests fail)
RUN python -m pytest tests/ -v --tb=short -x

# Run synthetic evaluation
RUN python evaluate.py --num_samples 200 --output /tmp/test_metrics.json

# ─────────────────────────────────────────────────────────
# Stage 3: Production image
FROM base AS production

# Copy source code
COPY src/ src/
COPY run_demo.py .
COPY evaluate.py .
COPY download_model.py .
COPY README.md .

# Create non-root user
RUN useradd --create-home appuser
USER appuser

# Cache directory for HuggingFace models
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/hub

# Health check: import pipeline modules
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from src.hypothesis_test import HallucinationHypothesisTest; print('OK')"

# Default: run demo in synthetic mode
ENTRYPOINT ["python"]
CMD ["run_demo.py", "--synthetic"]

# ─────────────────────────────────────────────────────────
# Usage:
#
#   # Build (runs tests automatically)
#   docker build -t hallucination-detection .
#
#   # Run demo (synthetic, no GPU needed)
#   docker run hallucination-detection
#
#   # Run demo with custom prompt
#   docker run hallucination-detection run_demo.py --synthetic --prompt "The moon is made of"
#
#   # Run evaluation
#   docker run hallucination-detection evaluate.py --num_samples 1000
#
#   # Run with GPT-2 (downloads model on first run, ~500MB)
#   docker run -v hf_cache:/home/appuser/.cache/huggingface \
#       hallucination-detection run_demo.py --prompt "Hello world"
#
#   # Interactive shell
#   docker run -it --entrypoint bash hallucination-detection
# ─────────────────────────────────────────────────────────
