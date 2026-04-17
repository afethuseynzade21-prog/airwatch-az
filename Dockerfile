# ── AirWatch AZ — Multi-stage Dockerfile ─────────────────────────────────────
# Stage 1: build deps (cached layer; only rebuilds on requirements change)
# Stage 2: lean runtime image

FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for geopandas, lightgbm, prophet
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgeos-dev libproj-dev gdal-bin libgdal-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="AirWatch AZ" \
      description="Baku Air Quality Intelligence Platform" \
      version="2.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev libproj-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY . .

# Create data + output dirs
RUN mkdir -p data/raw data/db outputs logs

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

# Expose ports: 8501 (Streamlit) | 8000 (FastAPI)
EXPOSE 8501 8000

# Default: Streamlit dashboard
# Override with: docker run ... uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
