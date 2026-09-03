# Multi-stage build for UIDAI Governance Intelligence Framework
# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/cleaned data/analysis

# Default port (Cloud Run sets PORT=8080 by default; local defaults to 8080 or 8501)
ENV PORT=8080

# Expose container port
EXPOSE 8080 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Default command: run Streamlit dashboard honoring $PORT for Cloud Run and local environments
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]

# Made with Bob
