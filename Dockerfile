# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add Flask and CORS for the web API
RUN pip install --no-cache-dir flask flask-cors

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/chroma /app/data

# Copy data files if they exist
COPY data/ /app/data/

# Set environment variables
ENV PYTHONPATH=/app
ENV CHROMA_PATH=/app/chroma
ENV DATA_PATH=/app/data
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "app.py"] 