FROM python:3.10-slim
WORKDIR /app

# Install system deps for common ML packages (may vary per image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command: run setup validator
CMD ["python", "setup_local.py"]
