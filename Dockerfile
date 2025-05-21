FROM python:3.11-slim

# Environment config
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system tools and uv
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv and make sure it's available in PATH
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Verify uv installation
RUN uv --version

# Copy only dependency files first to cache install layer
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies using uv
RUN uv sync --locked

# Install required packages directly using pip to ensure they're available
# This is not the best practive for production, but it ensures the packages are
# available for testing purpose
RUN pip install flask openai sqlalchemy pytest upstash-vector flask-sqlalchemy

# Verify pytest and flask are installed
RUN python -c "import pytest, flask, openai, sqlalchemy; print(f'Dependencies verified: pytest={pytest.__version__}, flask={flask.__version__}')"

# Copy the rest of the app
COPY . .

# Expose Flask port
EXPOSE 8080

# Run Flask app
CMD ["uv", "run", "main.py"]
