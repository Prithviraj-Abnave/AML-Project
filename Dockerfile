# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies required for Playwright and other libraries
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and its dependencies
RUN playwright install --with-deps chromium

# Copy the rest of the application
COPY . .

# Expose the port Flask runs on (Hugging Face Spaces defaults to 7860)
EXPOSE 7860

# Run the application with Gunicorn, binding to port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
