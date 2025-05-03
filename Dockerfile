# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (app.py, model, etc.)
COPY . .

# Expose streamlit port
EXPOSE 8501

# Run the app

CMD ["python", "-m", "streamlit", "run", "app.py"]
