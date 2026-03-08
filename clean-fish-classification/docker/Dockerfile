FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only from official index
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu

# Copy requirements and install other dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (build context should be parent directory)
COPY grd.py model.py transforms.py ./
COPY labels.json fish_classifier.pth ./

# Expose the UI port for Gradio to run on
EXPOSE 7860

# Set env vars for Gradio - need 0.0.0.0 to be able to access the UI from outside
ENV GRADIO_SERVER_NAME=0.0.0.0

# Run the application
CMD ["python", "grd.py"]