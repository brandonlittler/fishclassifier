FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install NumPy 1.26.4 first (required for PyTorch 2.2.0 compatibility)
RUN pip install --no-cache-dir "numpy==1.26.4"

# Install PyTorch CPU-only from official index
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu

# Copy requirements and install other dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt && \
    pip check && \
    python -c "import numpy; assert numpy.__version__ == '1.26.4', f'NumPy version {numpy.__version__} is wrong, expected 1.26.4'; print(f'NumPy version verified: {numpy.__version__}')" && \
    pip show numpy

# Pre-download EfficientNet weights to avoid runtime download issues
RUN python -c "from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights; efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)"

# Copy application files (build context should be parent directory)
COPY grd.py model.py transforms.py ./
COPY labels.json fish_classifier.pth ./

# Expose the UI port for Gradio to run on
EXPOSE 7860

# Set env vars for Gradio - need 0.0.0.0 to be able to access the UI from outside
ENV GRADIO_SERVER_NAME=0.0.0.0

# Run the application
CMD ["python", "grd.py"]