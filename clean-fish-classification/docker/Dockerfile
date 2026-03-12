FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy==1.26.4"
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu
COPY docker/requirements.txt .
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt && \
    pip check && \
    python -c "import numpy; assert numpy.__version__ == '1.26.4', f'NumPy version {numpy.__version__} is wrong, expected 1.26.4'; print(f'NumPy version verified: {numpy.__version__}')" && \
    pip show numpy
RUN python -c "from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights; efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)"
COPY grd.py model.py transforms.py ./
COPY labels.json fish_classifier.pth ./
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
CMD ["python", "grd.py"]
