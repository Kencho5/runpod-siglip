FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn transformers sentencepiece protobuf pillow requests numpy
RUN python -c "from transformers import AutoModel, AutoProcessor; AutoModel.from_pretrained('google/siglip2-so400m-patch14-384'); AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')"
ENV HF_HUB_OFFLINE=1
COPY server.py .
CMD ["python", "-u", "server.py"]
