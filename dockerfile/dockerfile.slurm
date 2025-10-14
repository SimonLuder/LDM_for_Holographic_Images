# Use the official PyTorch image as base
FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the entire repository into the container
COPY ./requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENV PYTHONPATH=/app