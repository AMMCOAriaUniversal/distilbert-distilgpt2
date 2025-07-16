ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Use a base image with GPU support if needed
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy your application code
COPY . .

# Set environment variables if needed
ENV TF_ENABLE_ONEDNN_OPTS=0

# Command to run your app
CMD ["python3", "app.py"]
