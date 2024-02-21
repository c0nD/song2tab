# Use a base image with Python installed
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends gcc g++ libsndfile1 && \
#     rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install amt-tools without its dependencies
RUN pip install --no-deps amt-tools

RUN pip install --no-cache-dir $(grep -v 'evdev' requirements.txt)

# Copy the content of the local src directory to the working directory.
COPY src/ src/
COPY dataprocessing/ dataprocessing/
COPY models/ models/

# Specify the command to run on container start
CMD [ "python", "./models/experimentation/train_tab_cnn.py"]
