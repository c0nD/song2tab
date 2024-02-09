# Official lightweight Python image.
FROM python:3.8-slim

# Set environment variables.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /app.
WORKDIR /app

# Copy the dependencies file to the working directory.
COPY requirements.txt .

# Install any dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory.
COPY . .

# Command to run on container start.
# Replace `app.py` with the script you use to run your application.
CMD ["python", "models/experimentation/train_tab_cnn.py"]
