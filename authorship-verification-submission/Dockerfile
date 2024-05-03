# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /code

# Install dependencies
RUN pip install numpy pandas scikit-learn joblib

# Copy all files into the container
COPY . /code

# Command to run the script
CMD ["python", "run.py"]
