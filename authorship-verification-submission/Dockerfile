# Base image
FROM python:3.8-slim

# Set working directory in the container
WORKDIR /code

# Install Python dependencies
RUN pip install numpy pandas scikit-learn joblib

# Copy all local files into the container at /code
COPY . /code

# Define the command to run when the container starts
ENTRYPOINT ["python", "/code/run.py"]