# Using a specific Python base image
FROM python:3.8-slim

# Set the working directory in the Docker container
WORKDIR /code

# Install necessary Python packages
RUN pip install numpy pandas scikit-learn joblib

# Add your script and model file to the container
COPY . /code

# Set the default command to execute the script when the container starts
ENTRYPOINT ["python", "/code/run.py"]
