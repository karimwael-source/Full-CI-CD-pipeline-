# ─────────────────────────────────────────────────────────────
# Dockerfile for the ML Application
# ─────────────────────────────────────────────────────────────
# This file is the "blueprint" Docker uses to build a container.
# Think of a container like a tiny isolated computer that has
# exactly what your app needs — nothing more, nothing less.
# ─────────────────────────────────────────────────────────────

# STEP 1: Choose the base image.
# "python:3.10-slim" = a minimal Linux OS with Python 3.10 pre-installed.
# "slim" means it strips out unnecessary packages to keep size small.
FROM python:3.10-slim

# STEP 2: Declare a Build Argument.
# ARG is like a variable you can pass IN when running "docker build".
# The GitHub Action will pass the MLflow Run ID here:
#   docker build --build-arg RUN_ID=abc123 ...
# Inside the container, RUN_ID becomes available as a variable.
ARG RUN_ID

# STEP 3: Set an Environment Variable from the ARG.
# ENV makes the value available to processes running INSIDE the container
# (not just during the build). This lets your app code read it via
# os.environ["MLFLOW_RUN_ID"].
ENV MLFLOW_RUN_ID=${RUN_ID}

# STEP 4: Set the working directory inside the container.
# All subsequent commands will run from /app.
WORKDIR /app

# STEP 5: Copy and install Python dependencies.
# We copy requirements.txt first (before code) because Docker caches
# each step. If only code changes, Docker reuses the cached pip install
# layer — making rebuilds much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 6: Copy the application source code into the container.
COPY src/ ./src/

# STEP 7: Simulate downloading the model from MLflow.
# In a real project, this would run:
#   mlflow artifacts download --artifact-uri runs:/<RUN_ID>/model --dst-path /opt/model
# For this assignment, we echo to simulate it (since we may not have
# a publicly accessible MLflow server at build time).
RUN echo "Simulating model download for Run ID: ${RUN_ID}" && \
    mkdir -p /opt/model && \
    echo "Model downloaded from MLflow run: ${RUN_ID}" > /opt/model/model_info.txt

# STEP 8: Default command when the container starts.
# This just prints that the app is running (in a real project,
# this would start a Flask/FastAPI server).
CMD ["python", "-c", "import os; print(f'ML App is running. Model Run ID: {os.environ.get(\"MLFLOW_RUN_ID\", \"not set\")}')"]