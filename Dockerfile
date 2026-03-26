FROM python:3.10-slim
ARG RUN_ID
ENV MLFLOW_RUN_ID=${RUN_ID}
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# here start to download the model from the workflow using the run id and save it in /opt/model directory in the container
RUN echo "Simulating model download for Run ID: ${RUN_ID}" && \
    mkdir -p /opt/model && \
    echo "Model downloaded from MLflow run: ${RUN_ID}" > /opt/model/model_info.txt
CMD ["python", "-c", "import os; print(f'ML App is running. Model Run ID: {os.environ.get(\"MLFLOW_RUN_ID\", \"not set\")}')"]