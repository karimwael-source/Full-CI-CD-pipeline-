import sys
import mlflow

THRESHOLD = 0.85  # Minimum acceptable accuracy

# read the model information of evaluation 
try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Checking Run ID: {run_id}")
except FileNotFoundError:
    print("ERROR: model_info.txt not found. Did the validate job run?")
    sys.exit(1)

#connecting to mlflow and fetching the run details using the run_id
client = mlflow.tracking.MlflowClient()

try:
    run = client.get_run(run_id)
except Exception as e:
    print(f"ERROR: Could not fetch run from MLflow. Details: {e}")
    sys.exit(1)

# extracting the accuracy metrics from run data 
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("ERROR: 'accuracy' metric not found in this MLflow run.")
    sys.exit(1)

print(f"Model accuracy from MLflow: {accuracy:.4f}")
print(f"Required threshold:         {THRESHOLD}")

# decide if the model's accuracy passed or not 
if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}.")
    print("Blocking deployment. The model is not good enough for production.")
    sys.exit(1) # if model's accuracy does not pass 0.85 so it will be shown on github that the code exit with 1 
else:
    print(f"PASSED: Accuracy {accuracy:.4f} meets the threshold {THRESHOLD}.")
    print("Model approved for deployment!")
    sys.exit(0)  # Exit code 0 = SUCCESS → pipeline continues

    # if appears on github on runing the workflow that the code exit with 0 so it means the model's accuracy passed the threshold (0.85) 
    # and the deployment will continue to be deployed to production environment.

    # and if it appears on git hub on runing workflow that the code exit with 1 so that means the model's accuracy did not pass the threshold(0.85) 
    # it is not allowed to be deployed for production environment.