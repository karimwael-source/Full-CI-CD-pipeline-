"""
check_threshold.py - The Gatekeeper Script.

This script runs in the DEPLOY job. It:
  1. Reads the Run ID from model_info.txt
  2. Connects to MLflow and fetches the accuracy for that run
  3. Compares it against the threshold (0.85)
  4. If accuracy < 0.85 → exits with code 1 (FAILS the pipeline)
  5. If accuracy >= 0.85 → exits with code 0 (pipeline continues)

When a script exits with code 1, GitHub Actions marks the step
as FAILED and stops the job. This is how we "block" bad models.
"""

import sys
import mlflow

THRESHOLD = 0.85  # Minimum acceptable accuracy

# ─────────────────────────────────────────────
# 1. Read the Run ID written by train.py
# ─────────────────────────────────────────────
try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Checking Run ID: {run_id}")
except FileNotFoundError:
    print("ERROR: model_info.txt not found. Did the validate job run?")
    sys.exit(1)

# ─────────────────────────────────────────────
# 2. Connect to MLflow and fetch the run's data
# ─────────────────────────────────────────────
client = mlflow.tracking.MlflowClient()

try:
    run = client.get_run(run_id)
except Exception as e:
    print(f"ERROR: Could not fetch run from MLflow. Details: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# 3. Extract the accuracy metric
# ─────────────────────────────────────────────
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("ERROR: 'accuracy' metric not found in this MLflow run.")
    sys.exit(1)

print(f"Model accuracy from MLflow: {accuracy:.4f}")
print(f"Required threshold:         {THRESHOLD}")

# ─────────────────────────────────────────────
# 4. Make the decision
# ─────────────────────────────────────────────
if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}.")
    print("Blocking deployment. The model is not good enough for production.")
    sys.exit(1)  # Exit code 1 = FAILURE → GitHub Actions fails the step
else:
    print(f"PASSED: Accuracy {accuracy:.4f} meets the threshold {THRESHOLD}.")
    print("Model approved for deployment!")
    sys.exit(0)  # Exit code 0 = SUCCESS → pipeline continues