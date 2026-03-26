"""
train.py - Trains a classifier and logs everything to MLflow.
After training, it saves the Run ID to model_info.txt so the
deploy job can find this exact run later.

Model selection is controlled by MODEL_VARIANT env var:
- strong (default): RandomForest, typically passes threshold
- weak: DummyClassifier, typically fails threshold
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# 1. Load Data
#    In a real project, you'd load from data/
#    We use Iris here for simplicity.
# ─────────────────────────────────────────────
print("Loading data...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 2. Train Model inside an MLflow Run
#    mlflow.start_run() opens a "session" where
#    everything we log gets saved together.
# ─────────────────────────────────────────────
print("Starting MLflow run...")
mlflow.set_experiment("mlops-assignment")  # group runs under one experiment name

with mlflow.start_run() as run:
    # Save the Run ID — we need it in check_threshold.py
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Train
    model_variant = os.getenv("MODEL_VARIANT", "strong").strip().lower()
    if model_variant == "weak":
        model = DummyClassifier(strategy="most_frequent")
        mlflow.log_param("model_variant", "weak")
        mlflow.log_param("model_type", "DummyClassifier")
    else:
        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        mlflow.log_param("model_variant", "strong")
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Log parameters and metrics to MLflow
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model itself as an artifact
    mlflow.sklearn.log_model(model, "model")

    print("Logged to MLflow successfully.")

# ─────────────────────────────────────────────
# 3. Export the Run ID to model_info.txt
#    The deploy job will download this file and
#    use the Run ID to query MLflow for accuracy.
# ─────────────────────────────────────────────
with open("model_info.txt", "w") as f:
    f.write(run_id)

print(f"Saved Run ID '{run_id}' to model_info.txt")
print("Training complete!")