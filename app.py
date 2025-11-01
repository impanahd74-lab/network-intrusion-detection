import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
PRED_SUMMARY_PATH = MODEL_DIR / "last_predict.json"
MODEL_PATH = MODEL_DIR / "best_model.joblib"

ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "nids-secret-key")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

for directory in (UPLOAD_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


state: Dict[str, Optional[object]] = {
    "metrics": None,
    "best_model": None,
    "best_model_name": None,
    "target_column": None,
    "last_prediction": None,
    "last_trained": None,
}


MODELS = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=500, multi_class="auto"),
    "Random Forest": RandomForestClassifier(random_state=42),
}


def _sanitize_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return value


def _sanitize_records(records):
    return [
        {key: _sanitize_value(val) for key, val in record.items()}
        for record in records
    ]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - surface message to user
        raise ValueError(f"Unable to read CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV appears to be empty.")

    if df.shape[1] < 2:
        raise ValueError(
            "The CSV must contain at least one feature column and one target column."
        )
    return df


def split_features_target(df: pd.DataFrame):
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y, target_column


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if len(categorical_cols) > 0:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    if not transformers:
        transformers.append(("identity", "passthrough", X.columns))

    return ColumnTransformer(transformers)


def train_models(df: pd.DataFrame):
    X, y, target_column = split_features_target(df)
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    metrics = {}
    best_model_name = None
    best_score = -np.inf
    best_pipeline: Optional[Pipeline] = None

    for name, estimator in MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics[name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "classification_report": report,
            "confusion_matrix": cm,
        }

        if accuracy > best_score:
            best_score = accuracy
            best_model_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("Model training failed to produce a valid pipeline.")

    joblib.dump(best_pipeline, MODEL_PATH)

    metrics_payload = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "target_column": target_column,
        "best_model": best_model_name,
        "models": metrics,
    }

    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2))

    state.update(
        {
            "metrics": metrics_payload,
            "best_model": best_pipeline,
            "best_model_name": best_model_name,
            "target_column": target_column,
            "last_trained": metrics_payload["trained_at"],
        }
    )


def load_best_model() -> Optional[Pipeline]:
    if state.get("best_model") is not None:
        return state["best_model"]

    if MODEL_PATH.exists():
        try:
            pipeline = joblib.load(MODEL_PATH)
        except Exception:  # pragma: no cover
            return None
        state["best_model"] = pipeline
        return pipeline
    return None


def load_metrics() -> Optional[Dict[str, object]]:
    if state.get("metrics") is not None:
        return state["metrics"]

    if METRICS_PATH.exists():
        try:
            payload = json.loads(METRICS_PATH.read_text())
        except json.JSONDecodeError:
            return None
        state["metrics"] = payload
        state["best_model_name"] = payload.get("best_model")
        state["target_column"] = payload.get("target_column")
        state["last_trained"] = payload.get("trained_at")
        return payload
    return None


def load_prediction_summary() -> Optional[Dict[str, object]]:
    if state.get("last_prediction") is not None:
        return state["last_prediction"]

    if PRED_SUMMARY_PATH.exists():
        try:
            payload = json.loads(PRED_SUMMARY_PATH.read_text())
        except json.JSONDecodeError:
            return None
        state["last_prediction"] = payload
        return payload
    return None


def persist_prediction_summary(summary: Dict[str, object]):
    PRED_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    state["last_prediction"] = summary


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_):
    flash("The uploaded file exceeds the 50 MB limit.")
    return redirect(request.referrer or url_for("index"))


@app.route("/")
def index():
    metrics = load_metrics()
    prediction_summary = load_prediction_summary()
    return render_template(
        "index.html",
        metrics=metrics,
        prediction_summary=prediction_summary,
        best_model=metrics.get("best_model") if metrics else None,
    )


@app.route("/train", methods=["POST"])
def train():
    file = request.files.get("train_file")
    if file is None or file.filename == "":
        flash("Please select a CSV file to train the models.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Only CSV files are supported for training.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    destination = UPLOAD_DIR / filename
    file.save(destination)

    try:
        df = load_csv(destination)
        train_models(df)
        flash("Training completed successfully. View the results below.", "success")
    except ValueError as exc:
        flash(str(exc))
    except Exception as exc:  # pragma: no cover
        flash(f"An unexpected error occurred during training: {exc}")

    return redirect(url_for("results"))


@app.route("/results")
def results():
    metrics = load_metrics()
    prediction_summary = load_prediction_summary()
    return render_template(
        "results.html",
        metrics=metrics,
        prediction_summary=prediction_summary,
    )


@app.route("/predict", methods=["POST"])
def predict():
    model = load_best_model()
    metrics = load_metrics()

    if model is None or metrics is None:
        flash("Train the models before running predictions.")
        return redirect(url_for("index"))

    file = request.files.get("predict_file")
    if file is None or file.filename == "":
        flash("Please select a CSV file for prediction.")
        return redirect(url_for("results"))

    if not allowed_file(file.filename):
        flash("Only CSV files are supported for prediction.")
        return redirect(url_for("results"))

    filename = secure_filename(file.filename)
    destination = UPLOAD_DIR / filename
    file.save(destination)

    try:
        df = load_csv(destination)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("results"))

    target_column = metrics.get("target_column")
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])

    try:
        predictions = model.predict(df)
    except Exception as exc:  # pragma: no cover
        flash(f"Failed to generate predictions: {exc}")
        return redirect(url_for("results"))

    counts = pd.Series(predictions).value_counts().sort_index()
    counts_dict = {str(label): int(value) for label, value in counts.items()}
    sample_records = _sanitize_records(df.head(5).to_dict(orient="records"))
    summary = {
        "predicted_at": datetime.utcnow().isoformat() + "Z",
        "total_predictions": int(len(predictions)),
        "counts": counts_dict,
        "sample": sample_records,
    }

    persist_prediction_summary(summary)
    flash("Predictions generated successfully.", "success")
    return redirect(url_for("results"))


@app.route("/metrics.json")
def metrics_json() -> Response:
    metrics = load_metrics()
    if metrics is None:
        return jsonify({"error": "No metrics available."}), 404
    return jsonify(metrics)


@app.route("/pred_summary.json")
def prediction_summary_json() -> Response:
    summary = load_prediction_summary()
    if summary is None:
        return jsonify({"error": "No prediction summary available."}), 404
    return jsonify(summary)


if __name__ == "__main__":
    app.run(debug=True)
