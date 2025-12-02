"""
Runner module for model training and prediction.
"""

from typing import Dict, Any
import pandas as pd

from .rf_model import RFClassificationPieline
from ..alert.dooray import dooray_notify
from ..database.connector import MariaDBHandler


# Constants
PREDICTION_TABLE = "test_table"
PREDICTION_TABLE_SCHEMA = """
    id           INT AUTO_INCREMENT PRIMARY KEY,
    predict_time DATETIME NOT NULL,
    code         VARCHAR(100),
    svc_type     VARCHAR(100),
    svr_type     VARCHAR(100),
    instance     VARCHAR(100),
    object       VARCHAR(100),
    number       FLOAT
"""


def train_model(
    data: pd.DataFrame,
    pred_object: str,
    model_path: str,
    model_name: str,
    threshold: float,
    window_size: int,
) -> None:
    """
    Train a Random Forest classification model.

    Args:
        data: Training data DataFrame
        pred_object: Target object type (cpu/memory/disk)
        model_path: Directory to save the model
        model_name: Name of the model file
        threshold: Threshold value for classification
        window_size: Prediction window size in minutes
    """
    pipeline = RFClassificationPieline(window_size)
    report_value = pipeline.train(data, threshold, test_size=0.2)
    pipeline.save_model(
        output_dir=model_path, model_name=model_name, report_value=report_value
    )


def predict_data(
    data: pd.DataFrame,
    pred_object: str,
    model_path: str,
    model_name: str,
    alert_config: Dict[str, Any],
    server_info: Dict[str, str],
    handler: MariaDBHandler,
    conditions: Dict[str, str],
) -> None:
    """
    Make predictions using a trained model and handle alerts.

    Args:
        data: Prediction data DataFrame
        pred_object: Target object type (cpu/memory/disk)
        model_path: Directory containing the model
        model_name: Name of the model file
        alert_config: Alert configuration dictionary
        server_info: Server information dictionary
        handler: Database handler instance
        conditions: Condition dictionary for identification
    """
    # Load model
    pipeline = RFClassificationPieline.load_model(
        output_dir=model_path, model_name=model_name
    )

    # Clean data if necessary
    data = _clean_prediction_data(data, pred_object)
    # Make predictions
    results = pipeline.predict(data, pred_object)

    if results is None or len(results) == 0:
        print("No valid predictions generated")
        return

    # Get latest prediction
    latest = results.iloc[-1]
    threshold = float(pipeline.best_threshold)

    # Build prediction result
    prediction_result = _build_prediction_result(
        pred_object, latest, threshold, pipeline.prediction_minutes
    )

    # Handle high-risk predictions
    if prediction_result["will_be_high"]:
        _handle_high_risk_prediction(
            prediction_result,
            alert_config,
            server_info,
            handler,
            conditions,
            pred_object,
            threshold,
        )
    else:
        _log_normal_prediction(prediction_result, threshold)


def _clean_prediction_data(data: pd.DataFrame, pred_object: str) -> pd.DataFrame:
    """Remove unnecessary columns from prediction data."""
    columns_to_drop = []

    # CPU-specific cleanup
    if pred_object == "cpu" and "process_cpu_time" in data.columns:
        columns_to_drop.append("process_cpu_time")

    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)

    return data


def _build_prediction_result(
    pred_object: str, latest: pd.Series, threshold: float, prediction_minutes: int
) -> Dict[str, Any]:
    """Build prediction result dictionary."""
    return {
        "object_value": pred_object,
        "timestamp": latest["timestamp"],
        "current_value": float(latest["current_value"]),
        "probability": float(latest["probability"]),
        "prediction": int(latest["prediction"]),
        "prediction_label": str(latest["prediction_label"]),
        "will_be_high": bool(latest["prediction"] == 1),
        "threshold_used": threshold,
        "prediction_time": f"{prediction_minutes}분 후",
    }


def _handle_high_risk_prediction(
    prediction_result: Dict[str, Any],
    alert_config: Dict[str, Any],
    server_info: Dict[str, str],
    handler: MariaDBHandler,
    conditions: Dict[str, str],
    pred_object: str,
    threshold: float,
) -> None:
    """Handle high-risk prediction by sending alerts and storing in database."""
    # Send alert
    dooray_notify(prediction_result, alert_config, server_info, threshold, conditions)

    # Prepare data for database
    insert_data = {
        "predict_time": prediction_result["timestamp"],
        "code": conditions["CODE"],
        "svc_type": conditions["SVC_TYPE"],
        "svr_type": conditions["SVR_TYPE"],
        "instance": conditions["INSTANCE"],
        "object": pred_object,
        "number": prediction_result["current_value"],
    }

    # Store in database
    _store_prediction(handler, insert_data)


def _store_prediction(handler: MariaDBHandler, insert_data: Dict[str, Any]) -> None:
    """Store prediction in database, creating table if necessary."""
    if handler.table_exists(PREDICTION_TABLE):
        print(f"Table '{PREDICTION_TABLE}' exists. Inserting data.")
        handler.insert(PREDICTION_TABLE, insert_data)
    else:
        print(f"Table '{PREDICTION_TABLE}' does not exist. Creating table.")
        handler.create_table(PREDICTION_TABLE, PREDICTION_TABLE_SCHEMA)
        handler.insert(PREDICTION_TABLE, insert_data)


def _log_normal_prediction(prediction_result: Dict[str, Any], threshold: float) -> None:
    """Log normal prediction (no alert needed)."""
    print(
        f"Normal prediction - No alert sent. "
        f"Threshold: {threshold:.2f}, "
        f"Probability: {prediction_result['probability']:.2f}"
    )
