"""
Random Forest Classification Pipeline for resource utilization prediction.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# Constants
DEFAULT_PREDICTION_MINUTES = 60
DEFAULT_THRESHOLD = 0.5
MIN_ANOMALY_SAMPLES = 5
EPSILON = 1e-6

# Columns to exclude from features
EXCLUDED_COLUMNS = [
    "timestamp",
    "disk_util_future",
    "memory_util_future",
    "cpu_util_future",
    "target",
]


class RFClassificationPipeline:
    """
    Random Forest Classification Pipeline for time series anomaly prediction.
    """

    def __init__(self, prediction_minutes: int = DEFAULT_PREDICTION_MINUTES):
        """
        Initialize the pipeline.

        Args:
            prediction_minutes: Number of minutes ahead to predict
        """
        self.prediction_minutes = prediction_minutes
        self.model: Optional[RandomForestClassifier] = None
        self.best_threshold: float = DEFAULT_THRESHOLD
        self.feature_cols: Optional[List[str]] = None
        self.threshold: Optional[float] = None  # Target threshold (e.g., 85%)

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        min_recall: float = 0.8,
        min_f1: float = 0.5,
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        Find optimal classification threshold by maximizing F1-score.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            min_recall: Minimum required recall
            min_f1: Minimum required F1-score

        Returns:
            Tuple of (best_threshold, results_list)

        Raises:
            ValueError: If no threshold meets the minimum requirements
        """
        print("=" * 70)
        print("Searching for optimal threshold (maximizing F1-score)")
        print("=" * 70 + "\n")

        best_threshold = DEFAULT_THRESHOLD
        best_f1 = 0
        best_info = {}

        # Track best metrics even if they don't meet requirements
        best_f1_any = 0
        best_recall_any = 0
        best_precision_any = 0

        thresholds = np.arange(0.05, 0.74, 0.02)
        results = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            # Skip if no positive predictions
            if y_pred.sum() == 0:
                continue

            recall = recall_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)

            # Calculate F1-score
            if recall > 0 and precision > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            # Track best metrics regardless of requirements
            if f1 > best_f1_any and recall > 0.5:
                best_f1_any = f1
                best_recall_any = recall
                best_precision_any = precision

            # Check if this threshold meets requirements
            if f1 > best_f1 and recall >= min_recall and f1 >= min_f1:
                best_f1 = f1
                best_threshold = thresh
                best_info = {
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "n_predicted": int(y_pred.sum()),
                }

            results.append(
                {
                    "threshold": thresh,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                }
            )

        # Check if we found a valid threshold
        if not best_info:
            raise ValueError(
                f"Could not find threshold meeting min_recall={min_recall}. "
                f"Best metrics found: F1={best_f1_any:.3f}, "
                f"Recall={best_recall_any:.3f}, "
                f"Precision={best_precision_any:.3f}. "
                "Training performance insufficient."
            )

        # Print results
        print(f"\n{'='*70}")
        print(f"Optimal threshold: {best_threshold:.2f}")
        print(f"  - Recall: {best_info['recall']:.3f}")
        print(f"  - Precision: {best_info['precision']:.3f}")
        print(f"  - F1: {best_info['f1']:.3f}")
        print(f"{'='*70}\n")

        return best_threshold, results

    def train(
        self, df: pd.DataFrame, threshold: float, test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model with optimal threshold finding.

        Args:
            df: Training data with features and target
            threshold: Target threshold value (e.g., 85 for 85% utilization)
            test_size: Proportion of data to use for testing

        Returns:
            Dictionary containing evaluation metrics

        Raises:
            ValueError: If insufficient anomaly samples or training fails
        """
        self.threshold = threshold

        # Prepare features and target
        self.feature_cols = [col for col in df.columns if col not in EXCLUDED_COLUMNS]

        X = df[self.feature_cols]
        y = df["target"]

        # Print data statistics
        self._print_training_stats(df, y)

        # Validate sufficient anomaly samples
        if y.sum() < MIN_ANOMALY_SAMPLES:
            raise ValueError(
                f"Insufficient anomaly samples for training! "
                f"Found {y.sum()} samples, need at least {MIN_ANOMALY_SAMPLES}."
            )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        print(f"Train: {len(X_train):,} rows (Anomalies: {y_train.sum():,})")
        print(f"Test: {len(X_test):,} rows (Anomalies: {y_test.sum():,})\n")

        # Calculate class weights
        class_weight = self._calculate_class_weights(y_train)
        print(f"Class weights: {class_weight}\n")

        # Train model
        print("Training Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=400,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=3,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_train, y_train)
        print("✓ Training complete!\n")

        # Get predictions
        try:
            y_proba_test = self.model.predict_proba(X_test)[:, 1]
        except IndexError:
            raise ValueError(
                f"Model training failed! Insufficient test anomalies: "
                f"{y_test.sum()} samples"
            )

        # Find optimal threshold
        self.best_threshold, _ = self.find_optimal_threshold(
            y_test, y_proba_test, min_recall=0.8, min_f1=0.3
        )

        # Evaluate with optimal threshold
        y_pred_test = (y_proba_test >= self.best_threshold).astype(int)

        # Generate and print evaluation metrics
        report_value = self._evaluate_model(y_test, y_pred_test, self.best_threshold)

        # Print feature importance
        self._print_feature_importance()

        return report_value

    def save_model(
        self,
        output_dir: str = "./models",
        model_name: str = "model",
        report_value: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save trained model and configuration.

        Args:
            output_dir: Directory to save model files
            model_name: Base name for model files
            report_value: Evaluation metrics to save
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(output_dir, model_name)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save configuration
        config = {
            "best_threshold": float(self.best_threshold),
            "prediction_minutes": self.prediction_minutes,
            "feature_cols": self.feature_cols,
            "target_threshold": self.threshold,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation": report_value,
        }

        config_path = os.path.join(output_dir, f"{model_name}_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Print save confirmation
        print("=" * 70)
        print("Model saved successfully")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Config: {config_path}")
        print(f"\nSaved configuration:")
        print(f"  - Prediction threshold: {self.best_threshold:.2f}")
        print(f"  - Prediction window: {self.prediction_minutes} minutes")
        print(f"  - Target threshold: {self.threshold}%")
        print(f"  - Number of features: {len(self.feature_cols)}")
        print("=" * 70 + "\n")

    @classmethod
    def load_model(
        cls,
        output_dir: str = "./models",
        model_name: str = "model",
        best_threshold: Optional[float] = None,
    ) -> "RFClassificationPipeline":
        """
        Load saved model and configuration.

        Args:
            output_dir: Directory containing model files
            model_name: Base name of model files
            best_threshold: Optional custom threshold (uses saved if None)

        Returns:
            Loaded pipeline instance
        """
        # Load configuration
        config_path = os.path.join(output_dir, f"{model_name}_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load model
        model_path = os.path.join(output_dir, model_name)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Create pipeline instance
        pipeline = cls(prediction_minutes=config["prediction_minutes"])
        pipeline.model = model
        pipeline.feature_cols = config["feature_cols"]
        pipeline.threshold = config["target_threshold"]

        # Set threshold
        saved_threshold = config["best_threshold"]

        if best_threshold is not None:
            pipeline.best_threshold = best_threshold
            threshold_source = "User-specified"
        else:
            pipeline.best_threshold = saved_threshold
            threshold_source = "Saved optimal"

        # Print load confirmation
        print("Model loaded successfully")

        return pipeline

    def predict(self, df: pd.DataFrame, target_object: str) -> Optional[pd.DataFrame]:
        """
        Make predictions on new data.

        Args:
            df: Input data with at least 60 minutes of recent data
            target_object: Type of resource (cpu/memory/disk)

        Returns:
            DataFrame with predictions, or None if no valid data

        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded!")

        # Prepare data
        df_prep = df.copy()
        df_prep = df_prep.sort_values("timestamp").reset_index(drop=True)
        df_prep["timestamp"] = pd.to_datetime(df_prep["timestamp"])

        # Select feature columns
        available_features = [
            col for col in self.feature_cols if col in df_prep.columns
        ]

        # Remove missing values
        df_prep = df_prep[available_features].dropna()

        if len(df_prep) == 0:
            print("⚠ No valid data after feature generation!")
            return None

        # Make predictions
        X = df_prep[available_features]
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.best_threshold).astype(int)

        # Build results DataFrame
        results = pd.DataFrame(
            {
                "timestamp": df.loc[df_prep.index, "timestamp"],
                "current_value": df.loc[df_prep.index, f"{target_object}_utilization"],
                "probability": probabilities,
                "prediction": predictions,
                "prediction_label": [
                    "High VALUE" if p == 1 else "Normal" for p in predictions
                ],
                "threshold_used": self.best_threshold,
            }
        )

        return results.reset_index(drop=True)

    def _print_training_stats(self, df: pd.DataFrame, y: pd.Series) -> None:
        """Print training data statistics."""
        print("=" * 70)
        print("Model Training")
        print("=" * 70)
        print(f"Number of features: {len(self.feature_cols)}")
        print(f"Total samples: {len(df):,}")
        print(f"Anomalies (High): {y.sum():,} ({y.mean()*100:.2f}%)")
        print(f"Normal: {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.2f}%)\n")

    def _calculate_class_weights(self, y_train: pd.Series) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        high_count = y_train.sum()
        normal_count = len(y_train) - high_count

        # Limit weight ratio to prevent over-weighting
        weight_ratio = min((normal_count / high_count) * 2, 2)

        return {0: 1, 1: weight_ratio}

    def _evaluate_model(
        self, y_test: pd.Series, y_pred: np.ndarray, threshold: float
    ) -> Dict[str, Any]:
        """Evaluate model and print metrics."""
        print("=" * 70)
        print(f"Final Evaluation (threshold: {threshold:.2f})")
        print("=" * 70 + "\n")

        # Classification report
        report_dict = classification_report(
            y_test, y_pred, target_names=["Normal", "High"], output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        high_total = int(y_test.sum())
        high_missed = int(cm[1, 0])
        high_detected = int(cm[1, 1])
        high_missed_ratio = high_missed / high_total if high_total > 0 else None
        high_detected_ratio = high_detected / high_total if high_total > 0 else None

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(f"                Predicted Normal  Predicted High")
        print(f"Actual Normal   {cm[0,0]:>15}  {cm[0,1]:>14}")
        print(f"Actual High     {cm[1,0]:>15}  {cm[1,1]:>14}\n")

        # Print detection rates
        if high_total > 0:
            print(
                f"⚠ Missed anomalies: {high_missed}/{high_total} "
                f"({high_missed/high_total*100:.1f}%)"
            )
            print(
                f"✓ Detected anomalies: {high_detected}/{high_total} "
                f"({high_detected/high_total*100:.1f}%)\n"
            )

        return {
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "high": {
                "total": high_total,
                "missed": high_missed,
                "detected": high_detected,
                "missed_ratio": high_missed_ratio,
                "detected_ratio": high_detected_ratio,
            },
        }

    def _print_feature_importance(self, top_n: int = 10) -> None:
        """Print top N most important features."""
        print("=" * 70)
        print(f"Feature Importance (Top {top_n})")
        print("=" * 70)

        importances = self.model.feature_importances_
        top_features = (
            pd.DataFrame({"feature": self.feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        print(top_features.to_string(index=False))
        print()


# Legacy class name for backward compatibility
RFClassificationPieline = RFClassificationPipeline
