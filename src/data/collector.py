"""
Data collection module for Prometheus metrics.
Handles both training and prediction data collection.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import requests

from .query_handler import (
    make_cpu_queries,
    make_memory_queries,
    make_disk_queries,
    create_cpu_features,
    create_memory_features,
    create_disk_features,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Constants
API_TIMEOUT = 60
API_WAIT_TIME = 0.2
DAY_WAIT_TIME = 1
PREDICTION_WINDOW_MINUTES = 120
MIN_PREDICTION_WINDOW = 60


class Collector:
    """Prometheus data collector."""

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.api_endpoint = f"{self.prometheus_url}/api/v1"

    def get_metric_labels(
        self,
        metric_name: str,
        code: str,
        instance: str,
        svc_type: str,
        svr_type: str,
    ) -> Dict[str, List[str]]:
        """Query metric labels from Prometheus."""
        query = (
            f"{metric_name}{{"
            f'CODE="{code}",INSTANCE="{instance}",'
            f'SVC_TYPE="{svc_type}",SVR_TYPE="{svr_type}"}}'
        )
        url = f"{self.api_endpoint}/query"

        try:
            response = requests.get(url, params={"query": query}, timeout=5)
            result = response.json()

            if result.get("status") == "success":
                data = result.get("data", {}).get("result", [])
                labels = {}

                # Extract unique label values
                excluded_labels = {
                    "__name__",
                    "CODE",
                    "INSTANCE",
                    "SVC_TYPE",
                    "SVR_TYPE",
                    "job",
                    "instance",
                }

                for item in data:
                    metric = item.get("metric", {})
                    for key, value in metric.items():
                        if key not in excluded_labels:
                            if key not in labels:
                                labels[key] = set()
                            labels[key].add(value)

                return {k: sorted(list(v)) for k, v in labels.items()}
        except Exception as e:
            logger.error(f"Failed to get metric labels: {e}")

        return {}

    def query_range(
        self, query: str, start_time: datetime, end_time: datetime, step: str = "1m"
    ) -> Optional[Dict]:
        """Execute range query against Prometheus."""
        url = f"{self.api_endpoint}/query_range"
        params = {
            "query": query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": step,
        }

        try:
            response = requests.get(url, params=params, timeout=API_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None


def collect_train_data(
    target_object: str,
    instances: List[str],
    instances_ids: List[int],
    prometheus_url: str,
    end_time: datetime,
    window_size: int,
    code: str,
    svc_type: str,
    svr_type: str,
) -> pd.DataFrame:
    """Collect training data for multiple instances."""
    start_time = end_time - timedelta(days=window_size)
    collector = Collector(prometheus_url)

    all_data = []

    for instance, instance_id in zip(instances, instances_ids):
        filter_query = _create_filter_query(code, instance, svc_type, svr_type)

        data = _collect_data(
            collector=collector,
            target_object=target_object,
            filter_query=filter_query,
            code=code,
            instance=instance,
            svc_type=svc_type,
            svr_type=svr_type,
            start_time=start_time,
            end_time=end_time,
            window_size=window_size,
            instance_id=instance_id,
        )

        all_data.append(data)

    return pd.concat(all_data, axis=0, ignore_index=True)


# def collect_predict_data(
#     target_object: str,
#     instance: str,
#     instance_id: int,
#     prometheus_url: str,
#     end_time: datetime,
#     window_size: int,
#     code: str,
#     svc_type: str,
#     svr_type: str,
# ) -> pd.DataFrame:
#     """Collect prediction data for a single instance."""
#     start_time = end_time - timedelta(days=window_size)
#     collector = Collector(prometheus_url)
#     filter_query = _create_filter_query(code, instance, svc_type, svr_type)

#     data = _collect_data(
#         collector=collector,
#         target_object=target_object,
#         filter_query=filter_query,
#         code=code,
#         instance=instance,
#         svc_type=svc_type,
#         svr_type=svr_type,
#         start_time=start_time,
#         end_time=end_time,
#         window_size=window_size,
#         instance_id=instance_id,
#     )
#     return data.reset_index(drop=True)


def collect_predict_data(
    target_object: str,
    instance: str,
    instance_id: int,
    prometheus_url: str,
    end_time: datetime,
    window_size: int,
    code: str,
    svc_type: str,
    svr_type: str,
) -> pd.DataFrame:
    """Collect prediction data for a single instance."""
    print(f"\n{'='*70}")
    print(f"[DEBUG] collect_predict_data started")
    print(f"{'='*70}")
    print(f"  Target: {target_object}")
    print(f"  Instance: {instance} (ID: {instance_id})")
    print(f"  Code: {code}, SvcType: {svc_type}, SvrType: {svr_type}")
    print(f"  End time: {end_time}")
    print(f"  Window size: {window_size} day(s)")
    print(f"{'='*70}\n")

    start_time = end_time - timedelta(days=window_size)
    collector = Collector(prometheus_url)
    filter_query = _create_filter_query(code, instance, svc_type, svr_type)

    print(f"[DEBUG] Filter query: {filter_query}")
    print(f"[DEBUG] Time range: {start_time} ~ {end_time}\n")

    data = _collect_data(
        collector=collector,
        target_object=target_object,
        filter_query=filter_query,
        code=code,
        instance=instance,
        svc_type=svc_type,
        svr_type=svr_type,
        start_time=start_time,
        end_time=end_time,
        window_size=window_size,
        instance_id=instance_id,
    )

    print(f"[DEBUG] Data collected. Shape: {data.shape}")
    print(f"[DEBUG] Columns: {list(data.columns)[:5]}... ({len(data.columns)} total)")
    print(f"[DEBUG] collect_predict_data completed\n")

    return data.reset_index(drop=True)


def _create_filter_query(code: str, instance: str, svc_type: str, svr_type: str) -> str:
    """Create Prometheus filter query string."""
    return (
        f'CODE="{code}",INSTANCE="{instance}",'
        f'SVC_TYPE="{svc_type}",SVR_TYPE="{svr_type}"'
    )


def _get_queries(
    target_object: str,
    filter_query: str,
    collector: Collector,
    code: str,
    instance: str,
    svc_type: str,
    svr_type: str,
) -> Dict[str, str]:
    """Get appropriate queries based on target object."""
    if target_object == "cpu":
        cpu_time_labels = collector.get_metric_labels(
            "system_cpu_time_seconds_total", code, instance, svc_type, svr_type
        )
        states = cpu_time_labels.get("state", [])

        if not states:
            logger.error("Could not find 'state' labels for CPU metrics")

        return make_cpu_queries(states, filter_query)

    elif target_object == "memory":
        return make_memory_queries(filter_query)

    elif target_object == "disk":
        return make_disk_queries(filter_query)

    else:
        raise ValueError(f"Unknown target object: {target_object}")


def _create_data_filepath(
    output_dir: str,
    target_object: str,
    code: str,
    svc_type: str,
    svr_type: str,
    end_time: datetime,
    instance: Optional[str] = None,
    is_training: bool = True,
) -> str:
    """Generate standardized data file path."""
    data_type = "train" if is_training else "predict"
    base_path = f"{output_dir}/{data_type}/{target_object}"
    os.makedirs(base_path, exist_ok=True)

    if is_training:
        filename = (
            f"{code}_{svc_type}_{svr_type}_{end_time.year}_{end_time.month}_data.csv"
        )
    else:
        filename = f"{code}_{svc_type}_{svr_type}_{instance}_data.csv"

    return os.path.join(base_path, filename)


def _collect_data(
    collector: Collector,
    target_object: str,
    filter_query: str,
    code: str,
    instance: str,
    svc_type: str,
    svr_type: str,
    start_time: datetime,
    end_time: datetime,
    window_size: int,
    instance_id: int,
    output_dir: str = "./data",
) -> pd.DataFrame:
    """Core data collection logic."""
    # Get queries
    queries = _get_queries(
        target_object, filter_query, collector, code, instance, svc_type, svr_type
    )

    # Determine if this is training or prediction
    is_training = window_size > 1

    # Generate file path
    file_path = _create_data_filepath(
        output_dir,
        target_object,
        code,
        svc_type,
        svr_type,
        end_time,
        instance,
        is_training,
    )

    # Handle training data collection
    if is_training:
        return _collect_training_data(
            file_path,
            queries,
            collector,
            start_time,
            end_time,
            window_size,
            instance_id,
            target_object,
        )

    # Handle prediction data collection
    return _collect_prediction_data(
        file_path, queries, collector, start_time, end_time, instance_id, target_object
    )


def _collect_training_data(
    file_path: str,
    queries: Dict[str, str],
    collector: Collector,
    start_time: datetime,
    end_time: datetime,
    window_size: int,
    instance_id: int,
    target_object: str,
) -> pd.DataFrame:
    """Collect and save training data."""
    # Check if data already exists
    if os.path.exists(file_path):
        logger.info(f"Loading existing training data from {file_path}")
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # Collect new data day by day
    all_data = {}
    current_start = start_time
    day_count = 0
    total_points = 0

    while current_start < end_time:
        day_count += 1
        current_end = min(current_start + timedelta(days=1), end_time)

        logger.info(f"{'='*70}")
        logger.info(
            f"[Day {day_count}/{window_size}] "
            f"{current_start.strftime('%Y-%m-%d %H:%M')} ~ "
            f"{current_end.strftime('%Y-%m-%d %H:%M')}"
        )
        logger.info(f"{'='*70}")

        all_data, day_points = _collect_data_points(
            queries, current_start, current_end, all_data, collector
        )
        total_points += day_points

        logger.info(f"\n  Day {day_count} total: {day_points:,} data points")
        logger.info(f"  Cumulative total: {total_points:,} data points")

        current_start = current_end
        time.sleep(DAY_WAIT_TIME)

    # Convert to DataFrame and process
    df = _convert_to_dataframe(all_data)
    df = _add_instance_column(df, instance_id)
    df = _apply_feature_engineering(df, target_object)

    # Save to file
    df.to_csv(file_path, index=False)
    logger.info(f"Saved training data to {file_path}")

    return df


# def _collect_prediction_data(
#     file_path: str,
#     queries: Dict[str, str],
#     collector: Collector,
#     start_time: datetime,
#     end_time: datetime,
#     instance_id: int,
#     target_object: str,
# ) -> pd.DataFrame:
#     """Collect and update prediction data."""
#     print("do this")
#     if not os.path.exists(file_path):
#         # No existing data - collect initial window
#         return _collect_initial_prediction_data(
#             file_path, queries, collector, end_time, instance_id, target_object
#         )

#     # Load existing data
#     existing_df = pd.read_csv(file_path)
#     if existing_df.shape[0] == 0:
#         # No existing data - collect initial window
#         return _collect_initial_prediction_data(
#             file_path, queries, collector, end_time, instance_id, target_object
#         )

#     existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
#     max_time = existing_df["timestamp"].max()

#     time_diff = end_time - max_time
#     # Check if we need to update
#     if time_diff > timedelta(minutes=2):
#         logger.info("Time gap too large. Collecting new 120-minute window.")
#         return _recollect_prediction_window(
#             file_path, queries, collector, end_time, instance_id, target_object
#         )

#     if timedelta(minutes=1) < time_diff <= timedelta(minutes=2):
#         logger.info("Updating with latest data point.")
#         return _update_prediction_data(
#             file_path,
#             existing_df,
#             queries,
#             collector,
#             end_time,
#             instance_id,
#             target_object,
#         )

#     logger.info("Data is already up to date.")
#     return existing_df


def _collect_prediction_data(
    file_path: str,
    queries: Dict[str, str],
    collector: Collector,
    start_time: datetime,
    end_time: datetime,
    instance_id: int,
    target_object: str,
) -> pd.DataFrame:
    """Collect and update prediction data."""
    print(f"[DEBUG] _collect_prediction_data called")
    print(f"[DEBUG] File path: {file_path}")
    print(f"[DEBUG] File exists: {os.path.exists(file_path)}")

    return _collect_initial_prediction_data(
        file_path, queries, collector, end_time, instance_id, target_object
    )

    # # Load existing data
    # print(f"[DEBUG] Loading existing data from {file_path}")
    # existing_df = pd.read_csv(file_path)

    # if existing_df.shape[0] == 0:
    #     print(f"[DEBUG] file is empty. Collecting initial 61-minute window...")
    #     # No existing data - collect initial window
    #     return _collect_initial_prediction_data(
    #         file_path, queries, collector, end_time, instance_id, target_object
    #     )

    # existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
    # max_time = existing_df["timestamp"].max()

    # time_diff = end_time - max_time

    # print(f"[DEBUG] Existing data info:")
    # print(f"  - Last timestamp: {max_time}")
    # print(f"  - Current time: {end_time}")
    # print(f"  - Time difference: {time_diff}")
    # print(f"  - Existing rows: {len(existing_df)}")

    # # Check if we need to update
    # if time_diff > timedelta(minutes=2):
    #     print(f"[DEBUG] Time gap > 2 minutes. Recollecting 120-minute window...")
    #     logger.info("Time gap too large. Collecting new 120-minute window.")
    #     return _recollect_prediction_window(
    #         file_path, queries, collector, end_time, instance_id, target_object
    #     )

    # if timedelta(minutes=1) < time_diff <= timedelta(minutes=2):
    #     print(
    #         f"[DEBUG] Time gap between 1-2 minutes. Updating with latest data point..."
    #     )
    #     logger.info("Updating with latest data point.")
    #     return _update_prediction_data(
    #         file_path,
    #         existing_df,
    #         queries,
    #         collector,
    #         end_time,
    #         instance_id,
    #         target_object,
    #     )

    # print(f"[DEBUG] Data is already up to date. Using existing data.")
    # logger.info("Data is already up to date.")
    # return existing_df


# def _collect_initial_prediction_data(
#     file_path: str,
#     queries: Dict[str, str],
#     collector: Collector,
#     end_time: datetime,
#     instance_id: int,
#     target_object: str,
# ) -> pd.DataFrame:
#     """Collect initial prediction window (61 minutes)."""
#     start_time = end_time - timedelta(minutes=60)
#     all_data = {}

#     all_data, _ = _collect_data_points(
#         queries, start_time, end_time, all_data, collector
#     )

#     df = _convert_to_dataframe(all_data)
#     df = _add_instance_column(df, instance_id)
#     df = _apply_feature_engineering(df, target_object)

#     df.to_csv(file_path, index=False)
#     return df


def _collect_initial_prediction_data(
    file_path: str,
    queries: Dict[str, str],
    collector: Collector,
    end_time: datetime,
    instance_id: int,
    target_object: str,
) -> pd.DataFrame:
    """Collect initial prediction window (61 minutes)."""
    print(f"[DEBUG] _collect_initial_prediction_data called")

    start_time = end_time - timedelta(minutes=60)
    print(f"[DEBUG] Collecting initial window: {start_time} ~ {end_time}")
    print(f"[DEBUG] Number of queries: {len(queries)}")

    all_data = {}

    all_data, points_collected = _collect_data_points(
        queries, start_time, end_time, all_data, collector
    )

    print(f"[DEBUG] Data points collected: {points_collected}")
    print(f"[DEBUG] Unique timestamps: {len(all_data)}")

    df = _convert_to_dataframe(all_data)
    print(f"[DEBUG] DataFrame shape after conversion: {df.shape}")

    df = df.set_index("timestamp")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    df = df.reindex(full_range)
    df.index.name = "timestamp"
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.fillna(0)

    df = _add_instance_column(df, instance_id)
    print(f"[DEBUG] Instance column added. Instance ID: {instance_id}")

    df = _apply_feature_engineering(df, target_object)
    print(f"[DEBUG] Feature engineering applied. Final shape: {df.shape}")

    df.to_csv(file_path, index=False)
    print(f"[DEBUG] Data saved to: {file_path}")
    print(f"[DEBUG] _collect_initial_prediction_data completed\n")

    return df


# def _recollect_prediction_window(
#     file_path: str,
#     queries: Dict[str, str],
#     collector: Collector,
#     end_time: datetime,
#     instance_id: int,
#     target_object: str,
# ) -> pd.DataFrame:
#     """Recollect prediction window when gap is too large."""
#     start_time = end_time - timedelta(minutes=PREDICTION_WINDOW_MINUTES)
#     all_data = {}

#     all_data, _ = _collect_data_points(
#         queries, start_time, end_time, all_data, collector
#     )

#     df = _convert_to_dataframe(all_data)
#     df = df.dropna()
#     df = df.sort_values("timestamp").reset_index(drop=True)
#     df = df.tail(MIN_PREDICTION_WINDOW)

#     df = _add_instance_column(df, instance_id)
#     df = _apply_feature_engineering(df, target_object)

#     df.to_csv(file_path, index=False)
#     return df


def _recollect_prediction_window(
    file_path: str,
    queries: Dict[str, str],
    collector: Collector,
    end_time: datetime,
    instance_id: int,
    target_object: str,
) -> pd.DataFrame:
    """Recollect prediction window when gap is too large."""
    print(f"[DEBUG] _recollect_prediction_window called")

    start_time = end_time - timedelta(minutes=PREDICTION_WINDOW_MINUTES)
    print(f"[DEBUG] Recollecting window: {start_time} ~ {end_time} (120 minutes)")

    all_data = {}

    all_data, points_collected = _collect_data_points(
        queries, start_time, end_time, all_data, collector
    )

    print(f"[DEBUG] Data points collected: {points_collected}")
    print(f"[DEBUG] Unique timestamps: {len(all_data)}")

    df = _convert_to_dataframe(all_data)
    print(f"[DEBUG] DataFrame shape: {df.shape}")

    df = df.dropna()
    print(f"[DEBUG] After dropna: {df.shape}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.tail(MIN_PREDICTION_WINDOW)
    print(f"[DEBUG] After taking last 61 rows: {df.shape}")

    df = _add_instance_column(df, instance_id)
    df = _apply_feature_engineering(df, target_object)
    print(f"[DEBUG] After feature engineering: {df.shape}")

    df.to_csv(file_path, index=False)
    print(f"[DEBUG] Data saved to: {file_path}")
    print(f"[DEBUG] _recollect_prediction_window completed\n")

    return df


def _update_prediction_data(
    file_path: str,
    existing_df: pd.DataFrame,
    queries: Dict[str, str],
    collector: Collector,
    end_time: datetime,
    instance_id: int,
    target_object: str,
) -> pd.DataFrame:
    """Update prediction data with new data point."""
    all_data = {}
    all_data, _ = _collect_data_points(queries, end_time, end_time, all_data, collector)

    if not all_data:
        logger.warning("Failed to collect new data point")
        return existing_df

    # Create new data point
    new_df = _convert_to_dataframe(all_data)
    new_df = _add_instance_column(new_df, instance_id)
    new_df = _apply_feature_engineering(new_df, target_object)

    # Remove oldest data point from existing data
    filtered_existing = existing_df[
        existing_df["timestamp"] != existing_df["timestamp"].min()
    ]

    # Merge old and new data
    merged_df = pd.merge(
        filtered_existing,
        new_df,
        on=list(set(filtered_existing.columns) & set(new_df.columns)),
        how="outer",
    )
    merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

    merged_df.to_csv(file_path, index=False)
    return merged_df


def _collect_data_points(
    queries: Dict[str, str],
    start_time: datetime,
    end_time: datetime,
    all_data: Dict,
    collector: Collector,
) -> Tuple[Dict, int]:
    """Collect data points for given time range."""
    day_points = 0

    for metric_name, query in queries.items():
        logger.info(f"  Collecting: {metric_name}")

        result = collector.query_range(query, start_time, end_time, step="1m")

        if result and result.get("status") == "success":
            data = result.get("data", {}).get("result", [])

            if data:
                values = data[0].get("values", [])

                for timestamp, value in values:
                    ts = datetime.fromtimestamp(timestamp)

                    if ts not in all_data:
                        all_data[ts] = {}

                    all_data[ts][metric_name] = float(value)

                day_points += len(values)
                logger.info(f"    ✓ {len(values)} data points collected")
        else:
            logger.warning(f"    ✗ Query failed")

        time.sleep(API_WAIT_TIME)

    return all_data, day_points


def _convert_to_dataframe(data: Dict) -> pd.DataFrame:
    """Convert collected data dictionary to DataFrame."""
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "timestamp"
    df = df.reset_index()
    df = df.sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.floor("s")
    return df


def _add_instance_column(df: pd.DataFrame, instance_id: int) -> pd.DataFrame:
    """Add instance ID column to DataFrame."""
    df.insert(1, "instance", instance_id)
    df["instance"] = df["instance"].astype(int)  # Fixed typo: was "intance"
    return df


def _apply_feature_engineering(df: pd.DataFrame, target_object: str) -> pd.DataFrame:
    """Apply feature engineering based on target object."""
    feature_functions = {
        "cpu": lambda df: create_cpu_features(df, 85),
        "memory": lambda df: create_memory_features(df, 85),
        "disk": lambda df: create_disk_features(df, 90),
    }

    if target_object not in feature_functions:
        logger.error(f"Unknown target object: {target_object}")
        raise ValueError(f"Unknown target object: {target_object}")

    return feature_functions[target_object](df)
