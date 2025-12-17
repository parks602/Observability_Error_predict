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
        game_type: str,
        instance: str,
        svc_type: str,
        svr_type: str,
        world: str,
    ) -> Dict[str, List[str]]:
        """Query metric labels from Prometheus."""
        if pd.notna(world):
            query = (
                f"{metric_name}{{"
                f'CODE="{code}",GAME_TYPE="{game_type}",INSTANCE="{instance}",'
                f'SVC_TYPE="{svc_type}",SVR_TYPE="{svr_type}",WORLD="{world}"}}'
            )
        else:
            query = (
                f"{metric_name}{{"
                f'CODE="{code}",GAME_TYPE="{game_type}",INSTANCE="{instance}",'
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
                    "GAME_TYPE",
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
    server_map: pd.DataFrame,
    prometheus_url: str,
    end_time: datetime,
    window_size: int,
    code: str,
    game_type: str,
    svc_type: str,
    svr_type: str,
    world: str,
) -> pd.DataFrame:
    """Collect training data for multiple instances."""
    start_time = end_time - timedelta(days=window_size)
    collector = Collector(prometheus_url)

    all_data = []
    filtered_df = server_map[
        (server_map["CODE"] == code)
        & (server_map["GAME_TYPE"] == game_type)
        & (server_map["SVC_TYPE"] == svc_type)
    ]
    for _, unique_row in filtered_df.iterrows():
        svr_type = unique_row["SVR_TYPE"]
        svr_id = unique_row["svr_id"]
        world = unique_row["WORLD"]
        word_id = unique_row["world_id"]
        instance = unique_row["INSTANCE"]
        instance_id = unique_row["instance_id"]

        filter_query = _create_filter_query(
            code, game_type, instance, svc_type, svr_type, world
        )
        data = _collect_data(
            collector=collector,
            target_object=target_object,
            filter_query=filter_query,
            code=code,
            game_type=game_type,
            instance=instance,
            svc_type=svc_type,
            svr_type=svr_type,
            world=world,
            start_time=start_time,
            end_time=end_time,
            window_size=window_size,
            svr_id=svr_id,
            world_id=word_id,
            instance_id=instance_id,
        )
        all_data.append(data)

    return pd.concat(all_data, axis=0, ignore_index=True)


def collect_predict_data(
    target_object: str,
    instance: str,
    instance_id: int,
    prometheus_url: str,
    end_time: datetime,
    window_size: int,
    code: str,
    game_type: str,
    svc_type: str,
    svr_type: str,
    world: str,
    svr_id: int,
    world_id: int,
) -> pd.DataFrame:
    """Collect prediction data for a single instance."""

    start_time = end_time - timedelta(days=window_size)
    collector = Collector(prometheus_url)
    filter_query = _create_filter_query(
        code, game_type, instance, svc_type, svr_type, world
    )

    data = _collect_data(
        collector=collector,
        target_object=target_object,
        filter_query=filter_query,
        code=code,
        game_type=game_type,
        instance=instance,
        svc_type=svc_type,
        svr_type=svr_type,
        world=world,
        start_time=start_time,
        end_time=end_time,
        window_size=window_size,
        svr_id=svr_id,
        world_id=world_id,
        instance_id=instance_id,
    )

    return data.reset_index(drop=True)


def _create_filter_query(
    code: str, game_type: str, instance: str, svc_type: str, svr_type: str, world: str
) -> str:
    """Create Prometheus filter query string."""
    if not pd.isna(world):
        return (
            f'CODE="{code}",GAME_TYPE="{game_type}",INSTANCE="{instance}",'
            f'SVC_TYPE="{svc_type}",SVR_TYPE="{svr_type}",WORLD="{world}"'
        )
    else:
        return (
            f'CODE="{code}",GAME_TYPE="{game_type}",INSTANCE="{instance}",'
            f'SVC_TYPE="{svc_type}",SVR_TYPE="{svr_type}"'
        )


def _get_queries(
    target_object: str,
    filter_query: str,
    collector: Collector,
    code: str,
    game_type: str,
    instance: str,
    svc_type: str,
    svr_type: str,
    world: str,
) -> Dict[str, str]:
    """Get appropriate queries based on target object."""
    if target_object == "cpu":
        cpu_time_labels = collector.get_metric_labels(
            "system_cpu_time_seconds_total",
            code,
            game_type,
            instance,
            svc_type,
            svr_type,
            world,
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
    game_type: str,
    svc_type: str,
    end_time: datetime,
    is_training: bool = True,
) -> str:
    """Generate standardized data file path."""
    data_type = "train" if is_training else "predict"
    base_path = f"{output_dir}/{data_type}/{target_object}"
    os.makedirs(base_path, exist_ok=True)

    if is_training:
        filename = (
            f"{code}_{game_type}_{svc_type}_{end_time.year}_{end_time.month}_data.csv"
        )
    else:
        filename = f"{code}_{game_type}_{svc_type}_data.csv"

    return os.path.join(base_path, filename)


def _collect_data(
    collector: Collector,
    target_object: str,
    filter_query: str,
    code: str,
    game_type: str,
    instance: str,
    svc_type: str,
    svr_type: str,
    world: str,
    start_time: datetime,
    end_time: datetime,
    window_size: int,
    svr_id: int,
    world_id: int,
    instance_id: int,
) -> pd.DataFrame:
    """Core data collection logic."""
    # Get queries
    queries = _get_queries(
        target_object,
        filter_query,
        collector,
        code,
        game_type,
        instance,
        svc_type,
        svr_type,
        world,
    )

    # Determine if this is training or prediction
    is_training = window_size > 1

    # Handle training data collection
    if is_training:
        return _collect_training_data(
            queries,
            collector,
            start_time,
            end_time,
            window_size,
            svr_id,
            world_id,
            instance_id,
            target_object,
        )

    # Handle prediction data collection
    return _collect_prediction_data(
        queries,
        collector,
        start_time,
        end_time,
        instance_id,
        target_object,
        svr_id,
        world_id,
    )


def _collect_training_data(
    queries: Dict[str, str],
    collector: Collector,
    start_time: datetime,
    end_time: datetime,
    window_size: int,
    svr_id: int,
    world_id: int,
    instance_id: int,
    target_object: str,
) -> pd.DataFrame:
    """Collect and save training data."""

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
    df = _add_instance_column(df, svr_id, world_id, instance_id)
    df = _apply_feature_engineering(df, target_object)

    return df


def _collect_prediction_data(
    queries: Dict[str, str],
    collector: Collector,
    start_time: datetime,
    end_time: datetime,
    instance_id: int,
    target_object: str,
    svr_id: int,
    world_id: int,
) -> pd.DataFrame:
    """Collect and update prediction data."""

    return _collect_initial_prediction_data(
        queries,
        collector,
        end_time,
        instance_id,
        target_object,
        svr_id,
        world_id,
    )


def _collect_initial_prediction_data(
    queries: Dict[str, str],
    collector: Collector,
    end_time: datetime,
    instance_id: int,
    target_object: str,
    svr_id: int,
    world_id: int,
) -> pd.DataFrame:
    """Collect initial prediction window (61 minutes)."""

    start_time = end_time - timedelta(minutes=60)

    all_data = {}

    all_data, points_collected = _collect_data_points(
        queries, start_time, end_time, all_data, collector
    )

    df = _convert_to_dataframe(all_data)
    df = df.set_index("timestamp")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    df = df.reindex(full_range)
    df.index.name = "timestamp"
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.fillna(0)

    df = _add_instance_column(df, svr_id, world_id, instance_id)

    df = _apply_feature_engineering(df, target_object)

    return df


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

    df = _add_instance_column(df, svr_id, world_id, instance_id)
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
    new_df = _add_instance_column(df, svr_id, world_id, instance_id)
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].dt.floor("s")
    return df


def _add_instance_column(
    df: pd.DataFrame, svr_id: int, world_id: int, instance_id: int
) -> pd.DataFrame:
    """Add instance-related ID columns to DataFrame."""
    df.insert(1, "svr_id", int(svr_id))
    df.insert(2, "world_id", int(world_id))
    df.insert(3, "instance", int(instance_id))
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
