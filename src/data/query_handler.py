"""
Query handler module for generating Prometheus queries and feature engineering.
"""

from typing import Dict, List
import pandas as pd


# Constants for feature engineering
DEFAULT_LAG_PERIODS = [1, 5, 10, 15, 30, 45, 60]
DEFAULT_ROLLING_WINDOWS = [10, 30, 60]
DEFAULT_DIFF_PERIODS = [1, 10, 30, 60]


def make_cpu_queries(states: List[str], filter_query: str) -> Dict[str, str]:
    """
    Generate CPU-related Prometheus queries.

    Args:
        states: List of CPU states (idle, user, system, etc.)
        filter_query: Filter conditions for the query

    Returns:
        Dictionary mapping metric names to PromQL queries
    """
    queries = {}

    # CPU Utilization
    if "idle" in states:
        queries["cpu_utilization"] = (
            f"100 - (avg(irate(system_cpu_time_seconds_total{{"
            f'state="idle",{filter_query}}}[1m])) * 100)'
        )

    # CPU time per state
    for state in states:
        queries[f"cpu_time_{state}"] = (
            f"irate(system_cpu_time_seconds_total{{"
            f'state="{state}",{filter_query}}}[1m])'
        )

    # Load Averages
    queries["load_avg_1m"] = f"system_cpu_load_average_1m{{{filter_query}}}"
    queries["load_avg_5m"] = f"system_cpu_load_average_5m{{{filter_query}}}"
    queries["load_avg_15m"] = f"system_cpu_load_average_15m{{{filter_query}}}"

    return queries


def make_memory_queries(filter_query: str) -> Dict[str, str]:
    """
    Generate Memory-related Prometheus queries.

    Args:
        filter_query: Filter conditions for the query

    Returns:
        Dictionary mapping metric names to PromQL queries
    """
    queries = {}

    # Memory usage by state
    queries["memory_used_bytes"] = (
        f'system_memory_usage_bytes{{state="used",{filter_query}}}'
    )
    queries["memory_free_bytes"] = (
        f'system_memory_usage_bytes{{state="free",{filter_query}}}'
    )

    # Total memory
    queries["memory_total_bytes"] = f"sum(system_memory_usage_bytes{{{filter_query}}})"

    # Memory usage rate of change
    queries["memory_used_rate"] = (
        f'irate(system_memory_usage_bytes{{state="used",{filter_query}}}[1m])'
    )

    return queries


def make_disk_queries(filter_query: str) -> Dict[str, str]:
    """
    Generate Disk-related Prometheus queries.

    Args:
        filter_query: Filter conditions for the query

    Returns:
        Dictionary mapping metric names to PromQL queries
    """
    queries = {}

    # Filesystem usage by state
    states = ["used", "free", "reserved"]
    for state in states:
        queries[f"disk_{state}_bytes"] = (
            f'system_filesystem_usage_bytes{{state="{state}",{filter_query}}}'
        )

    # Total disk capacity
    queries["disk_total_bytes"] = (
        f"sum(system_filesystem_usage_bytes{{{filter_query}}}) "
        f"by (device, mountpoint)"
    )

    # Disk utilization percentage
    queries["disk_utilization"] = (
        f'(system_filesystem_usage_bytes{{state="used",{filter_query}}} / '
        f"sum(system_filesystem_usage_bytes{{{filter_query}}}) "
        f"by (device, mountpoint)) * 100"
    )

    # Free space ratio
    queries["disk_free_ratio"] = (
        f'(system_filesystem_usage_bytes{{state="free",{filter_query}}} / '
        f"sum(system_filesystem_usage_bytes{{{filter_query}}}) "
        f"by (device, mountpoint)) * 100"
    )

    # I/O Bytes rates
    queries["disk_read_bytes_rate"] = (
        f'irate(system_disk_io_bytes_total{{direction="read",{filter_query}}}[1m])'
    )
    queries["disk_write_bytes_rate"] = (
        f'irate(system_disk_io_bytes_total{{direction="write",{filter_query}}}[1m])'
    )
    queries["disk_total_io_bytes_rate"] = (
        f"sum(irate(system_disk_io_bytes_total{{{filter_query}}}[1m])) " f"by (device)"
    )

    # I/O Operations rates
    queries["disk_read_ops_rate"] = (
        f'irate(system_disk_operations_total{{direction="read",{filter_query}}}[1m])'
    )
    queries["disk_write_ops_rate"] = (
        f'irate(system_disk_operations_total{{direction="write",{filter_query}}}[1m])'
    )

    # I/O Time metrics
    queries["disk_io_time_rate"] = (
        f"irate(system_disk_io_time_seconds_total{{{filter_query}}}[1m])"
    )
    queries["disk_operation_time_rate"] = (
        f"irate(system_disk_operation_time_seconds_total{{{filter_query}}}[1m])"
    )

    # Pending operations
    queries["disk_pending_operations"] = (
        f"system_disk_pending_operations{{{filter_query}}}"
    )

    return queries


def create_cpu_features(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Create CPU-related features for model training.

    Args:
        df: Input DataFrame with raw CPU metrics
        threshold: Threshold value for creating target variable (e.g., 85%)

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Starting feature engineering...")

    # 1. Time-based features
    df = _add_time_features(df)

    # 2. Lag features
    for lag in DEFAULT_LAG_PERIODS:
        df[f"cpu_util_lag_{lag}m"] = df["cpu_utilization"].shift(lag)

    # 3. Rolling statistics
    for window in DEFAULT_ROLLING_WINDOWS:
        df[f"cpu_util_mean_{window}m"] = df["cpu_utilization"].rolling(window).mean()
        df[f"cpu_util_std_{window}m"] = df["cpu_utilization"].rolling(window).std()
        df[f"cpu_util_max_{window}m"] = df["cpu_utilization"].rolling(window).max()
        df[f"cpu_util_min_{window}m"] = df["cpu_utilization"].rolling(window).min()

    # 4. Change rates
    for diff in DEFAULT_DIFF_PERIODS:
        df[f"cpu_util_change_{diff}m"] = df["cpu_utilization"].diff(diff)

    # 5. Load Average ratios
    if "load_avg_1m" in df.columns and "load_avg_5m" in df.columns:
        df["load_ratio_1m_5m"] = df["load_avg_1m"] / (df["load_avg_5m"] + 1e-6)
        df["load_ratio_1m_15m"] = df["load_avg_1m"] / (df["load_avg_15m"] + 1e-6)

    # 6. CPU time ratios
    if "cpu_time_user" in df.columns and "cpu_time_system" in df.columns:
        df["cpu_user_system_ratio"] = df["cpu_time_user"] / (
            df["cpu_time_system"] + 1e-6
        )

    # 7. Target variable
    df = _create_target_variable(df, "cpu_utilization", threshold, 60)
    # Remove missing values
    original_len = len(df)
    df = df.dropna()
    print(f"✓ Original df: {original_len} rows (removed: {original_len - len(df)})\n")

    return df


def create_memory_features(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Create Memory-related features for model training.

    Args:
        df: Input DataFrame with raw memory metrics
        threshold: Threshold value for creating target variable (e.g., 85%)

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Starting feature engineering...")

    # 1. Time-based features
    df = _add_time_features(df)

    # 2. Basic calculations
    df["memory_utilization"] = (
        df["memory_used_bytes"] / df["memory_total_bytes"]
    ) * 100
    df["memory_free_ratio"] = (df["memory_free_bytes"] / df["memory_total_bytes"]) * 100
    df["memory_used_free_ratio"] = df["memory_used_bytes"] / (
        df["memory_free_bytes"] + 1e-9
    )

    # 3. Lag features
    for lag in [1, 5, 10, 30, 60]:
        df[f"memory_util_lag_{lag}m"] = df["memory_utilization"].shift(lag)
        df[f"memory_used_lag_{lag}m"] = df["memory_used_bytes"].shift(lag)

    # 4. Rolling statistics
    for window in DEFAULT_ROLLING_WINDOWS:
        df[f"memory_util_mean_{window}m"] = (
            df["memory_utilization"].rolling(window).mean()
        )
        df[f"memory_util_std_{window}m"] = (
            df["memory_utilization"].rolling(window).std()
        )
        df[f"memory_util_max_{window}m"] = (
            df["memory_utilization"].rolling(window).max()
        )
        df[f"memory_util_min_{window}m"] = (
            df["memory_utilization"].rolling(window).min()
        )

    # 5. Change rates
    for diff in DEFAULT_DIFF_PERIODS:
        df[f"memory_util_change_{diff}m"] = df["memory_utilization"].diff(diff)
        df[f"memory_used_change_{diff}m"] = df["memory_used_bytes"].diff(diff)

    # 6. Trend features
    df["memory_vs_mean_30m"] = df["memory_utilization"] - df["memory_util_mean_30m"]
    df["memory_trend_30m"] = df["memory_utilization"] - df["memory_util_lag_30m"]
    df["memory_acceleration"] = df["memory_util_change_1m"].diff(1)

    # 7. Volatility features
    df["memory_range_10m"] = df["memory_util_max_10m"] - df["memory_util_min_10m"]
    df["memory_range_30m"] = df["memory_util_max_30m"] - df["memory_util_min_30m"]
    df["memory_cv_30m"] = df["memory_util_std_30m"] / (
        df["memory_util_mean_30m"] + 1e-9
    )

    # 8. Threshold-related features
    df["memory_distance_to_threshold"] = threshold - df["memory_utilization"]
    df["memory_threshold_proximity"] = df["memory_utilization"] / threshold

    # 9. Rate-based features
    df["memory_rate_change_1m"] = df["memory_used_rate"].diff(1)
    df["memory_rate_mean_10m"] = df["memory_used_rate"].rolling(10).mean()

    # 10. Target variable
    df = _create_target_variable(df, "memory_utilization", threshold, 60)

    # Remove missing values
    original_len = len(df)
    df = df.dropna()
    print(f"✓ Removed NaN: {len(df):,} rows (removed: {original_len - len(df)})\n")

    return df


def create_disk_features(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Create Disk-related features for model training.

    Args:
        df: Input DataFrame with raw disk metrics
        threshold: Threshold value for creating target variable (e.g., 90%)

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Starting feature engineering...")

    # 1. Time-based features
    df = _add_time_features(df)

    # 2. Basic features
    df["disk_utilization"] = (df["disk_used_bytes"] / df["disk_total_bytes"]) * 100
    df["disk_free_gb"] = df["disk_free_bytes"] / (1024**3)
    df["disk_used_gb"] = df["disk_used_bytes"] / (1024**3)
    df["disk_total_gb"] = df["disk_total_bytes"] / (1024**3)
    df["disk_free_ratio"] = (df["disk_free_bytes"] / df["disk_total_bytes"]) * 100

    # 3. Growth rate features
    df["disk_growth_10m"] = df["disk_used_bytes"].diff(10)
    df["disk_growth_60m"] = df["disk_used_bytes"].diff(60)
    df["disk_growth_rate_60m"] = df["disk_growth_60m"] / 60

    # 4. Trend features
    df["disk_trend_30m"] = df["disk_utilization"] - df["disk_utilization"].shift(30)
    df["disk_trend_60m"] = df["disk_utilization"] - df["disk_utilization"].shift(60)
    df["disk_acceleration"] = df["disk_growth_10m"].diff(10)

    # 5. Rolling averages
    df["disk_util_mean_30m"] = df["disk_utilization"].rolling(30).mean()
    df["disk_util_mean_60m"] = df["disk_utilization"].rolling(60).mean()

    # 6. Threshold-related features
    df["disk_distance_to_threshold"] = threshold - df["disk_utilization"]
    df["disk_low_space"] = (df["disk_free_gb"] < 10).astype(int)

    # 7. I/O features
    df["disk_total_io_rate"] = df["disk_read_bytes_rate"] + df["disk_write_bytes_rate"]
    df["disk_io_mean_10m"] = df["disk_total_io_rate"].rolling(10).mean()
    df["disk_io_bottleneck"] = (
        df["disk_io_time_rate"] > df["disk_io_time_rate"].rolling(10).mean() * 2
    ).astype(int)

    # 8. Target variable
    df = _create_target_variable(df, "disk_utilization", threshold, 60)

    # Remove missing values
    original_len = len(df)
    df = df.dropna()
    print(f"✓ Removed NaN: {len(df):,} rows (removed: {original_len - len(df)})\n")

    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to DataFrame."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hour"] = df["hour"].between(9, 18).astype(int)
    return df


def _create_target_variable(
    df: pd.DataFrame, utilization_col: str, threshold: int, prediction_window: int
) -> pd.DataFrame:
    """Create binary target variable based on future utilization."""
    future_col = f"{utilization_col.split('_')[0]}_util_future"
    df[future_col] = df[utilization_col].shift(-prediction_window)
    df["target"] = (df[future_col] >= threshold).astype(int)
    df = df.drop(columns=[future_col])
    return df
