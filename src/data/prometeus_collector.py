import logging
from datetime import datetime, timedelta
import time
import os
import pandas as pd
import warnings
import requests
from collections import defaultdict
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


class Collector:
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
    ):
        """메트릭의 레이블 값 조회"""
        query = f'{metric_name}{{CODE="{code}",INSTANCE="{instance}",SVC_TYPE="{svc_type}",SVR_TYPE="{svr_type}"}}'
        url = f"{self.api_endpoint}/query"

        try:
            response = requests.get(url, params={"query": query}, timeout=5)
            result = response.json()

            if result.get("status") == "success":
                data = result.get("data", {}).get("result", [])

                labels = {}
                for item in data:
                    metric = item.get("metric", {})
                    for key, value in metric.items():
                        if key not in [
                            "__name__",
                            "CODE",
                            "INSTANCE",
                            "SVC_TYPE",
                            "SVR_TYPE",
                            "job",
                            "instance",
                        ]:
                            if key not in labels:
                                labels[key] = set()
                            labels[key].add(value)

                return {k: sorted(list(v)) for k, v in labels.items()}
        except:
            pass

        return {}

    def query_range(
        self, query: str, start_time: datetime, end_time: datetime, step: str = "1m"
    ):
        url = f"{self.api_endpoint}/query_range"

        params = {
            "query": query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": step,
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"쿼리 실패: {e}")
            return None


def collect_train_data(
    target_object,
    instances,
    instances_ids,
    prometheus_url,
    end_time,
    window_size,
    code,
    svc_type,
    svr_type,
):

    start_time = end_time - timedelta(days=window_size)
    collector = Collector(prometheus_url)
    datas = None
    filter_query = f'CODE="{code}",INSTANCE="{instance}",SVC_TYPE="{svc_type}",SVR_TYPE="{svr_type}"'
    if target_object == "cpu":
        for i, instance in enumerate(instances):

            data = collect_cpu_data(
                collector,
                filter_query,
                code,
                instance,
                svc_type,
                svr_type,
                start_time,
                end_time,
                window_size,
            )
            if datas is None:
                datas = data
            else:
                datas = pd.concat([datas, data], axis=0)
        datas.insert(1, "instance", instances_ids[i])
        datas["intance"] = datas["instance"].astype(int)

        return datas.reset_index(drop=True)
    elif target_object == "memory":
        pass
    elif target_object == "disk":
        pass
    else:
        pass


def collect_predict_data(
    target_object,
    instance,
    instance_id,
    prometheus_url,
    end_time,
    window_size,
    code,
    svc_type,
    svr_type,
):

    start_time = end_time - timedelta(days=window_size)
    collector = Collector(prometheus_url)
    filter_query = f'CODE="{code}",INSTANCE="{instance}",SVC_TYPE="{svc_type}",SVR_TYPE="{svr_type}", INSTANCE="{instance}"'
    if target_object == "cpu":

        data = collect_cpu_data(
            collector,
            filter_query,
            code,
            instance,
            svc_type,
            svr_type,
            start_time,
            end_time,
            window_size,
            instance_id=instance_id,
        )

        return data.reset_index(drop=True)

    elif target_object == "memory":
        pass
    elif target_object == "disk":
        pass
    else:
        pass


def get_data_point_count(queries, current_start, current_end, all_data, collector):
    day_points = 0

    # 각 메트릭 수집
    for metric_name, query in queries.items():
        logger.info(f"  수집 중: {metric_name}")

        result = collector.query_range(query, current_start, current_end, step="1m")

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
                logger.info(f"    ✓ {len(values)} 데이터 포인트 수집")
        else:
            logger.warning(f"    ✗ 쿼리 실패")

        time.sleep(0.2)  # API 부하 방지
    return all_data, day_points


def collect_cpu_data(
    collector: Collector,
    filter_query: str,
    code: str,
    instance: str,
    svc_type: str,
    svr_type: str,
    start_time: datetime,
    end_time: datetime,
    window_size: int = 60,
    output_dir: str = "./data",
    instance_id: str = None,
):

    cpu_time_labels = collector.get_metric_labels(
        "system_cpu_time_seconds_total", code, instance, svc_type, svr_type
    )

    if "state" in cpu_time_labels:
        states = cpu_time_labels["state"]
    else:
        logger.error("collect_cpu_data | state 레이블을 찾을 수 없습니다.")
        states = []

    queries = make_cpu_queries(states, filter_query)
    all_data = {}

    if window_size > 1:
        file_name = f"{output_dir}/train/cpu/{code}_{svc_type}_{svr_type}_{end_time.year}_{end_time.month}_data.csv"
        if not os.path.exists(file_name):
            current_start = start_time
            day_count = 0
            total_points = 0

            while current_start < end_time:
                day_count += 1
                current_end = min(current_start + timedelta(days=1), end_time)

                logger.info(f"\n{'='*70}")
                logger.info(
                    f"[Day {day_count}/{window_size}] {current_start.strftime('%Y-%m-%d %H:%M')} ~ {current_end.strftime('%Y-%m-%d %H:%M')}"
                )
                logger.info(f"{'='*70}")

                all_data, day_points = get_data_point_count(
                    queries, current_start, current_end, all_data, collector
                )
                total_points += day_points

                logger.info(f"\n  Day {day_count} 합계: {day_points:,} 데이터 포인트")
                logger.info(f"  누적 합계: {total_points:,} 데이터 포인트")

                # 다음 날짜로 이동
                current_start = current_end
                time.sleep(1)  # 일 단위 수집 간 대기
            all_data = data_to_df(all_data)
            all_data.to_csv(file_name, index=False)
            return all_data
        else:
            all_data = pd.read_csv(file_name)
            all_data["timestamp"] = pd.to_datetime(all_data["timestamp"])
            return all_data

    elif window_size == 1:
        file_name = (
            f"{output_dir}/predict/cpu/{code}_{svc_type}_{svr_type}_{instance}_data.csv"
        )
        if os.path.exists(file_name):
            existed_df = pd.read_csv(file_name)
            max_time = pd.to_datetime(existed_df["timestamp"].max())
            print(end_time, max_time, end_time - max_time)
            if (end_time - max_time) > timedelta(minutes=2):
                print(
                    "마지막 데이터와 시간차이가 너무 큽니다. 60분 데이터를 새로 수집합니다."
                )
                start_time = end_time - timedelta(minutes=120)
                result, _ = get_data_point_count(
                    queries, start_time, end_time, all_data, collector
                )
                result_df = data_to_df(result)
                result_df = result_df.dropna()
                result_df = result_df.sort_values("timestamp").reset_index(drop=True)
                result_df = result_df.tail(61)
                result_df.insert(1, "instance", instance_id)
                result_df["intance"] = result_df["instance"].astype(int)
                result_df.to_csv(
                    file_name,
                    index=False,
                )
                return result_df
            if timedelta(minutes=1) < (end_time - max_time) < timedelta(minutes=2):
                result, _ = get_data_point_count(
                    queries, end_time, end_time, all_data, collector
                )
                if result:
                    result_df = data_to_df(result)
                    filtered_existed = existed_df[
                        existed_df["timestamp"] != existed_df["timestamp"].min()
                    ]
                    filtered_existed["timestamp"] = pd.to_datetime(
                        filtered_existed["timestamp"]
                    )

                    result_df["timestamp"] = pd.to_datetime(result_df["timestamp"])
                    result_df.insert(1, "instance", instance_id)
                    result_df["intance"] = result_df["instance"].astype(int)
                    merged_df = pd.merge(
                        filtered_existed,
                        result_df,
                        on=list(set(filtered_existed.columns) & set(result_df.columns)),
                        how="outer",
                    )
                    merged_df = merged_df.sort_values("timestamp").reset_index(
                        drop=True
                    )
                    merged_df.to_csv(
                        file_name,
                        index=False,
                    )
                    return merged_df
                else:
                    logger.warning(f"    ✗ 쿼리 실패")
                    return existed_df
            else:
                print("이미 최신 데이터가 존재합니다.")

                return existed_df
        else:
            # 61 minute data collection
            start_time = end_time - timedelta(minutes=60)
            result, _ = get_data_point_count(
                queries, start_time, end_time, all_data, collector
            )
            result_df = data_to_df(result)
            result_df.insert(1, "instance", instance_id)
            result_df["intance"] = result_df["instance"].astype(int)
            result_df.to_csv(
                file_name,
                index=False,
            )
            return result_df


def collect_memory_data(
    collector: Collector,
    filter_query: str,
    code: str,
    instance: str,
    svc_type: str,
    svr_type: str,
    start_time: datetime,
    end_time: datetime,
    window_size: int = 60,
    output_dir: str = "./data",
    instance_id: str = None,
):

    memory_time_labels = collector.get_metric_labels(
        "system_memory_available_bytes", code, instance, svc_type, svr_type
    )

    if "state" in memory_time_labels:
        states = memory_time_labels["state"]
    else:
        logger.error("collect_memory_data | state 레이블을 찾을 수 없습니다.")
        states = []

    queries = make_memory_queries(filter_query)
    all_data = {}


def collect_disk_data():
    pass


def data_to_df(data):
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "timestamp"
    df = df.reset_index()
    df = df.sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.floor("s")
    return df


def get_server_info(PROMETHEUS_URL: str, server_info: dict):
    prometheus_url = PROMETHEUS_URL.rstrip("/")
    api_endpoint = f"{prometheus_url}/api/v1"
    query = (
        f"0 * system_cpu_time_seconds_total{{"
        f'GAME_TYPE="{server_info["game_type"]}",'
        f'WORLD="{server_info["world"]}",'
        f'INSTANCE="{server_info["instance"]}",'
        f'SVC_TYPE="{server_info["svc_type"]}",'
        f'SVR_TYPE="{server_info["svr_type"]}"'
        f"}}"
    )
    url = f"{api_endpoint}/query"
    try:
        response = requests.get(url, params={"query": query}, timeout=5)
        result = response.json()
        return result["data"]["result"][0]["metric"]
    except:
        return 1


def get_hierarchical_labels_with_info(
    prometheus_url, metric_name="system_cpu_time_seconds_total"
):
    """
    계층 구조 + 호스트 정보 수집
    """

    url = f"{prometheus_url}/api/v1/query"
    params = {"query": metric_name}

    response = requests.get(url, params=params)
    data = response.json()

    # 계층 구조 + 호스트 정보
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for result in data["data"]["result"]:
        labels = result["metric"]

        # 필수 라벨 체크
        required = ["CODE", "SVC_TYPE", "SVR_TYPE", "INSTANCE"]
        if not all(key in labels for key in required):
            continue

        code = labels["CODE"]
        svc_type = labels["SVC_TYPE"]
        svr_type = labels["SVR_TYPE"]
        instance = labels["INSTANCE"]

        # 호스트 정보 추출
        host_info = {
            "host_ip": labels.get("host_ip", ""),
            "host_mac": labels.get("host_mac", ""),
            "host_name": labels.get("host_name", ""),
        }

        # 이미 저장된 정보가 없거나, 같은 조합이면 저장
        key = (code, svc_type, svr_type, instance)
        if instance not in hierarchy[code][svc_type][svr_type]:
            hierarchy[code][svc_type][svr_type][instance] = host_info

    return hierarchy


def create_hierarchy_dataframe_with_info(hierarchy):
    """
    계층 구조 + 호스트 정보를 DataFrame으로 변환

    Returns:
        DataFrame with columns:
        [CODE, code_id, SVC_TYPE, svc_id, SVR_TYPE, svr_id, INSTANCE, instance_id,
         full_id, host_ip, host_mac, host_name]
    """

    rows = []

    # 1계층: CODE
    code_id = 1
    for code in sorted(hierarchy.keys()):
        code_id_str = f"{code_id:03d}"

        # 2계층: SVC_TYPE
        svc_id = 1
        for svc_type in sorted(hierarchy[code].keys()):
            svc_id_str = f"{svc_id:03d}"

            # 3계층: SVR_TYPE
            svr_id = 1
            for svr_type in sorted(hierarchy[code][svc_type].keys()):
                svr_id_str = f"{svr_id:03d}"

                # 4계층: INSTANCE
                inst_id = 1
                for instance in sorted(hierarchy[code][svc_type][svr_type].keys()):
                    inst_id_str = f"{inst_id:03d}"

                    # 호스트 정보
                    host_info = hierarchy[code][svc_type][svr_type][instance]

                    # Full ID
                    full_id = f"{code_id_str}-{svc_id_str}-{svr_id_str}-{inst_id_str}"

                    rows.append(
                        {
                            "CODE": code,
                            "code_id": code_id_str,
                            "SVC_TYPE": svc_type,
                            "svc_id": svc_id_str,
                            "SVR_TYPE": svr_type,
                            "svr_id": svr_id_str,
                            "INSTANCE": instance,
                            "instance_id": inst_id_str,
                            "full_id": full_id,
                            "host_ip": host_info["host_ip"],
                            "host_mac": host_info["host_mac"],
                            "host_name": host_info["host_name"],
                        }
                    )

                    inst_id += 1

                svr_id += 1

            svc_id += 1

        code_id += 1

    df = pd.DataFrame(rows)

    return df


def make_server_info(
    server_info_dir: str, object: str, work_date: datetime, prometheus_url: str
):
    output_file = "%s/%s/%s_%s_server_hierarchy.csv" % (
        server_info_dir,
        object,
        work_date.year,
        work_date.month,
    )
    if not os.path.exists(output_file):
        hierarchy = get_hierarchical_labels_with_info(prometheus_url)
        df = create_hierarchy_dataframe_with_info(hierarchy)
        df["used"] = 2  # 0: 사용, 1: 학습실패, 2: 학습 미실시
        df["reason"] = ""  # 학습 실패 사유 기록용
        df.to_csv(output_file, index=False)
        return df
    else:
        df = pd.read_csv(
            output_file,
            dtype={"code_id": str, "svc_id": str, "svr_id": str, "instance_id": str},
        )
        return df


def update_server_info(server_info_dir, server_map, object: str, work_date: datetime):
    output_file = "%s/%s/%s_%s_server_hierarchy.csv" % (
        server_info_dir,
        object,
        work_date.year,
        work_date.month,
    )
    server_map.to_csv(output_file, index=False)


def get_instance_ids(unique_row, server_map):
    instances = server_map[
        (server_map["CODE"] == unique_row[0])
        & (server_map["SVC_TYPE"] == unique_row[1])
        & (server_map["SVR_TYPE"] == unique_row[2])
    ]["INSTANCE"].tolist()

    instance_ids = server_map[
        (server_map["CODE"] == unique_row[0])
        & (server_map["SVC_TYPE"] == unique_row[1])
        & (server_map["SVR_TYPE"] == unique_row[2])
    ]["instance_id"].tolist()
    return instances, instance_ids


def create_memory_features(df):
    """
    Memory 파생변수 생성

    기본 메트릭:
    - memory_used_bytes
    - memory_free_bytes
    - memory_total_bytes
    - memory_used_rate
    """

    # ========================================
    # 1. 기본 계산
    # ========================================

    # 메모리 사용률 (%)
    df["memory_utilization"] = (
        df["memory_used_bytes"] / df["memory_total_bytes"]
    ) * 100

    # 가용 메모리 비율 (%)
    df["memory_free_ratio"] = (df["memory_free_bytes"] / df["memory_total_bytes"]) * 100

    # 사용/가용 비율
    df["memory_used_free_ratio"] = df["memory_used_bytes"] / (
        df["memory_free_bytes"] + 1e-9
    )

    # ========================================
    # 2. Lag 특성 (과거 값)
    # ========================================

    for lag in [1, 5, 10, 30, 60]:
        df[f"memory_util_lag_{lag}m"] = df["memory_utilization"].shift(lag)
        df[f"memory_used_lag_{lag}m"] = df["memory_used_bytes"].shift(lag)

    # ========================================
    # 3. 롤링 통계
    # ========================================

    for window in [10, 30, 60]:
        # 평균
        df[f"memory_util_mean_{window}m"] = (
            df["memory_utilization"].rolling(window).mean()
        )

        # 표준편차
        df[f"memory_util_std_{window}m"] = (
            df["memory_utilization"].rolling(window).std()
        )

        # 최대값
        df[f"memory_util_max_{window}m"] = (
            df["memory_utilization"].rolling(window).max()
        )

        # 최소값
        df[f"memory_util_min_{window}m"] = (
            df["memory_utilization"].rolling(window).min()
        )

    # ========================================
    # 4. 변화율
    # ========================================

    for diff in [1, 10, 30, 60]:
        # 사용률 변화량
        df[f"memory_util_change_{diff}m"] = df["memory_utilization"].diff(diff)

        # 사용량 변화량 (bytes)
        df[f"memory_used_change_{diff}m"] = df["memory_used_bytes"].diff(diff)

    # ========================================
    # 5. 추세
    # ========================================

    # 현재 vs 평균 비교
    df["memory_vs_mean_30m"] = df["memory_utilization"] - df["memory_util_mean_30m"]

    # 추세 (양수면 증가, 음수면 감소)
    df["memory_trend_30m"] = df["memory_utilization"] - df["memory_util_lag_30m"]

    # 가속도 (변화율의 변화)
    df["memory_acceleration"] = df["memory_util_change_1m"].diff(1)

    # ========================================
    # 6. 변동성
    # ========================================

    # 변동 폭
    df["memory_range_10m"] = df["memory_util_max_10m"] - df["memory_util_min_10m"]
    df["memory_range_30m"] = df["memory_util_max_30m"] - df["memory_util_min_30m"]

    # 변동계수 (CV)
    df["memory_cv_30m"] = df["memory_util_std_30m"] / (
        df["memory_util_mean_30m"] + 1e-9
    )

    # ========================================
    # 7. 임계값 관련
    # ========================================

    threshold = 85  # 85%

    # 임계값까지 거리
    df["memory_distance_to_threshold"] = threshold - df["memory_utilization"]

    # 임계값 근접도
    df["memory_threshold_proximity"] = df["memory_utilization"] / threshold

    # ========================================
    # 8. Rate 기반 (memory_used_rate)
    # ========================================

    # Rate의 변화
    df["memory_rate_change_1m"] = df["memory_used_rate"].diff(1)

    # Rate 평균
    df["memory_rate_mean_10m"] = df["memory_used_rate"].rolling(10).mean()

    # Rate 급증 여부
    df["target"] = (df["memory_utilization"] >= threshold).astype(int)

    return df
