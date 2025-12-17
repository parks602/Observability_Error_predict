import logging
from datetime import datetime, timedelta
import time
import os
import pandas as pd
import warnings
import requests
from collections import defaultdict


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def make_server_info(
    server_info_dir: str, object: str, work_date: datetime, prometheus_url: str
):
    file_dir = "%s/%s" % (server_info_dir, object)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    output_file = "%s/%s_%s_server_hierarchy.csv" % (
        file_dir,
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
            dtype={
                "code_id": str,
                "game_id": str,
                "svc_id": str,
                "svr_id": str,
                "instance_id": str,
                "world_id": str,
            },
        )
        return df


def get_hierarchical_labels_with_info(
    prometheus_url, metric_name="system_filesystem_usage_bytes"
):
    """
    계층 구조 + 호스트 정보 수집
    """

    url = f"{prometheus_url}/api/v1/query"
    params = {"query": metric_name}

    response = requests.get(url, params=params)
    data = response.json()

    def tree():
        return defaultdict(tree)

    # 계층 구조 + 호스트 정보
    hierarchy = tree()

    for result in data["data"]["result"]:
        labels = result["metric"]

        # 필수 라벨 (WORLD 제외)
        required = ["CODE", "SVC_TYPE", "SVR_TYPE", "INSTANCE", "GAME_TYPE"]
        if not all(key in labels for key in required):
            continue

        code = labels["CODE"]
        svc_type = labels["SVC_TYPE"]
        svr_type = labels["SVR_TYPE"]
        instance = labels["INSTANCE"]
        game_type = labels["GAME_TYPE"]

        # WORLD는 선택 라벨 → 없으면 ""
        world = labels.get("WORLD", "")

        # 호스트 정보
        host_info = {
            "host_ip": labels.get("host_ip", ""),
            "host_mac": labels.get("host_mac", ""),
            "host_name": labels.get("host_name", ""),
        }

        # 계층 키
        key = (code, game_type, svc_type, svr_type, world, instance)

        if instance not in hierarchy[code][game_type][svc_type][svr_type][world]:
            hierarchy[code][game_type][svc_type][svr_type][world][instance] = host_info

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
        game_id = 1
        for game_type in sorted(hierarchy[code].keys()):
            game_id_str = f"{game_id:03d}"
            # 3계층: SVC_TYPE
            svc_id = 1
            for svc_type in sorted(hierarchy[code][game_type].keys()):
                svc_id_str = f"{svc_id:03d}"

                # 4계층: SVR_TYPE
                svr_id = 1
                for svr_type in sorted(hierarchy[code][game_type][svc_type].keys()):
                    svr_id_str = f"{svr_id:03d}"

                    # 5계층: INSTANCE
                    world_id = 1

                    for world in sorted(
                        hierarchy[code][game_type][svc_type][svr_type].keys()
                    ):
                        world_id_str = f"{world_id:03d}"

                        # 6계층: WORLD
                        inst_id = 1
                        for instance in sorted(
                            hierarchy[code][game_type][svc_type][svr_type][world].keys()
                        ):
                            inst_id_str = f"{inst_id:03d}"

                            # 호스트 정보
                            host_info = hierarchy[code][game_type][svc_type][svr_type][
                                world
                            ][instance]

                            # Full ID
                            full_id = f"{code_id_str}-{game_id_str}-{svc_id_str}-{svr_id_str}-{inst_id_str}-{world_id_str}"

                            rows.append(
                                {
                                    "CODE": code,
                                    "code_id": code_id_str,
                                    "GAME_TYPE": game_type,
                                    "game_id": game_id_str,
                                    "SVC_TYPE": svc_type,
                                    "svc_id": svc_id_str,
                                    "SVR_TYPE": svr_type,
                                    "svr_id": svr_id_str,
                                    "INSTANCE": instance,
                                    "instance_id": inst_id_str,
                                    "WORLD": world,
                                    "world_id": world_id_str,
                                    "full_id": full_id,
                                    "host_ip": host_info["host_ip"],
                                    "host_mac": host_info["host_mac"],
                                    "host_name": host_info["host_name"],
                                }
                            )
                            inst_id += 1

                        world_id += 1

                    svr_id += 1

                svc_id += 1

            game_id += 1

        code_id += 1

    df = pd.DataFrame(rows)

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
        & (server_map["GAME_TYPE"] == unique_row[1])
        & (server_map["SVC_TYPE"] == unique_row[2])
        & (server_map["SVR_TYPE"] == unique_row[3])
        & (server_map["WORLD"] == unique_row[4])
    ]["INSTANCE"].tolist()

    instance_ids = server_map[
        (server_map["CODE"] == unique_row[0])
        & (server_map["GAME_TYPE"] == unique_row[1])
        & (server_map["SVC_TYPE"] == unique_row[2])
        & (server_map["SVR_TYPE"] == unique_row[3])
        & (server_map["WORLD"] == unique_row[4])
    ]["instance_id"].tolist()
    return instances, instance_ids
