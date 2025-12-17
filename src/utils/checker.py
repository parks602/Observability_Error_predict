import os
import ast


def model_date_checker(model_path: str, model_name: str, train_value: int):
    if not os.path.exists(f"{model_path}/{model_name}") & train_value == 2:
        print(f"Model file does not exist: {model_path}/{model_name}")
        return True
    elif os.path.getsize(model_path + model_name) == 0:
        print(f"Model file is empty: {model_path}/{model_name}")
        return False
    else:
        print(f"Model file exists: {model_path}/{model_name}")
        return False


def host_ip_converter(ip_str):
    try:
        # 문자열을 실제 리스트로 변환
        ip_list = ast.literal_eval(ip_str)
    except:
        return ""

    # IPv6 (: 포함) 제거 후 IPv4만 남기기
    ipv4_list = [ip for ip in ip_list if ":" not in ip]

    # 다시 한 줄 문자열로 변환
    return ",".join(ipv4_list)
