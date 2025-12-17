import pyodbc
import requests

import pandas as pd


def send_dooray_notification(
    message: str, dooray_web_hook_url: str, bot_name: str = "Observability Bot"
):
    """Dooray 알림 전송"""
    payload = {
        "botName": bot_name,
        "text": message,
    }

    try:
        response = requests.post(
            dooray_web_hook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        print("Dooray 알림 전송 성공")
        return True
    except Exception as e:
        print(f"Dooray 알림 전송 실패: {e}")
        return False


def dooray_notify(
    prediction_result: dict,
    alert_config,
    server_info: dict,
    threshold: float,
    unique_row: pd.Series,
):
    """예측 결과에 따라 Dooray 알림 전송"""
    # prediction_result["will_be_high"] = True
    message = (
        f"⚠️ {prediction_result['object_value'].upper()} 사용량 경고 예측 ⚠️\n"
        f"- 현재 시각: {prediction_result['timestamp']}\n"
        f"- 서버 게임: {unique_row['GAME_TYPE']}\n"
        f"- 서버 월드: {unique_row['WORLD']}\n"
        f"- 서버 코드: {unique_row['CODE']}\n"
        f"- 서비스 유형: {unique_row['SVC_TYPE']}\n"
        f"- 서버 유형: {unique_row['SVR_TYPE']}\n"
        f"- 인스턴스: {unique_row['INSTANCE']}\n"
        f"- 서버 IP: {server_info['host_ip']}\n"
        f"- 서버 HOSTNAME: {server_info['host_name']}\n"
        f"- 현재 {prediction_result['object_value'].upper()} 사용량: {prediction_result['current_value']:.2f}%\n"
        f"- 예측 결과: {prediction_result['prediction_label']}\n"
        f"- 예측 시점: {prediction_result['prediction_time']}\n"
    )
    if not prediction_result["object_value"] == "disk":
        if prediction_result["current_value"] < threshold:
            send_dooray_notification(
                message,
                dooray_web_hook_url=alert_config["webhook_url"],
                bot_name=alert_config.get("bot_name", "Observability Bot"),
            )
