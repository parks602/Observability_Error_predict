import pyodbc
import requests


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
    conditions: dict,
):
    """예측 결과에 따라 Dooray 알림 전송"""
    # prediction_result["will_be_high"] = True
    message = (
        f"⚠️ {prediction_result['object_value'].upper()} 사용량 경고 예측 ⚠️\n"
        f"- 현재 시각: {prediction_result['timestamp']}\n"
        f"- 서버 코드: {conditions['CODE']}\n"
        f"- 서비스 유형: {conditions['SVC_TYPE']}\n"
        f"- 서버 유형: {conditions['SVR_TYPE']}\n"
        f"- 인스턴스: {conditions.get('INSTANCE', 'N/A')}\n"
        f"- 서버 IP: {server_info['host_ip']}\n"
        f"- 서버 HOSTNAME: {server_info['host_name']}\n"
        f"- 현재 {prediction_result['object_value'].upper()} 사용량: {prediction_result['current_value']:.2f}%\n"
        f"- 예측 결과: {prediction_result['prediction_label']}\n"
        f"- 예측 시점: {prediction_result['prediction_time']}\n"
    )
    if prediction_result["current_value"] < threshold:
        send_dooray_notification(
            message,
            dooray_web_hook_url=alert_config["webhook_url"],
            bot_name=alert_config.get("bot_name", "Observability Bot"),
        )
