import yaml
import os
import logging
from string import Template

logger = logging.getLogger(__name__)


class Config:
    """
    설정 관리 클래스

    기능:
    - config.yaml 로드
    - 환경변수 치환 (${VAR_NAME})
    - 점 표기법 접근 (config.prometheus.url)
    """

    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self._config = {}

        self._load_config()

    def _substitute_env_vars(self, obj):
        """
        환경변수 치환
        ${VAR_NAME} → os.getenv('VAR_NAME')
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # ${VAR_NAME} 패턴 찾기
            if "${" in obj and "}" in obj:
                # Template을 사용한 안전한 치환
                template = Template(obj)
                try:
                    # 환경변수 딕셔너리 생성
                    env_dict = {k: v for k, v in os.environ.items()}
                    result = template.substitute(env_dict)

                    # 치환되지 않은 변수가 있는지 확인
                    if "${" in result:
                        logger.warning(f"⚠️ 환경변수 누락: {obj}")

                    return result
                except KeyError as e:
                    logger.warning(f"⚠️ 환경변수 없음: {e}")
                    return obj
            return obj
        else:
            return obj

    def _load_config(self):
        """설정 파일 로드"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            # 환경변수 치환
            self._config = self._substitute_env_vars(raw_config)

            logger.info(f"✓ 설정 파일 로드: {self.config_path}")

        except FileNotFoundError:
            logger.error(f"❌ 설정 파일 없음: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML 파싱 에러: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 설정 로드 실패: {e}")
            raise

    def get(self, key: str, default: any = None) -> any:
        """
        점 표기법으로 설정 가져오기

        Args:
            key: 설정 키 (예: 'prometheus.url')
            default: 기본값

        Returns:
            설정 값 또는 기본값

        Examples:
            >>> config.get('prometheus.url')
            'http://prometheus:9090'

            >>> config.get('alert.slack.webhook_url')
            'https://hooks.slack.com/...'
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> dict:
        """
        전체 섹션 가져오기

        Args:
            section: 섹션 이름 (예: 'prometheus')

        Returns:
            섹션 딕셔너리

        Examples:
            >>> config.get_section('prometheus')
            {'url': '...', 'timeout': 30, ...}
        """
        return self.get(section, {})

    def to_dict(self) -> dict:
        """전체 설정을 딕셔너리로 반환"""
        return self._config.copy()

    def print_config(self, mask_secrets: bool = True):
        """
        설정 출력 (디버깅용)

        Args:
            mask_secrets: 민감한 정보 마스킹 여부
        """
        import json

        config_copy = self._config.copy()

        if mask_secrets:
            config_copy = self._mask_secrets(config_copy)

        print("=" * 70)
        print("현재 설정")
        print("=" * 70)
        print(json.dumps(config_copy, indent=2, ensure_ascii=False))
        print("=" * 70)

    def _mask_secrets(self, obj: any) -> any:
        """민감한 정보 마스킹"""
        secret_keys = ["password", "token", "key", "secret", "webhook_url"]

        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if any(secret in k.lower() for secret in secret_keys):
                    result[k] = "***MASKED***" if v else None
                else:
                    result[k] = self._mask_secrets(v)
            return result
        elif isinstance(obj, list):
            return [self._mask_secrets(item) for item in obj]
        else:
            return obj


def load_env_file(env_file: str = ".env"):
    """
    .env 파일에서 환경변수 로드

    Args:
        env_file: .env 파일 경로
    """
    if not os.path.exists(env_file):
        logger.warning(f"⚠️ .env 파일 없음: {env_file}")
        return

    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()

                # 주석이나 빈 줄 무시
                if not line or line.startswith("#"):
                    continue

                # KEY=VALUE 파싱
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # 따옴표 제거
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    os.environ[key] = value

        logger.info(f".env 파일 로드: {env_file}")
    except Exception as e:
        logger.error(f".env 파일 로드 실패: {e}")
