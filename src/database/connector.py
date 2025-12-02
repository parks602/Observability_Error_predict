import mariadb
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MariaDBHandler:
    """
    MariaDB 연결 및 쿼리 핸들러
    """

    def __init__(
        self, host: str, user: str, password: str, database: str, port: int = 80
    ):
        """
        Args:
            host: DB 호스트
            user: DB 사용자
            password: DB 비밀번호
            database: DB 이름
            port: DB 포트 (기본: 80)
        """
        self.config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
            "port": port,
        }
        self.conn = None
        self.cursor = None

    def connect(self):
        """DB 연결"""
        try:
            self.conn = mariadb.connect(**self.config)
            self.cursor = self.conn.cursor()
            logger.info(
                f"✓ DB 연결 성공: {self.config['host']}/{self.config['database']}"
            )
        except mariadb.Error as e:
            logger.error(f"❌ DB 연결 실패: {e}")
            raise

    def close(self):
        """DB 연결 종료"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("✓ DB 연결 종료")

    def __enter__(self):
        """with 문 사용"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 문 종료"""
        self.close()

    def execute(self, query: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """
        쿼리 실행

        Args:
            query: SQL 쿼리
            params: 쿼리 파라미터

        Returns:
            조회 결과 (SELECT인 경우)
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            # SELECT인 경우 결과 반환
            if query.strip().upper().startswith("SELECT"):
                return self.cursor.fetchall()

            return []

        except mariadb.Error as e:
            logger.error(f"❌ 쿼리 실행 실패: {e}")
            logger.error(f"   쿼리: {query}")
            raise

    def commit(self):
        """트랜잭션 커밋"""
        if self.conn:
            self.conn.commit()

    def rollback(self):
        """트랜잭션 롤백"""
        if self.conn:
            self.conn.rollback()

    def table_exists(self, table_name: str) -> bool:
        """
        테이블 존재 여부 확인

        Args:
            table_name: 테이블 이름

        Returns:
            존재 여부
        """
        query = f"""
        SELECT COUNT(*)
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = '{self.config['database']}'
          AND TABLE_NAME = '{table_name}'
        """

        result = self.execute(query)
        return result[0][0] > 0

    def create_table(self, table_name: str, schema: str):
        """
        테이블 생성

        Args:
            table_name: 테이블 이름
            schema: CREATE TABLE 스키마
        """
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {schema}
        )
        """

        self.execute(query)
        self.commit()
        logger.info(f"✓ 테이블 생성: {table_name}")

    def insert(self, table_name: str, data: Dict[str, Any]):
        """
        데이터 삽입

        Args:
            table_name: 테이블 이름
            data: 삽입할 데이터 딕셔너리

        Example:
            handler.insert('test_table', {
                'code': 'OK200',
                'game_type': 'GAME',
                'svc_type': 'APP',
                'instance': 'SERVER1',
                'metric': 'CPU',
                'value': 77
            })
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))

        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        self.execute(query, tuple(data.values()))
        self.commit()
        logger.debug(f"✓ 데이터 삽입: {table_name}")

    def insert_many(self, table_name: str, columns: List[str], data_list: List[Tuple]):
        """
        여러 데이터 한번에 삽입

        Args:
            table_name: 테이블 이름
            columns: 컬럼 리스트
            data_list: 데이터 리스트

        Example:
            handler.insert_many('test_table',
                ['code', 'game_type', 'value'],
                [
                    ('OK200', 'GAME', 77),
                    ('OK201', 'GAME', 85),
                ]
            )
        """
        columns_str = ", ".join(columns)
        placeholders = ", ".join(["?"] * len(columns))

        query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

        try:
            self.cursor.executemany(query, data_list)
            self.commit()
            logger.info(f"✓ 데이터 {len(data_list)}개 삽입: {table_name}")
        except mariadb.Error as e:
            logger.error(f"❌ 배치 삽입 실패: {e}")
            self.rollback()
            raise

    def select(
        self,
        table_name: str,
        columns: str = "*",
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple]:
        """
        데이터 조회

        Args:
            table_name: 테이블 이름
            columns: 조회할 컬럼 (기본: *)
            where: WHERE 조건
            order_by: ORDER BY 조건
            limit: LIMIT 개수

        Returns:
            조회 결과

        Example:
            rows = handler.select('test_table',
                columns='code, value',
                where="metric = 'CPU'",
                order_by='timestamp DESC',
                limit=10
            )
        """
        query = f"SELECT {columns} FROM {table_name}"

        if where:
            query += f" WHERE {where}"

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit:
            query += f" LIMIT {limit}"

        return self.execute(query)

    def update(self, table_name: str, set_clause: Dict[str, Any], where: str):
        """
        데이터 업데이트

        Args:
            table_name: 테이블 이름
            set_clause: 업데이트할 컬럼과 값
            where: WHERE 조건

        Example:
            handler.update('test_table',
                {'value': 90},
                "code = 'OK200'"
            )
        """
        set_str = ", ".join([f"{k} = ?" for k in set_clause.keys()])
        query = f"UPDATE {table_name} SET {set_str} WHERE {where}"

        self.execute(query, tuple(set_clause.values()))
        self.commit()
        logger.info(f"✓ 데이터 업데이트: {table_name}")

    def delete(self, table_name: str, where: str):
        """
        데이터 삭제

        Args:
            table_name: 테이블 이름
            where: WHERE 조건

        Example:
            handler.delete('test_table', "timestamp < '2024-01-01'")
        """
        query = f"DELETE FROM {table_name} WHERE {where}"

        self.execute(query)
        self.commit()
        logger.info(f"✓ 데이터 삭제: {table_name}")
