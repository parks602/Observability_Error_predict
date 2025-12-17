"""
Main module for system resource prediction system.
Handles training and prediction for CPU, memory, and disk resources.
"""

import sys
import io

# Windows 한글 환경 인코딩 설정
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import argparse
import os
from datetime import datetime
from typing import Dict, Any
import pandas as pd

from src.utils.config_manager import Config
from src.utils.checker import model_date_checker, host_ip_converter
from src.data.prometeus_collector import (
    make_server_info,
    get_instance_ids,
    update_server_info,
)
from src.data.collector import (
    collect_train_data,
    collect_predict_data,
    _create_data_filepath,
)
from src.core.runner import train_model, predict_data
from src.database.connector import MariaDBHandler


# Constants
RESOURCE_TYPES = ["cpu", "memory", "disk"]
TRAIN_VALUE_FAILED = 1
TRAIN_VALUE_NEED_TRAINING = 2
TRAIN_VALUE_TRAINED = 0


class PredictionSystem:
    """Main prediction system orchestrator."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = Config("config/config.yaml")
        # self.server_config = Config("config/server.yaml")
        self.run_time = datetime.now().replace(microsecond=0)

        # Initialize database handler
        db_config = self.config.get_section("database")
        self.db_handler = self._init_database(db_config)

        # Get configurations
        self.object_config = self._get_object_config()
        self.alert_config = self.config.get_section("alert.dooray")
        self.server_info_dir = self.config.get_section("server_info.dir")
        self.prometheus_url = self.config.get("prometheus.url")

    def _init_database(self, db_config: Dict[str, Any]) -> MariaDBHandler:
        """Initialize database connection."""
        handler = MariaDBHandler(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            port=db_config["port"],
        )
        handler.connect()
        return handler

    def _get_object_config(self) -> Dict[str, Any]:
        """Get configuration for the target resource type."""
        config_map = {
            "cpu": "models.cpu",
            "memory": "models.memory",
            "disk": "models.disk",
        }
        return self.config.get_section(config_map[self.args.object])

    def _get_model_path(self) -> str:
        """Generate model path based on current date."""
        model_path = (
            f"{self.object_config['model_path']}/"
            f"{self.run_time.year}/{self.run_time.month:02d}"
        )
        os.makedirs(model_path, exist_ok=True)
        return model_path

    def _generate_model_name(self, code: str, game_type: str, svc_type: str) -> str:
        """Generate standardized model filename."""
        return f"{code}_{game_type}_{svc_type}.pkl"

    def _handle_training(
        self,
        server_map: pd.DataFrame,
        unique_row: pd.Series,
        model_path: str,
        model_name: str,
    ) -> None:
        """Handle model training process."""
        code = unique_row["CODE"]
        game_type = unique_row["GAME_TYPE"]
        svc_type = unique_row["SVC_TYPE"]
        svr_type = unique_row["SVR_TYPE"]
        world = unique_row["WORLD"]
        print(
            f"Training model for {code}-{game_type}-{svc_type} " f"- {self.args.object}"
        )
        mask = (
            (server_map["CODE"] == code)
            & (server_map["GAME_TYPE"] == game_type)
            & (server_map["SVC_TYPE"] == svc_type)
        )
        try:
            # Collect training data
            data = collect_train_data(
                target_object=self.args.object,
                server_map=server_map,
                prometheus_url=self.prometheus_url,
                end_time=self.run_time,
                window_size=30,
                code=code,
                game_type=game_type,
                svc_type=svc_type,
                svr_type=svr_type,
                world=world,
            )

            # Train model
            train_model(
                data=data,
                pred_object=self.args.object,
                model_path=model_path,
                model_name=model_name,
                threshold=self.object_config["threshold"],
                window_size=60,
            )

            # Update server info on success

            server_map.loc[mask, "used"] = TRAIN_VALUE_TRAINED
            update_server_info(
                self.server_info_dir, server_map, self.args.object, self.run_time
            )

        except ValueError as e:
            print(f"Model training failed: {e}")
            server_map.loc[mask, "used"] = TRAIN_VALUE_FAILED
            server_map.loc[mask, "reason"] = str(e)
            update_server_info(
                self.server_info_dir, server_map, self.args.object, self.run_time
            )

    def _handle_prediction(
        self,
        server_map: pd.DataFrame,
        unique_row: pd.Series,
        model_path: str,
        model_name: str,
    ) -> None:
        """Handle prediction for all instances."""
        code = unique_row["CODE"]
        game_type = unique_row["GAME_TYPE"]
        svc_type = unique_row["SVC_TYPE"]
        svr_type = unique_row["SVR_TYPE"]
        world = unique_row["WORLD"]
        instance = unique_row["INSTANCE"]
        instance_id = unique_row["instance_id"]
        svr_id = unique_row["svr_id"]
        world_id = unique_row["world_id"]
        print("=" * 70)
        print(
            f"object: {self.args.object} - "
            f"Predicting for {code}-{game_type}-{svc_type}-{svr_type}-{world}-{instance} "
        )

        # Collect prediction data
        data = collect_predict_data(
            target_object=self.args.object,
            instance=instance,
            instance_id=instance_id,
            prometheus_url=self.prometheus_url,
            end_time=self.run_time,
            window_size=1,
            code=code,
            game_type=game_type,
            svc_type=svc_type,
            svr_type=svr_type,
            world=world,
            svr_id=svr_id,
            world_id=world_id,
        )
        # Get server info

        server_info = {
            "host_ip": host_ip_converter(unique_row["host_ip"]),
            "host_name": unique_row["host_name"],
        }

        # Make prediction
        predict_data(
            data=data,
            pred_object=self.args.object,
            model_path=model_path,
            model_name=model_name,
            alert_config=self.alert_config,
            server_info=server_info,
            handler=self.db_handler,
            unique_row=unique_row,
        )

    def run(self) -> None:
        """Main execution method."""
        # Get server information
        server_map = make_server_info(
            self.server_info_dir, self.args.object, self.run_time, self.prometheus_url
        )

        # Process each unique server combination
        for _, unique_row in server_map.iterrows():
            instances, instances_ids = get_instance_ids(unique_row, server_map)
            code = unique_row["CODE"]
            game_type = unique_row["GAME_TYPE"]
            svc_type = unique_row["SVC_TYPE"]
            train_value = unique_row["used"]

            # Prepare model paths
            model_path = self._get_model_path()
            model_name = self._generate_model_name(code, game_type, svc_type)

            # Handle based on training status
            if train_value == TRAIN_VALUE_FAILED:
                print(
                    f"Skipping training for {code}-{game_type}-{svc_type}"
                    "- Previous training failed."
                )
                continue

            elif train_value == TRAIN_VALUE_NEED_TRAINING:
                if model_date_checker(
                    model_path=model_path,
                    model_name=model_name,
                    train_value=train_value,
                ):
                    self._handle_training(
                        server_map,
                        unique_row,
                        model_path,
                        model_name,
                    )

            elif train_value == TRAIN_VALUE_TRAINED:
                self._handle_prediction(
                    server_map,
                    unique_row,
                    model_path,
                    model_name,
                )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="System Resource Prediction System")
    parser.add_argument(
        "--object",
        type=str,
        required=True,
        choices=RESOURCE_TYPES,
        help="Target resource type (cpu/memory/disk)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    system = PredictionSystem(args)
    system.run()


if __name__ == "__main__":
    main()
