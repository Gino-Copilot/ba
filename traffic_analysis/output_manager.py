import sys
import os
import platform
import json
import shutil
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

class OutputManager:
    """
    Manages output structure and file paths for traffic analysis.
    Implements a thread-safe singleton pattern.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OutputManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, base_dir: str = "traffic_results", cleanup_old: bool = False):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True

        project_root = Path(__file__).parent.parent
        self.base_dir = project_root / base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.current_model = None

        self._setup_logging()

        if cleanup_old:
            self._cleanup_old_results()

        self._save_configuration()
        logging.info(f"OutputManager initialized with base directory: {self.base_dir}")

    def _setup_logging(self):
        """Sets up logging for the OutputManager."""
        log_dir = self.base_dir / "logs" / self.timestamp / "system" / "info"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "output_manager.log"
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def _cleanup_old_results(self):
        """Removes old results in the base directory if cleanup_old=True."""
        try:
            if self.base_dir.exists():
                old_logs = self.base_dir / "logs"
                if old_logs.exists():
                    backup_dir = self.base_dir.parent / "logs_backup"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_logs), str(backup_dir / f"logs_{self.timestamp}"))

                shutil.rmtree(self.base_dir)
                logging.info(f"Cleaned up old results in {self.base_dir}")
        except Exception as e:
            logging.error(f"Error cleaning up old results: {e}")

    def _save_configuration(self):
        """Saves current configuration to a JSON file."""
        try:
            config = {
                "timestamp": self.timestamp,
                "base_dir": str(self.base_dir),
                "created_at": datetime.now().isoformat(),
                "python_version": sys.version,
                "platform": platform.platform()
            }

            config_dir = self.base_dir / self.timestamp
            config_dir.mkdir(parents=True, exist_ok=True)

            with open(config_dir / "config.json", "w") as f:
                json.dump(config, f, indent=4, default=str)

            logging.info(f"Configuration saved to {config_dir / 'config.json'}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    def get_path(self, category: str, subcategory: str, filename: str) -> Path:
        """
        Dynamically creates (if needed) and returns a file path.

        Args:
            category (str): e.g. "models", "features", "nfstream"
            subcategory (str): e.g. "metrics", "correlations"
            filename (str): Name of the file to be saved/used.

        Returns:
            Path: The complete path
        """
        try:
            if not category:
                raise ValueError("Category cannot be empty.")
            if not subcategory:
                raise ValueError("Subcategory cannot be empty.")
            if not filename:
                raise ValueError("Filename cannot be empty.")

            if category == "models" and self.current_model:
                path = self.base_dir / self.timestamp / category / self.current_model / subcategory
            else:
                path = self.base_dir / self.timestamp / category / subcategory

            path.mkdir(parents=True, exist_ok=True)
            full_path = path / filename
            logging.debug(f"Generated path: {full_path}")
            return full_path
        except Exception as e:
            logging.error(f"Error generating path: {e}")
            raise

    def set_current_model(self, model_name: str):
        """Sets the current model name. Directories are created on demand in get_path()."""
        try:
            self.current_model = model_name
            logging.info(f"Current model set to: {model_name}")
        except Exception as e:
            logging.error(f"Error setting current model: {e}")
            raise

    def get_model_path(self, model_name: str) -> Path:
        """Returns the base path for a specific model."""
        return self.base_dir / self.timestamp / "models" / model_name

    def get_size_info(self) -> Dict[str, int]:
        """Computes the size of each top-level directory in bytes."""
        try:
            size_info = {}
            timestamp_dir = self.base_dir / self.timestamp

            if not timestamp_dir.exists():
                return size_info

            for category_dir in timestamp_dir.iterdir():
                if category_dir.is_dir():
                    total_size = sum(f.stat().st_size for f in category_dir.glob('**/*') if f.is_file())
                    size_info[category_dir.name] = total_size

            size_info['total'] = sum(size_info.values())

            with open(timestamp_dir / 'size_info.json', 'w') as f:
                json.dump(size_info, f, indent=4)

            return size_info
        except Exception as e:
            logging.error(f"Error calculating size info: {e}")
            return {}

    def cleanup_intermediate_files(self):
        """
        Removes intermediate files (e.g. raw data) to save disk space.
        """
        try:
            intermediate_dir = self.base_dir / self.timestamp / "nfstream" / "intermediate"
            if intermediate_dir.exists():
                shutil.rmtree(intermediate_dir)
                logging.info("Cleaned up intermediate files.")
        except Exception as e:
            logging.error(f"Error cleaning up intermediate files: {e}")
