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
        """
        Initializes the OutputManager.

        Args:
            base_dir (str): Base directory for all outputs (default is 'traffic_results').
            cleanup_old (bool): If True, old results in the base directory are removed.
        """
        # Prevent multiple initializations
        if hasattr(self, '_initialized'):
            return

        self._initialized = True

        # Derive the absolute path based on the project's directory
        project_root = Path(__file__).parent.parent
        self.base_dir = project_root / base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.current_model = None

        # Directory structure for organized output
        self.directory_structure = {
            "models": {
                "metrics": ["grid_search", "validation"],
                "plots": ["performance", "feature_importance"],
                "shap": ["global", "local"],
                "trained": ["final", "checkpoints"]
            },
            "features": {
                "correlations": ["matrices", "plots"],
                "distributions": ["univariate", "bivariate"],
                "importance": ["rankings", "plots"],
                "groups": ["statistics", "analyses"],
                "summaries": ["text", "json"]
            },
            "reports": {
                "summaries": ["model", "feature", "overall"],
                "visualizations": ["comparisons", "trends"]
            },
            "nfstream": {
                "intermediate": ["raw", "processed"],
                "processed": ["features", "labels"],
                "summaries": ["statistics", "metadata"]
            },
            "logs": {
                "analysis": ["info", "error"],
                "processing": ["info", "error"],
                "system": ["info", "error"]
            }
        }

        self._setup_logging()

        if cleanup_old:
            self._cleanup_old_results()

        self._create_base_structure()
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
        """Removes old results in the base directory."""
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

    def _create_base_structure(self):
        """Creates the complete directory structure based on the defined dictionary."""
        try:
            for main_dir, categories in self.directory_structure.items():
                main_path = self.base_dir / self.timestamp / main_dir
                for category, subdirs in categories.items():
                    for subdir in subdirs:
                        path = main_path / category / subdir
                        path.mkdir(parents=True, exist_ok=True)
            logging.info("Directory structure created successfully.")
        except Exception as e:
            logging.error(f"Error creating directory structure: {e}")
            raise

    def _save_configuration(self):
        """Saves the current configuration to a JSON file."""
        try:
            config = {
                "timestamp": self.timestamp,
                "base_dir": str(self.base_dir),
                "directory_structure": self.directory_structure,
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
        Generates a file path within the defined directory structure.

        Args:
            category (str): Top-level category (e.g., 'models').
            subcategory (str): Subcategory within the category (e.g., 'metrics').
            filename (str): Desired filename.

        Returns:
            Path: A complete, valid path to the requested file.
        """
        try:
            if category not in self.directory_structure:
                raise ValueError(f"Invalid category: {category}")
            if subcategory not in self.directory_structure[category]:
                raise ValueError(f"Invalid subcategory {subcategory} for category {category}")
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
        """
        Sets the current model and creates any required subdirectories.

        Args:
            model_name (str): Name of the current model.
        """
        try:
            self.current_model = model_name
            model_base = self.base_dir / self.timestamp / "models" / model_name

            for category, subdirs in self.directory_structure["models"].items():
                category_dir = model_base / category
                for subdir in subdirs:
                    (category_dir / subdir).mkdir(parents=True, exist_ok=True)

            logging.info(f"Current model set to: {model_name}")
        except Exception as e:
            logging.error(f"Error setting current model: {e}")
            raise

    def get_model_path(self, model_name: str) -> Path:
        """
        Returns the base path for a specific model.

        Args:
            model_name (str): Name of the model.

        Returns:
            Path: The base path associated with the given model.
        """
        return self.base_dir / self.timestamp / "models" / model_name

    def create_archive(self, include_intermediates: bool = False) -> Optional[str]:
        """
        Creates a ZIP archive of the current results.

        Args:
            include_intermediates (bool): If True, includes intermediate files.

        Returns:
            Optional[str]: Path to the created archive or None on failure.
        """
        try:
            archive_name = f"results_{self.timestamp}"
            if not include_intermediates:
                temp_dir = self.base_dir.parent / f"temp_{self.timestamp}"
                src = self.base_dir / self.timestamp
                shutil.copytree(src, temp_dir, ignore=shutil.ignore_patterns('intermediate*'))
                archive_path = shutil.make_archive(archive_name, 'zip', temp_dir)
                shutil.rmtree(temp_dir)
            else:
                archive_path = shutil.make_archive(archive_name, 'zip', self.base_dir, self.timestamp)

            logging.info(f"Created archive: {archive_path}")
            return archive_path
        except Exception as e:
            logging.error(f"Error creating archive: {e}")
            return None

    def get_size_info(self) -> Dict[str, int]:
        """
        Computes the size of each top-level directory in bytes.

        Returns:
            Dict[str, int]: A mapping of directory to size in bytes.
        """
        try:
            size_info = {}
            timestamp_dir = self.base_dir / self.timestamp

            for category in self.directory_structure:
                path = timestamp_dir / category
                total_size = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
                size_info[category] = total_size

            size_info['total'] = sum(size_info.values())

            with open(timestamp_dir / 'size_info.json', 'w') as f:
                json.dump(size_info, f, indent=4)

            return size_info
        except Exception as e:
            logging.error(f"Error calculating size info: {e}")
            return {}

    def cleanup_intermediate_files(self):
        """
        Removes intermediate files (e.g., raw data) to save disk space.
        """
        try:
            intermediate_dir = self.base_dir / self.timestamp / "nfstream" / "intermediate"
            if intermediate_dir.exists():
                shutil.rmtree(intermediate_dir)
                logging.info("Cleaned up intermediate files.")
        except Exception as e:
            logging.error(f"Error cleaning up intermediate files: {e}")


if __name__ == "__main__":
    # Example usage / quick test
    manager = OutputManager(cleanup_old=True)
    print(f"Base directory: {manager.base_dir}")
    print(f"Current timestamp: {manager.timestamp}\n")

    path = manager.get_path("models", "metrics", "test.csv")
    print(f"Generated path: {path}")
