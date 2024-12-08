import os
from datetime import datetime
from pathlib import Path
import shutil
import json
import logging
from typing import Dict, List, Optional, Union
import threading


class OutputManager:
    """
    Verwaltet die Ausgabestruktur und Dateipfade für die Verkehrsanalyse
    Thread-safe Singleton Implementation
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(OutputManager, cls).__new__(cls)
            return cls._instance

    def __init__(self, base_dir: str = "traffic_results", cleanup_old: bool = False):
        """
        Initialisiert den OutputManager

        Args:
            base_dir: Basis-Verzeichnis für alle Ausgaben (default: 'traffic_results')
            cleanup_old: Wenn True, werden alte Ergebnisse gelöscht
        """
        # Verhindert mehrfache Initialisierung des Singletons
        if hasattr(self, '_initialized'):
            return

        # Bestimme absoluten Pfad basierend auf Projektverzeichnis
        project_root = Path(__file__).parent.parent
        self.base_dir = project_root / base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.current_model = None

        # Directory Struktur
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

        self._initialized = True
        self._setup_logging()

        if cleanup_old:
            self._cleanup_old_results()

        self._create_base_structure()
        self._save_configuration()

        logging.info(f"OutputManager initialized with base directory: {self.base_dir}")

    def _setup_logging(self):
        """Konfiguriert Logging für den OutputManager"""
        log_dir = self.base_dir / "logs" / self.timestamp / "system" / "info"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"output_manager.log"

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def _cleanup_old_results(self):
        """Löscht alte Ergebnisse"""
        try:
            if self.base_dir.exists():
                # Sichere alte Logs
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
        """Erstellt die komplette Verzeichnisstruktur"""
        try:
            for main_dir, categories in self.directory_structure.items():
                main_path = self.base_dir / self.timestamp / main_dir

                for category, subdirs in categories.items():
                    for subdir in subdirs:
                        path = main_path / category / subdir
                        path.mkdir(parents=True, exist_ok=True)

            logging.info("Directory structure created successfully")

        except Exception as e:
            logging.error(f"Error creating directory structure: {e}")
            raise

    def _save_configuration(self):
        """Speichert die aktuelle Konfiguration"""
        try:
            config = {
                'timestamp': self.timestamp,
                'base_dir': str(self.base_dir),
                'directory_structure': self.directory_structure,
                'created_at': datetime.now().isoformat(),
                'python_version': sys.version,
                'platform': platform.platform()
            }

            config_dir = self.base_dir / self.timestamp
            config_dir.mkdir(parents=True, exist_ok=True)

            with open(config_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=4, default=str)

            logging.info(f"Configuration saved to {config_dir / 'config.json'}")

        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    def get_path(self, category: str, subcategory: str, filename: str) -> Path:
        """
        Generiert einen Pfad für eine Datei

        Args:
            category: Hauptkategorie (z.B. 'models')
            subcategory: Unterkategorie (z.B. 'metrics')
            filename: Name der Datei

        Returns:
            Path: Vollständiger Pfad zur Datei
        """
        try:
            # Validiere Kategorie und Unterkategorie
            if category not in self.directory_structure:
                raise ValueError(f"Invalid category: {category}")

            if subcategory not in self.directory_structure[category]:
                raise ValueError(f"Invalid subcategory {subcategory} for category {category}")

            if category == "models" and self.current_model:
                path = self.base_dir / self.timestamp / category / self.current_model / subcategory
            else:
                path = self.base_dir / self.timestamp / category / subcategory

            # Stelle sicher, dass das Verzeichnis existiert
            path.mkdir(parents=True, exist_ok=True)

            # Validiere Dateinamen
            if not filename:
                raise ValueError("Filename cannot be empty")

            # Erstelle vollständigen Pfad
            full_path = path / filename

            # Logge Pfaderstellung
            logging.debug(f"Generated path: {full_path}")

            return full_path

        except Exception as e:
            logging.error(f"Error generating path: {e}")
            raise

    def set_current_model(self, model_name: str):
        """
        Setzt das aktuelle Modell und erstellt entsprechende Verzeichnisse

        Args:
            model_name: Name des aktuellen Modells
        """
        try:
            self.current_model = model_name
            model_base = self.base_dir / self.timestamp / "models" / model_name

            # Erstelle alle notwendigen Unterverzeichnisse
            for category, subdirs in self.directory_structure["models"].items():
                category_dir = model_base / category
                for subdir in subdirs:
                    (category_dir / subdir).mkdir(parents=True, exist_ok=True)

            logging.info(f"Set current model to: {model_name}")

        except Exception as e:
            logging.error(f"Error setting current model: {e}")
            raise

    def get_model_path(self, model_name: str) -> Path:
        """
        Gibt den Basispfad für ein spezifisches Modell zurück

        Args:
            model_name: Name des Modells

        Returns:
            Path: Basispfad des Modells
        """
        return self.base_dir / self.timestamp / "models" / model_name

    def create_archive(self, include_intermediates: bool = False) -> Optional[str]:
        """
        Erstellt ein Archiv der aktuellen Ergebnisse

        Args:
            include_intermediates: Wenn True, werden auch Zwischenergebnisse archiviert

        Returns:
            Optional[str]: Pfad zum erstellten Archiv oder None bei Fehler
        """
        try:
            archive_name = f"results_{self.timestamp}"

            if not include_intermediates:
                # Erstelle temporäres Verzeichnis ohne Zwischenergebnisse
                temp_dir = self.base_dir.parent / f"temp_{self.timestamp}"
                shutil.copytree(self.base_dir / self.timestamp, temp_dir,
                                ignore=shutil.ignore_patterns('intermediate*'))
                archive_path = shutil.make_archive(archive_name, 'zip', temp_dir)
                shutil.rmtree(temp_dir)
            else:
                archive_path = shutil.make_archive(
                    archive_name, 'zip', self.base_dir, self.timestamp
                )

            logging.info(f"Created archive: {archive_path}")
            return archive_path

        except Exception as e:
            logging.error(f"Error creating archive: {e}")
            return None

    def get_size_info(self) -> Dict[str, int]:
        """
        Berechnet die Größe der verschiedenen Verzeichnisse

        Returns:
            Dict[str, int]: Größeninformationen in Bytes
        """
        try:
            size_info = {}
            timestamp_dir = self.base_dir / self.timestamp

            for category in self.directory_structure.keys():
                path = timestamp_dir / category
                total_size = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
                size_info[category] = total_size

            # Füge Gesamtgröße hinzu
            size_info['total'] = sum(size_info.values())

            # Speichere Größeninformationen
            with open(timestamp_dir / 'size_info.json', 'w') as f:
                json.dump(size_info, f, indent=4)

            return size_info

        except Exception as e:
            logging.error(f"Error calculating size info: {e}")
            return {}

    def cleanup_intermediate_files(self):
        """Löscht Zwischenergebnisse um Speicherplatz zu sparen"""
        try:
            intermediate_dir = self.base_dir / self.timestamp / "nfstream" / "intermediate"
            if intermediate_dir.exists():
                shutil.rmtree(intermediate_dir)
                logging.info("Cleaned up intermediate files")
        except Exception as e:
            logging.error(f"Error cleaning up intermediate files: {e}")


# Importiere benötigte Module für die Konfigurationsspeicherung
import sys
import platform

if __name__ == "__main__":
    # Test-Code
    output_manager = OutputManager(cleanup_old=True)
    print(f"Base directory: {output_manager.base_dir}")
    print(f"Current timestamp: {output_manager.timestamp}")
    print("\nTesting path generation:")
    test_path = output_manager.get_path("models", "metrics", "test.csv")
    print(f"Generated test path: {test_path}")