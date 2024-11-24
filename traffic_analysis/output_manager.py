import os
import time
import shutil
from datetime import datetime


class OutputManager:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.current_model = None
        self._create_base_structure()

    def _create_base_structure(self):
        """Erstellt die Basis-Verzeichnisstruktur"""
        directories = {
            "models": ["metrics", "plots", "shap"],
            "features": ["correlations", "distributions", "importance"],
            "nfstream": ["raw", "processed"],
            "reports": ["summaries", "visualizations"]
        }

        for main_dir, subdirs in directories.items():
            for subdir in subdirs:
                os.makedirs(
                    os.path.join(self.base_dir, main_dir, self.timestamp, subdir),
                    exist_ok=True
                )

    def get_path(self, category, subcategory, filename):
        """Generiert einen vollständigen Dateipfad"""
        if category == "models" and self.current_model:
            base = os.path.join(self.base_dir, category, self.timestamp,
                                self.current_model, subcategory)
        else:
            base = os.path.join(self.base_dir, category, self.timestamp, subcategory)

        os.makedirs(base, exist_ok=True)
        return os.path.join(base, filename)

    def set_current_model(self, model_name):
        """Setzt das aktuelle Modell"""
        self.current_model = model_name
        model_dir = os.path.join(self.base_dir, "models", self.timestamp, model_name)
        for subdir in ["metrics", "plots", "shap"]:
            os.makedirs(os.path.join(model_dir, subdir), exist_ok=True)