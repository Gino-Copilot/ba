import os
from datetime import datetime


class OutputManager:
    def __init__(self, base_dir="results"):
        """
        Initialize OutputManager

        Args:
            base_dir: Base directory for all outputs (default: 'results')
        """
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.current_model = None
        self._create_base_structure()

    def _create_base_structure(self):
        """Creates the base directory structure."""
        directories = {
            "models": ["metrics", "plots", "shap"],
            "features": ["correlations", "distributions", "importance"],
            "reports": ["summaries", "visualizations"]
        }

        for main_dir, subdirs in directories.items():
            for subdir in subdirs:
                dir_path = os.path.join(self.base_dir, main_dir, self.timestamp, subdir)
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")

    def get_path(self, category, subcategory, filename):
        """
        Generate complete file path.

        Args:
            category: Main category (e.g., 'models', 'features', 'reports')
            subcategory: Sub-category (e.g., 'metrics', 'plots', 'summaries')
            filename: Name of the file

        Returns:
            str: Complete file path
        """
        if category == "models" and self.current_model:
            base = os.path.join(self.base_dir, category, self.timestamp,
                                self.current_model, subcategory)
        else:
            base = os.path.join(self.base_dir, category, self.timestamp, subcategory)

        os.makedirs(base, exist_ok=True)
        full_path = os.path.join(base, filename)
        print(f"Generated path: {full_path}")
        return full_path

    def set_current_model(self, model_name):
        """
        Set current model and create its directories.

        Args:
            model_name: Name of the current model
        """
        self.current_model = model_name
        model_dir = os.path.join(self.base_dir, "models", self.timestamp, model_name)

        for subdir in ["metrics", "plots", "shap"]:
            dir_path = os.path.join(model_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created model directory: {dir_path}")

    def get_model_dir(self, model_name):
        """
        Get directory for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            str: Path to model directory
        """
        self.set_current_model(model_name)
        return os.path.join(self.base_dir, "models", self.timestamp, model_name)

    def get_summary_path(self, filename):
        """
        Get path for summary files in the reports/summaries directory.

        Args:
            filename: Name of the summary file

        Returns:
            str: Complete path to the summary file
        """
        summary_dir = os.path.join(self.base_dir, "reports", self.timestamp, "summaries")
        os.makedirs(summary_dir, exist_ok=True)
        full_path = os.path.join(summary_dir, filename)
        print(f"Generated summary path: {full_path}")
        return full_path

    def get_visualization_path(self, filename):
        """
        Get path for visualization files in the reports/visualizations directory.

        Args:
            filename: Name of the visualization file

        Returns:
            str: Complete path to the visualization file
        """
        visualization_dir = os.path.join(self.base_dir, "reports", self.timestamp, "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
        full_path = os.path.join(visualization_dir, filename)
        print(f"Generated visualization path: {full_path}")
        return full_path
