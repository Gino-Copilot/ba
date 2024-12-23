import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import ModelSelector
from traffic_analysis.feature_analyzer import FeatureAnalyzer
from traffic_analysis.data_visualizer import DataVisualizer
from traffic_analysis.output_manager import OutputManager
from traffic_analysis.shap_analyzer import SHAPAnalyzer


class TrafficAnalyzer:
    """
    Main entry point for analyzing proxy vs. normal traffic.
    It extracts features, performs feature analysis,
    trains/evaluates models (with GridSearch + Pipeline),
    and optionally runs SHAP analysis.
    """

    def __init__(self, proxy_dir: str, normal_dir: str, results_dir: str):
        """
        Initializes the TrafficAnalyzer with directory paths and logging setup.

        Args:
            proxy_dir (str): Path to the directory containing proxy traffic PCAP files.
            normal_dir (str): Path to the directory containing normal traffic PCAP files.
            results_dir (str): Output directory for logs, results, and other artifacts.
        """
        self.proxy_dir = self._validate_directory(proxy_dir)
        self.normal_dir = self._validate_directory(normal_dir)

        # Create the results directory if it does not exist
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize managers
        self.output_manager = OutputManager(base_dir=str(self.results_dir))
        self.data_visualizer = DataVisualizer(self.output_manager)
        self.model_selector = ModelSelector()

        self._setup_logging()
        logging.info("TrafficAnalyzer initialized successfully.")

    def _validate_directory(self, directory: str) -> str:
        """
        Ensures that the specified directory exists, otherwise raises an error.

        Args:
            directory (str): The directory path to validate.

        Returns:
            str: Absolute path of the validated directory.
        """
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        return str(path.resolve())

    def _setup_logging(self):
        """
        Configures logging to file (INFO+) and console (WARN+).
        """
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"

        # Acquire the root logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Internal level: DEBUG/INFO

        # Remove existing handlers to avoid duplication
        if logger.hasHandlers():
            logger.handlers.clear()

        # FileHandler (writes INFO and above to a file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # StreamHandler (prints WARN and ERROR to console)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('ERROR: %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        logging.info("Logging setup complete. (Detailed logs in file, warnings+ on console)")

    def run_analysis(self):
        """
        Executes the end-to-end analysis pipeline:
          1) Feature extraction from PCAPs
          2) Mapping 'normal'/'proxy' to 0/1
          3) Feature analysis
          4) Model training and evaluation (GridSearch + Pipeline)
          5) SHAP analysis if available
        """
        try:
            start_time = time.time()
            logging.info("Starting the traffic analysis pipeline...")

            # 1) Feature extraction
            extractor = NFStreamFeatureExtractor(self.output_manager)
            df = extractor.prepare_dataset(self.proxy_dir, self.normal_dir)
            if df.empty:
                logging.warning("No data extracted. Analysis aborted.")
                return

            # 2) Convert labels from 'normal'/'proxy' to 0/1
            if 'label' not in df.columns:
                logging.error("No 'label' column found in the dataset. Aborting.")
                return

            mapping = {'normal': 0, 'proxy': 1}
            df['label'] = df['label'].map(mapping)

            # Drop rows with unknown labels
            unknown_labels = df['label'].isna()
            if unknown_labels.any():
                logging.warning("Unknown labels found. Dropping affected rows.")
                df = df[~unknown_labels].copy()

            if df['label'].nunique() < 2:
                logging.warning("Less than two distinct classes found in the dataset. Aborting.")
                return

            # 3) Feature analysis
            analyzer = FeatureAnalyzer(self.output_manager)
            _ = analyzer.analyze_features(df)

            # 4) Train and evaluate models
            models = self.model_selector.get_all_models()
            for name, model in models.items():
                logging.info(f"Training and testing model: {name}")
                self.output_manager.set_current_model(name)

                classifier = ScikitLearnTrafficClassifier(
                    model=model,
                    output_manager=self.output_manager,
                    data_visualizer=self.data_visualizer,
                    test_size=0.2,
                    random_state=42,
                    cv_folds=5
                )
                metrics = classifier.train(df, target_column='label')

                # 5) SHAP analysis, falls Modell predict_proba unterstützt
                best_model_pipeline = classifier.best_estimator_
                if best_model_pipeline and hasattr(best_model_pipeline["model"], 'predict_proba'):
                    shap_analyzer = SHAPAnalyzer(best_model_pipeline["model"], self.output_manager)

                    # ---------------------------------------------
                    # Erstelle DataFrame mit Spaltennamen
                    # ---------------------------------------------
                    X_test_scaled = best_model_pipeline["scaler"].transform(classifier.X_test)
                    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=classifier.X_test.columns)

                    shap_analyzer.explain_global(X_test_scaled_df)
                else:
                    logging.info(f"Skipping SHAP analysis for {name} (model has no predict_proba).")

            duration = time.time() - start_time
            logging.info(f"Analysis completed in {duration:.2f} seconds.")
            logging.info(f"Results saved in: {self.results_dir}")

        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            raise

def main():
    """
    Entry point for standalone script execution.
    """
    try:
        # Hier deine tatsächlich existierenden Pfade angeben:
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/PROXY_test"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/PROTON_test"
        results_dir = "/home/gino/PycharmProjects/myenv/ba/results"

        analyzer = TrafficAnalyzer(proxy_dir, normal_dir, results_dir)
        analyzer.run_analysis()

    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
