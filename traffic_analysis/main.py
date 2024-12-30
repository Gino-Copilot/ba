# file: main.py

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Local modules
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import ModelSelector
from traffic_analysis.feature_analyzer import FeatureAnalyzer
from traffic_analysis.data_visualizer import DataVisualizer
from traffic_analysis.output_manager import OutputManager
from traffic_analysis.shap_analyzer import SHAPAnalyzer

# Updated imports
from traffic_analysis.data_cleaner import DataCleaner
from traffic_analysis.data_inspector import DataInspector

from joblib import dump


class TrafficAnalyzer:
    """
    Main pipeline that coordinates:
      1) Copying valid PCAP files (via DataInspector) to new "clean" folders
      2) Feature extraction (NFStream)
      3) Data cleaning (DataCleaner)
      4) Model training with GridSearch (multiple models)
      5) SHAP analysis (if applicable)
      6) Visualization and logging
      7) Storing only the best model's pipeline in a new 'trained/best' folder
         under the main timestamp directory (with an info file about data origin).
    """

    def __init__(self, proxy_dir: str, normal_dir: str, results_dir: str):
        """
        Initializes the TrafficAnalyzer with user-specified proxy/normal PCAP folders
        and a results directory. Stores all outputs in a subfolder named:
           "<proxy_name>_vs_<normal_name>_<dd-mm-yyyy_hh-mm>"

        Args:
            proxy_dir: Folder containing the original proxy PCAP files.
            normal_dir: Folder containing the original normal PCAP files.
            results_dir: Base results directory for all analyses.
        """
        self.proxy_dir = self._validate_directory(proxy_dir)
        self.normal_dir = self._validate_directory(normal_dir)

        # Use folder names for naming the result directory
        proxy_name = Path(self.proxy_dir).name
        normal_name = Path(self.normal_dir).name

        # Day-Month-Year_Hour-Minute (no seconds)
        timestamp = time.strftime('%d-%m-%Y_%H-%M')

        # Create the subfolder for this particular analysis
        folder_name = f"{proxy_name}_vs_{normal_name}_{timestamp}"
        analysis_dir = Path(results_dir) / folder_name
        analysis_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = analysis_dir

        # OutputManager for path handling
        self.output_manager = OutputManager(base_dir=str(self.results_dir))
        self.data_visualizer = DataVisualizer(self.output_manager)
        self.model_selector = ModelSelector()

        self._setup_logging()
        logging.info("TrafficAnalyzer initialized.")
        logging.info(f"Proxy folder: {self.proxy_dir}")
        logging.info(f"Normal folder: {self.normal_dir}")
        logging.info(f"Analysis folder: {self.results_dir}")

        # subfolders for "valid" PCAP copies
        self.clean_proxy_dir = str(self.results_dir / "clean_data" / "proxy")
        self.clean_normal_dir = str(self.results_dir / "clean_data" / "normal")

        # Track the best model overall
        self.best_model_name = None
        self.best_accuracy = 0.0
        self.best_pipeline = None

    def _validate_directory(self, directory: str) -> str:
        """
        Ensures the given directory exists; raises ValueError if it does not.
        """
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        return str(path.resolve())

    def _setup_logging(self):
        """
        Sets up logging to a file in self.results_dir/logs plus console (warnings+).
        """
        log_dir = Path(self.results_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log file (no seconds)
        log_file = log_dir / f"analysis_{time.strftime('%d-%m-%Y_%H-%M')}.log"

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        logging.info("Logging setup complete (INFO to file, WARNING+ to console).")

    def run_analysis(self):
        """
        Main pipeline method:
          1) Copy PCAPs from original folders -> 'clean_data' subfolders
          2) Plot PCAP size distribution
          3) NFStream feature extraction
          4) Data cleaning
          5) Model training (GridSearch if param_grid != {})
          6) SHAP analysis if predict_proba is supported
          7) Compare model accuracies
          8) Save only the best model pipeline in 'trained/best'
        """
        try:
            start_time = time.time()
            logging.info("Starting the traffic analysis pipeline...")

            # 1) DataInspector copies valid PCAPs
            inspector = DataInspector(min_file_size_bytes=1000, min_flow_count=2)
            inspector.copy_valid_pcaps(self.proxy_dir, self.clean_proxy_dir)
            inspector.copy_valid_pcaps(self.normal_dir, self.clean_normal_dir)

            # 2) Plot PCAP size distribution
            pcap_sizes = {
                "proxy": self._calculate_folder_bytes_list(self.clean_proxy_dir),
                "normal": self._calculate_folder_bytes_list(self.clean_normal_dir)
            }
            self.data_visualizer.plot_pcap_size_distribution(pcap_sizes)

            # 3) NFStream feature extraction
            extractor = NFStreamFeatureExtractor(
                self.output_manager,
                use_entropy=False,
                min_packets=2
            )
            df = extractor.prepare_dataset(self.clean_proxy_dir, self.clean_normal_dir)
            if df.empty:
                logging.error("DataFrame is empty after feature extraction.")
                return

            logging.info(f"DataFrame shape after extraction: {df.shape}")

            # 4) Data cleaning
            cleaner = DataCleaner(min_packet_threshold=2, impute_with_median=True)
            df = cleaner.clean_dataset(df)
            if df.empty:
                logging.error("DataFrame is empty after cleaning.")
                return

            if 'label' not in df.columns:
                logging.error("No 'label' column found in DataFrame.")
                return

            # Convert 'normal'/'proxy' -> 0/1
            label_map = {'normal': 0, 'proxy': 1}
            df['label'] = df['label'].map(label_map)
            unknown_rows = df['label'].isna()
            if unknown_rows.any():
                logging.warning("Dropping rows with unknown labels (NaN).")
                df = df[~unknown_rows]

            if df['label'].nunique() < 2:
                logging.warning("Less than two distinct labels remain. Aborting.")
                return

            # Inspector checks the final DataFrame
            inspector.check_flow_dataframe(df)

            # 5) Model training
            analyzer = FeatureAnalyzer(self.output_manager, target_column='label')
            _ = analyzer.analyze_features(df)

            models = self.model_selector.get_all_models()
            metrics_list = []

            for name, (model_obj, param_grid) in models.items():
                logging.info("=" * 50)
                logging.info(f"TRAINING MODEL: {name}")
                logging.info("=" * 50)

                self.output_manager.set_current_model(name)

                classifier = ScikitLearnTrafficClassifier(
                    model=model_obj,
                    output_manager=self.output_manager,
                    data_visualizer=self.data_visualizer,
                    test_size=0.2,
                    random_state=42,
                    cv_folds=5,
                    param_grid=param_grid,  # if empty, no GridSearch
                    gridsearch_scoring="accuracy"
                )
                metrics = classifier.train(df, target_column='label')
                accuracy = metrics.get("accuracy", 0.0)

                if accuracy:
                    metrics_list.append((name, accuracy))
                    logging.info(f"{name} => Accuracy: {accuracy:.3f}")
                    # Track best
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.best_model_name = name
                        self.best_pipeline = classifier.best_estimator_
                else:
                    logging.warning(f"No accuracy in metrics for {name}.")

                # 6) SHAP analysis if predict_proba is supported
                best_pipeline = classifier.best_estimator_
                if best_pipeline and hasattr(best_pipeline["model"], 'predict_proba'):
                    shap_analyzer = SHAPAnalyzer(
                        best_pipeline["model"],
                        self.output_manager,
                        max_display=10,
                        max_samples=50
                    )
                    X_test_scaled = best_pipeline["scaler"].transform(classifier.X_test)
                    X_test_df = pd.DataFrame(X_test_scaled, columns=classifier.X_test.columns)
                    shap_analyzer.explain_global(X_test_df)
                else:
                    logging.info(f"Skipping SHAP for {name} (no predict_proba).")

            # 7) Compare model accuracies
            if metrics_list:
                self.data_visualizer.plot_model_comparison(metrics_list)
            else:
                logging.warning("No metrics to compare among models.")

            # 8) Save only the best model pipeline in 'trained/best'
            if self.best_pipeline is not None and self.best_model_name is not None:
                self._save_best_model_info()

            duration = time.time() - start_time
            logging.info(f"Analysis completed in {duration:.2f} seconds.")
            logging.info(f"Results stored in: {self.results_dir}")

        except Exception as e:
            logging.error(f"Analysis failed: {e}", exc_info=True)
            raise

    def _save_best_model_info(self):
        """
        Saves the best model's pipeline and info into the result folder under 'trained/best',
        thus no fallback 'misc' should occur.
        """
        try:
            model_filename = f"BEST_{self.best_model_name}_pipeline.joblib"
            info_filename = f"BEST_{self.best_model_name}_info.txt"

            # We'll store them in "trained/best" -> ensures category = "trained", subcategory = "best"
            pipeline_path = self.output_manager.get_path("trained", "best", model_filename)
            dump(self.best_pipeline, pipeline_path)
            logging.info(f"Best model '{self.best_model_name}' saved to {pipeline_path}")

            info_path = self.output_manager.get_path("trained", "best", info_filename)
            with open(info_path, "w") as f:
                f.write("=== Best Model Info ===\n\n")
                f.write(f"Model Name: {self.best_model_name}\n")
                f.write(f"Accuracy: {self.best_accuracy:.4f}\n\n")
                f.write("Trained on data:\n")
                f.write(f"  Proxy folder: {self.proxy_dir}\n")
                f.write(f"  Normal folder: {self.normal_dir}\n")
                f.write(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            logging.info(f"Best model info written to {info_path}")

        except Exception as e:
            logging.error(f"Error saving best model info: {e}", exc_info=True)

    def _calculate_folder_bytes_list(self, directory: str):
        """
        Reads the sizes (in bytes) of all .pcap files in the specified directory.
        """
        path = Path(directory)
        sizes = [f.stat().st_size for f in path.glob("*.pcap")]
        return sizes


def main():
    """
    Entry point: Just an example usage with static paths. Adjust as needed.
    """
    try:
        # Example directories
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/test/shadow_test"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/test/PROTON_test"
        results_dir = "/home/gino/PycharmProjects/myenv/ba/results_training"

        analyzer = TrafficAnalyzer(proxy_dir, normal_dir, results_dir)
        analyzer.run_analysis()

    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
