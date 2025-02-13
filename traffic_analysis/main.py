# file: main.py

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for plots

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
from traffic_analysis.data_cleaner import DataCleaner
from traffic_analysis.data_inspector import DataInspector

from joblib import dump


class TrafficAnalyzer:
    """
    Main pipeline:
      1) Copy valid PCAPs
      2) Plot PCAP size distribution
      3) NFStream feature extraction
      4) Data cleaning
      5) Model training (GridSearch optional)
      6) SHAP if available
      7) Compare metrics
      8) Save best model
    """

    def __init__(self, proxy_dir: str, normal_dir: str, results_dir: str):
        # Validate directories
        self.proxy_dir = self._validate_directory(proxy_dir)
        self.normal_dir = self._validate_directory(normal_dir)

        # Timestamp without seconds
        time_str = time.strftime('%Y-%m-%d_%H-%M')

        # Combine timestamp + folder names
        proxy_name = Path(self.proxy_dir).name
        normal_name = Path(self.normal_dir).name
        folder_name = f"{time_str}__{proxy_name}_vs_{normal_name}"

        # Create results directory
        analysis_dir = Path(results_dir) / folder_name
        analysis_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = analysis_dir

        # OutputManager
        self.output_manager = OutputManager(base_dir=str(self.results_dir))
        self.data_visualizer = DataVisualizer(self.output_manager)
        self.model_selector = ModelSelector()

        self._setup_logging()
        logging.info("TrafficAnalyzer initialized.")
        logging.info(f"Proxy folder: {self.proxy_dir}")
        logging.info(f"Normal folder: {self.normal_dir}")
        logging.info(f"Analysis folder: {self.results_dir}")

        # Clean data subfolders
        self.clean_proxy_dir = str(self.results_dir / "clean_data" / "proxy")
        self.clean_normal_dir = str(self.results_dir / "clean_data" / "normal")

        self.best_model_name = None
        self.best_accuracy = 0.0
        self.best_pipeline = None

    def _validate_directory(self, directory: str) -> str:
        """Check if directory exists."""
        p = Path(directory)
        if not p.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        return str(p.resolve())

    def _setup_logging(self):
        """Setup logging to file and console."""
        log_dir = Path(self.results_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"analysis_{time.strftime('%Y-%m-%d_%H-%M')}.log"
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        logging.info("Logging setup complete (INFO to file, WARNING+ to console).")

    def run_analysis(self):
        """Main pipeline steps."""
        try:
            start_time = time.time()
            logging.info("Starting the traffic analysis pipeline...")

            # 1) Copy PCAPs
            inspector = DataInspector(min_file_size_bytes=10000, min_flow_count=5)
            inspector.copy_valid_pcaps(self.proxy_dir, self.clean_proxy_dir)
            inspector.copy_valid_pcaps(self.normal_dir, self.clean_normal_dir)

            # 2) PCAP size distribution
            pcap_sizes = {
                "proxy": self._calculate_folder_bytes_list(self.clean_proxy_dir),
                "normal": self._calculate_folder_bytes_list(self.clean_normal_dir),
            }
            self.data_visualizer.plot_pcap_size_distribution(pcap_sizes)

            # 3) Feature extraction
            extractor = NFStreamFeatureExtractor(
                output_manager=self.output_manager,
                use_entropy=False,
                min_packets=2
            )
            df = extractor.prepare_dataset(self.clean_proxy_dir, self.clean_normal_dir)
            if df.empty:
                logging.error("No flows extracted; DataFrame is empty.")
                return
            logging.info(f"DataFrame shape after extraction: {df.shape}")

            # 4) Data cleaning
            cleaner = DataCleaner(min_packet_threshold=2, impute_with_median=True)
            df = cleaner.clean_dataset(df)
            if df.empty:
                logging.error("DataFrame empty after cleaning.")
                return

            if 'label' not in df.columns:
                logging.error("No 'label' column found, aborting.")
                return

            # Convert label to 0/1
            df['label'] = df['label'].map({'normal': 0, 'proxy': 1})
            unknown_mask = df['label'].isna()
            if unknown_mask.any():
                logging.warning("Dropping rows with unknown labels.")
                df = df[~unknown_mask]

            if df['label'].nunique() < 2:
                logging.warning("Less than two distinct labels remain. Aborting.")
                return

            inspector.check_flow_dataframe(df)

            # 5) Model training + metrics
            analyzer = FeatureAnalyzer(self.output_manager, target_column='label')
            _ = analyzer.analyze_features(df)

            models = self.model_selector.get_all_models()
            metrics_list = []

            for name, (model_obj, param_grid) in models.items():
                logging.info("==================================================")
                logging.info(f"TRAINING MODEL: {name}")
                logging.info("==================================================")

                self.output_manager.set_current_model(name)
                classifier = ScikitLearnTrafficClassifier(
                    model=model_obj,
                    output_manager=self.output_manager,
                    data_visualizer=self.data_visualizer,
                    test_size=0.2,
                    random_state=42,
                    cv_folds=5,
                    param_grid=param_grid,
                    gridsearch_scoring="accuracy"
                )
                metrics = classifier.train(df, target_column='label')
                accuracy = metrics.get("accuracy", 0.0)

                if accuracy:
                    metrics_list.append((name, accuracy))
                    logging.info(f"{name} => Accuracy: {accuracy:.3f}")
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.best_model_name = name
                        self.best_pipeline = classifier.best_estimator_
                else:
                    logging.warning(f"No accuracy reported for {name}.")

                # 6) SHAP analysis if predict_proba is available
                best_pipeline = classifier.best_estimator_
                if best_pipeline and hasattr(best_pipeline["model"], 'predict_proba'):
                    # -------------- Important FIX for FEATURE-NAMES --------------
                    # Create a DataFrame from the scaled array,
                    # SHAP can show corret column names(instead "feature_0", "feature_1", etc.)
                    X_test_scaled = best_pipeline["scaler"].transform(classifier.X_test)
                    X_test_scaled_df = pd.DataFrame(
                        X_test_scaled,
                        columns=classifier.X_test.columns
                    )
                    y_test = classifier.y_test

                    # ROC Curve
                    self.data_visualizer.plot_roc_curve(
                        model=best_pipeline["model"],
                        X_test=X_test_scaled,
                        y_test=y_test,
                        model_name=name
                    )
                    # Precision-Recall
                    self.data_visualizer.plot_precision_recall_curve(
                        model=best_pipeline["model"],
                        X_test=X_test_scaled,
                        y_test=y_test,
                        model_name=name
                    )

                    # SHAP call with DataFrame :
                    shap_analyzer = SHAPAnalyzer(
                        best_pipeline["model"],
                        self.output_manager,
                        max_display=10,
                        max_samples=50
                    )
                    shap_analyzer.explain_global(X_test_scaled_df)
                else:
                    logging.info(f"Skipping SHAP & ROC for {name} (no predict_proba).")

            # 7) Compare model metrics
            if metrics_list:
                self.data_visualizer.plot_model_comparison(metrics_list)
            else:
                logging.warning("No model metrics to compare.")

            if self.best_pipeline and self.best_model_name:
                self._save_best_model_info()

            duration = time.time() - start_time
            logging.info(f"Analysis completed in {duration:.2f} seconds.")
            logging.info(f"Results stored in: {self.results_dir}")

        except Exception as e:
            logging.error(f"Analysis failed: {e}", exc_info=True)
            sys.exit(1)

    def _save_best_model_info(self):
        """Saves best model pipeline and info."""
        try:
            model_filename = f"BEST_{self.best_model_name}_pipeline.joblib"
            info_filename = f"BEST_{self.best_model_name}_info.txt"

            pipeline_path = self.output_manager.get_path("trained", "best", model_filename)
            dump(self.best_pipeline, pipeline_path)
            logging.info(f"Best model '{self.best_model_name}' saved at: {pipeline_path}")

            info_path = self.output_manager.get_path("trained", "best", info_filename)
            with open(info_path, "w") as f:
                f.write("=== Best Model Info ===\n\n")
                f.write(f"Model: {self.best_model_name}\n")
                f.write(f"Accuracy: {self.best_accuracy:.4f}\n\n")
                f.write("Trained on data:\n")
                f.write(f"  Proxy folder: {self.proxy_dir}\n")
                f.write(f"  Normal folder: {self.normal_dir}\n")
                f.write(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            logging.info(f"Best model info written to {info_path}")

        except Exception as e:
            logging.error(f"Error saving best model info: {e}", exc_info=True)

    def _calculate_folder_bytes_list(self, directory: str):
        """Returns sizes of all *.pcap in directory."""
        path = Path(directory)
        return [f.stat().st_size for f in path.glob("*.pcap")]


def main():
    """Example usage with static paths."""
    try:
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/shadowsocks_traffic"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/non_shadowsocks_traffic"
        results_dir = "/home/gino/PycharmProjects/myenv/ba/results_training"

        analyzer = TrafficAnalyzer(proxy_dir, normal_dir, results_dir)
        analyzer.run_analysis()

    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
