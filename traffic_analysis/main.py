import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid Tkinter GUI warnings

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
from traffic_analysis.data_cleaner import DataCleaner
from traffic_analysis.data_inspector import DataInspector  # The Inspector for PCAP stats

class TrafficAnalyzer:
    """
    Main entry point for analyzing proxy vs. normal traffic.
    It removes small PCAP files, extracts features, cleans data,
    performs feature analysis, trains/evaluates models, and optionally runs SHAP analysis.
    """

    def __init__(self, proxy_dir: str, normal_dir: str, results_dir: str):
        """
        Initializes the TrafficAnalyzer with directory paths and logging setup.
        """
        self.proxy_dir = self._validate_directory(proxy_dir)
        self.normal_dir = self._validate_directory(normal_dir)

        proxy_name = Path(self.proxy_dir).name
        normal_name = Path(self.normal_dir).name

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        comparison_folder_name = f"{proxy_name}_vs_{normal_name}_{timestamp}"
        comparison_results_dir = Path(results_dir) / "comparisons" / comparison_folder_name
        comparison_results_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = comparison_results_dir

        self.output_manager = OutputManager(base_dir=str(self.results_dir))
        self.data_visualizer = DataVisualizer(self.output_manager)
        self.model_selector = ModelSelector()

        self._setup_logging()
        logging.info("TrafficAnalyzer initialized successfully.")
        logging.info(f"Comparing:\n  Proxy folder: {self.proxy_dir}\n  Normal folder: {self.normal_dir}")
        logging.info(f"Results will be saved under: {self.results_dir}")

    def _validate_directory(self, directory: str) -> str:
        """
        Ensures that the directory exists. Raises ValueError if not found.
        Returns the absolute resolved path.
        """
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        return str(path.resolve())

    def _setup_logging(self):
        """
        Configures logging to file (INFO+) and console (WARN+).
        """
        log_dir = Path(self.results_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"

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
        console_formatter = logging.Formatter('ERROR: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        logging.info("Logging setup complete. (Detailed logs in file, warnings+ on console)")

    def _calculate_folder_bytes_list(self, directory: str):
        """
        Returns a list of file sizes (in bytes) for all .pcap files in the given directory.
        """
        path = Path(directory)
        sizes = []
        for pcap_file in path.glob("*.pcap"):
            sizes.append(pcap_file.stat().st_size)
        return sizes

    def run_analysis(self):
        """
        Executes the end-to-end analysis pipeline.
        """
        try:
            start_time = time.time()
            logging.info("Starting traffic analysis pipeline...")

            # 1) (Optional) Show the distribution of .pcap file sizes
            pcap_sizes = {
                "proxy": self._calculate_folder_bytes_list(self.proxy_dir),
                "normal": self._calculate_folder_bytes_list(self.normal_dir)
            }
            self.data_visualizer.plot_pcap_size_distribution(pcap_sizes)

            # 2) Remove small PCAP files so the extractor won't pick them up
            inspector = DataInspector(min_file_size_bytes=50_000, min_flow_count=10)
            inspector.remove_small_pcaps(self.proxy_dir)
            inspector.remove_small_pcaps(self.normal_dir)

            # 3) Feature extraction (skips the removed PCAP files automatically)
            extractor = NFStreamFeatureExtractor(self.output_manager)
            df = extractor.prepare_dataset(self.proxy_dir, self.normal_dir)
            if df.empty:
                logging.warning("No data extracted. Analysis aborted.")
                return

            # 4) Clean the DataFrame (ignore columns that do not exist)
            cleaner = DataCleaner(min_packet_threshold=5, impute_with_median=True)
            df = cleaner.clean_dataset(df)
            if df.empty:
                logging.warning("After cleaning, dataset is empty. Aborting.")
                return

            # 5) Convert 'normal'/'proxy' to 0/1 if the label column exists
            if 'label' not in df.columns:
                logging.error("No 'label' column found in the dataset. Aborting.")
                return

            mapping = {'normal': 0, 'proxy': 1}
            df['label'] = df['label'].map(mapping)

            unknown_labels = df['label'].isna()
            if unknown_labels.any():
                logging.warning("Unknown labels found. Dropping affected rows.")
                df = df[~unknown_labels].copy()

            if df['label'].nunique() < 2:
                logging.warning("Less than two distinct classes found. Aborting.")
                return

            # 6) Check DataFrame columns and flow count
            inspector.check_flow_dataframe(df)

            # 7) Feature analysis
            analyzer = FeatureAnalyzer(self.output_manager)
            _ = analyzer.analyze_features(df)

            # 8) Model training and evaluation
            models = self.model_selector.get_all_models()
            metrics_list = []

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

                if "accuracy" in metrics:
                    metrics_list.append((name, metrics["accuracy"]))

                # 9) SHAP analysis if predict_proba is supported
                best_model_pipeline = classifier.best_estimator_
                if (best_model_pipeline
                        and hasattr(best_model_pipeline["model"], 'predict_proba')):
                    shap_analyzer = SHAPAnalyzer(best_model_pipeline["model"], self.output_manager)
                    X_test_scaled = best_model_pipeline["scaler"].transform(classifier.X_test)
                    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=classifier.X_test.columns)
                    shap_analyzer.explain_global(X_test_scaled_df)
                else:
                    logging.info(f"Skipping SHAP analysis for {name} (model has no predict_proba).")

            # 10) Plot model comparison
            if metrics_list:
                self.data_visualizer.plot_model_comparison(metrics_list)
            else:
                logging.info("No metrics to plot. Possibly no successful training or missing 'accuracy'.")

            duration = time.time() - start_time
            logging.info(f"Analysis completed in {duration:.2f} seconds.")
            logging.info(f"Results saved in: {self.results_dir}")

        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            raise

def main():
    """
    Main function to run the TrafficAnalyzer with example directories.
    Adjust the paths according to your actual setup.
    """
    try:
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/shadowsocks_traffic_3_cec_selenium_only_port_8388_12-08"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/selenium_filtered_traffic_3_500_2024-12-23_16-35-50"
        results_dir = "/home/gino/PycharmProjects/myenv/ba/results"

        analyzer = TrafficAnalyzer(proxy_dir, normal_dir, results_dir)
        analyzer.run_analysis()

    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
