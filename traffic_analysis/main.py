# main.py

import sys
import time
import logging
from pathlib import Path

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
    Orchestrates the complete traffic analysis workflow:
      1) Extract NFStream features
      2) Perform feature analysis
      3) Train and evaluate models
      4) Perform (optional) SHAP analysis
    """

    def __init__(self, proxy_dir: str, normal_dir: str, results_dir: str):
        """
        Initializes the TrafficAnalyzer with directory paths and output management.

        Args:
            proxy_dir (str): Path to directory containing proxy (Shadowsocks, etc.) traffic PCAP files.
            normal_dir (str): Path to directory containing normal traffic PCAP files.
            results_dir (str): Path to output directory for logs, results, and other artifacts.
        """
        self.proxy_dir = self._validate_directory(proxy_dir)
        self.normal_dir = self._validate_directory(normal_dir)
        self.results_dir = Path(results_dir)

        # Create and configure the output manager
        self.output_manager = OutputManager(base_dir=str(self.results_dir))
        self.data_visualizer = DataVisualizer(self.output_manager)
        self.model_selector = ModelSelector()

        # Set up logging
        self._setup_logging()
        logging.info("TrafficAnalyzer initialized successfully.")

    def _validate_directory(self, directory: str) -> str:
        """
        Ensures the directory exists. Raises ValueError if it does not.

        Args:
            directory (str): Directory path to be validated.

        Returns:
            str: Absolute path of the validated directory.
        """
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        return str(path.resolve())

    def _setup_logging(self):
        """
        Initializes basic logging to both file and console.
        """
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging setup complete.")

    def run_analysis(self):
        """
        Runs the complete analysis pipeline:
          - Feature extraction from PCAPs
          - Feature analysis
          - Model training and evaluation
          - SHAP analysis (optional, if predict_proba is available)
        """
        try:
            start_time = time.time()
            logging.info("Starting the traffic analysis pipeline...")

            # 1) Extract features
            extractor = NFStreamFeatureExtractor(self.output_manager)
            df = extractor.prepare_dataset(self.proxy_dir, self.normal_dir)

            if df.empty:
                logging.warning("No data extracted. Analysis will be aborted.")
                return

            # 2) Analyze features
            analyzer = FeatureAnalyzer(self.output_manager)
            _ = analyzer.analyze_features(df)  # analysis results can be stored or returned

            # 3) Train and evaluate models
            models = self.model_selector.get_all_models()
            for name, model in models.items():
                logging.info(f"Training and testing model: {name}")
                self.output_manager.set_current_model(name)

                classifier = ScikitLearnTrafficClassifier(
                    model=model,
                    output_manager=self.output_manager,
                    data_visualizer=self.data_visualizer
                )
                metrics = classifier.train(df)

                # 4) SHAP analysis (if predict_proba is supported by the model)
                if hasattr(model, 'predict_proba'):
                    shap_analyzer = SHAPAnalyzer(classifier.model, self.output_manager)
                    shap_analyzer.explain_global(classifier.X_test_scaled)
                else:
                    logging.info(f"Skipping SHAP analysis for {name} (no predict_proba).")

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
        # Hard-coded paths can be replaced by CLI arguments or config files if desired.
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
