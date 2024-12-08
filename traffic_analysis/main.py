import time
import sys
import os
from pathlib import Path
import logging
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import ModelSelector
from traffic_analysis.feature_analyzer import FeatureAnalyzer
from traffic_analysis.data_visualizer import DataVisualizer
from traffic_analysis.output_manager import OutputManager
from traffic_analysis.shap_analyzer import SHAPAnalyzer


class TrafficAnalyzer:
    def __init__(self, proxy_dir: str, normal_dir: str, results_dir: str):
        self.proxy_dir = self._validate_directory(proxy_dir)
        self.normal_dir = self._validate_directory(normal_dir)

        # Erstelle Results-Verzeichnis wenn nicht vorhanden
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Manager und Analyzer initialisieren
        self.output_manager = OutputManager(base_dir=str(self.results_dir))
        self.data_visualizer = DataVisualizer(self.output_manager)
        self.model_selector = ModelSelector()

        # Logging Setup
        self._setup_logging()

    def _validate_directory(self, directory: str) -> str:
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        return str(path.absolute())

    def _setup_logging(self):
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def run_analysis(self):
        try:
            start_time = time.time()

            # Extract features
            logging.info("Extracting features...")
            extractor = NFStreamFeatureExtractor(self.output_manager)
            df = extractor.prepare_dataset(self.proxy_dir, self.normal_dir)

            # Analyze features
            logging.info("Analyzing features...")
            analyzer = FeatureAnalyzer(self.output_manager)
            analysis_results = analyzer.analyze_features(df)

            # Train and evaluate models
            models = self.model_selector.get_all_models()
            for name, model in models.items():
                logging.info(f"Testing {name}...")
                self.output_manager.set_current_model(name)

                classifier = ScikitLearnTrafficClassifier(
                    model=model,
                    output_manager=self.output_manager,
                    data_visualizer=self.data_visualizer
                )

                metrics = classifier.train(df)

                # SHAP Analysis wenn möglich
                if hasattr(model, 'predict_proba'):
                    shap_analyzer = SHAPAnalyzer(classifier.model, self.output_manager)
                    shap_analyzer.explain_global(classifier.X_test_scaled)

            duration = time.time() - start_time
            logging.info(f"Analysis completed in {duration:.2f} seconds")
            logging.info(f"Results saved in: {self.results_dir}")

        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            raise


def main():
    try:
        # Konfiguration
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/shadowsocks_traffic_3_cec_selenium_only_port_8388_12-08"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/firefox_without_proxy_2024-12-08"
        results_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_analysis/results"

        # Analyse durchführen
        analyzer = TrafficAnalyzer(
            str(proxy_dir),
            str(normal_dir),
            str(results_dir)
        )
        analyzer.run_analysis()

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()