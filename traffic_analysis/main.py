import time
import sys
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import MODELS
from traffic_analysis.feature_analyzer import FeatureAnalyzer  # Neue Import-Zeile


def main():
    try:
        start_time = time.time()
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/nov_1_traffic_PROXY"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/nov_1traffic"

        # Feature-Extraktion
        extractor = NFStreamFeatureExtractor()
        df = extractor.prepare_dataset(proxy_dir, normal_dir)

        # Feature-Analyse durchführen
        print("\nStarte Feature-Analyse...")
        analyzer = FeatureAnalyzer()
        analysis_results = analyzer.analyze_features(df)

        # Optional: Detaillierte Feature-Beitragsanalyse
        contribution_results = analyzer.analyze_feature_contribution(df)

        # Modell-Training wie gehabt
        for model_name, model in MODELS.items():
            print(f"\n===== Testing {model_name} =====")
            classifier = ScikitLearnTrafficClassifier(model)
            classifier.train(df)
            classifier.save_results(df)

        print(f"\nAnalyse abgeschlossen in {time.time() - start_time:.2f} Sekunden")
        print("\nErgebnisse wurden in den Verzeichnissen 'analysis_results' und 'nfstream_results' gespeichert.")

    except Exception as e:
        print(f"Fehler aufgetreten: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()