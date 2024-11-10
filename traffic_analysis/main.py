import time
import sys
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import MODELS


def main():
    try:
        start_time = time.time()
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/nov_1_traffic_PROXY"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/nov_1traffic"

        # Feature-Extraktion
        extractor = NFStreamFeatureExtractor()
        df = extractor.prepare_dataset(proxy_dir, normal_dir)

        # Teste jedes Modell
        for model_name, model in MODELS.items():
            print(f"\n===== Testing {model_name} =====")
            classifier = ScikitLearnTrafficClassifier(model)
            classifier.train(df)
            classifier.save_results(df)

        print(f"Analyse abgeschlossen in {time.time() - start_time:.2f} Sekunden")

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
