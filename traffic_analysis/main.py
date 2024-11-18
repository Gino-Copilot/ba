import time
import sys
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import MODELS
from traffic_analysis.feature_analyzer import FeatureAnalyzer


def main():
    try:
        start_time = time.time()
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/oct_26_ss_PROXY_traffic_sel"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/oct_26_ss_traffic_sel"

        # Extract features
        extractor = NFStreamFeatureExtractor()
        df = extractor.prepare_dataset(proxy_dir, normal_dir)

        # Run feature analysis
        print("\nStarting feature analysis...")
        analyzer = FeatureAnalyzer()
        analysis_results = analyzer.analyze_features(df)

        # Optional: Detailed feature contribution analysis
        contribution_results = analyzer.analyze_feature_contribution(df)

        # Train all models
        for model_name, model in MODELS.items():
            print(f"\n===== Testing {model_name} =====")
            classifier = ScikitLearnTrafficClassifier(model)
            classifier.train(df)
            classifier.save_results(df)

        print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
        print("\nResults have been saved in directories 'analysis_results' and 'nfstream_results'")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
