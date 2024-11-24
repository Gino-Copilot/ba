import time
import sys
import os
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import MODELS
from traffic_analysis.feature_analyzer import FeatureAnalyzer
from traffic_analysis.data_visualizer import DataVisualizer
from traffic_analysis.output_manager import OutputManager


def main():
    try:
        start_time = time.time()
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/nov_1_traffic_PROXY"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/nov_2_proton_traffic"

        # Initialisiere OutputManager
        output_manager = OutputManager()

        # Extract features
        extractor = NFStreamFeatureExtractor(output_manager)
        df = extractor.prepare_dataset(proxy_dir, normal_dir)

        # Run feature analysis
        print("\nStarting feature analysis...")
        analyzer = FeatureAnalyzer(output_manager)
        analysis_results = analyzer.analyze_features(df)

        # Create visualizer for model comparison
        visualizer = DataVisualizer(output_manager)

        # Train and evaluate all models
        for model_name, model in MODELS.items():
            print(f"\n===== Testing {model_name} =====")
            output_manager.set_current_model(model_name)
            classifier = ScikitLearnTrafficClassifier(model, output_manager)
            classifier.train(df)

        print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
        print(f"\nResults have been saved in: {output_manager.base_dir}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
