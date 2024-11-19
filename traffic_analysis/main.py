import time
import sys
import os
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import MODELS
from traffic_analysis.feature_analyzer import FeatureAnalyzer
from traffic_analysis.data_visualizer import DataVisualizer


def main():
    try:
        start_time = time.time()
        proxy_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/nov_1_traffic_PROXY"
        normal_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/nov_2_proton_traffic"

        # Create results directory if it doesn't exist
        os.makedirs("analysis_results", exist_ok=True)
        os.makedirs("nfstream_results", exist_ok=True)

        # Extract features
        extractor = NFStreamFeatureExtractor()
        df = extractor.prepare_dataset(proxy_dir, normal_dir)

        # Run feature analysis
        print("\nStarting feature analysis...")
        analyzer = FeatureAnalyzer()
        analysis_results = analyzer.analyze_features(df)

        # Optional: Detailed feature contribution analysis
        contribution_results = analyzer.analyze_feature_contribution(df)

        # Create visualizer for model comparison
        visualizer = DataVisualizer()

        # Train and evaluate all models
        for model_name, model in MODELS.items():
            print(f"\n===== Testing {model_name} =====")
            classifier = ScikitLearnTrafficClassifier(model)
            classifier.train(df)

        # Generate final comparison visualizations
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join("analysis_results", f"model_comparison_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        visualizer.plot_comprehensive_comparison(output_dir, timestamp)
        visualizer.save_comparison_table(output_dir, timestamp)

        print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
        print("\nResults have been saved in directories 'analysis_results' and 'nfstream_results'")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()