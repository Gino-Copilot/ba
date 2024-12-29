# file: deployment/predict_shadowsocks.py

"""
Script to load a pre-trained best model (pipeline) and use it for classifying new PCAPs.
It creates a dedicated inference results folder (time-stamped) containing:
    - A CSV file of predictions
    - A text file summarizing the paths used
    - A bar chart of predicted class distribution

Example usage:
    python predict_shadowsocks.py

Workflow:
    1) Extracts features from the PCAPs (minimally like in the pipeline).
    2) Loads the best pipeline from the .joblib file.
    3) Predicts whether each flow is "shadowsocks" (label=1) or not (label=0).
    4) Saves outputs (CSV, figure, summary) in a dedicated folder.
"""

import logging
import sys
import time
import os
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# This import assumes NFStreamFeatureExtractor is in traffic_analysis/nfstream_feature_extractor.py
# Adjust if needed to match your project structure.
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor


def setup_logging():
    """
    Basic logging to stdout. Can be customized as needed.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )


def main():
    # 1) DEFINE YOUR TWO PLACEHOLDERS (PCAP folder and trained model path)
    # Adjust these paths as needed:
    pcap_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/unlabeled/20241229_203756"
    model_path = "/home/gino/PycharmProjects/myenv/ba/model_training_results/shadowsocks_traffic_20_sec_selenium_only_port_8388_500_aes_128_12-28_vs_regular_youtube_traffic_on_port_443_20s_500_12-28_28-12-2024_15-17/20241228-151729/trained/best/BEST_RandomForestClassifier_pipeline.joblib"

    # 2) DEFINE WHERE THE INFERENCE RESULTS WILL BE SAVED
    # A base directory can be set; a time-stamped folder is created inside.
    results_base_dir = "/home/gino/PycharmProjects/myenv/ba/inference_results"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    inference_result_dir = Path(results_base_dir) / f"inference_{timestamp}"
    inference_result_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    logging.info("Starting Shadowsocks inference script...")

    # 3) EXTRACT FEATURES FROM NEW PCAPs
    logging.info(f"Extracting features from PCAPs in: {pcap_dir}")
    df = extract_features_from_pcaps(pcap_dir)
    if df.empty:
        logging.error("No flows extracted from the provided PCAP folder. Exiting.")
        return

    # 4) LOAD THE PRE-TRAINED PIPELINE
    pipeline = load_model_pipeline(model_path)
    if pipeline is None:
        logging.error("Failed to load pipeline. Exiting.")
        return

    # 5) MAKE PREDICTIONS
    # Ensure the extracted DataFrame has the same columns the pipeline expects
    logging.info("Predicting Shadowsocks vs. Non-Shadowsocks flows...")
    predictions = pipeline.predict(df)

    # 6) LOG AND SAVE PREDICTION SUMMARY
    ss_count = np.sum(predictions == 1)
    total = len(predictions)
    logging.info(
        f"Shadowsocks predicted flows: {ss_count} / {total} "
        f"({(ss_count/total)*100:.1f}% of flows)"
    )

    # 7) SAVE PREDICTIONS CSV
    predictions_csv_path = inference_result_dir / "shadowsocks_predictions.csv"
    df_out = df.copy()
    df_out['prediction'] = predictions
    df_out.to_csv(predictions_csv_path, index=False)
    logging.info(f"Saved predictions CSV to: {predictions_csv_path}")

    # 8) CREATE A SIMPLE BAR CHART FOR PREDICTION DISTRIBUTION
    logging.info("Creating class distribution plot...")
    class_labels = ['Non-Shadowsocks (0)', 'Shadowsocks (1)']
    counts = [np.sum(predictions == 0), np.sum(predictions == 1)]

    plt.figure(figsize=(6, 4))
    plt.bar(class_labels, counts, color=['skyblue', 'salmon'])
    plt.title("Predicted Class Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Number of Flows")
    plt.tight_layout()

    plot_path = inference_result_dir / "prediction_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Class distribution plot saved to: {plot_path}")

    # 9) SAVE A SUMMARY TXT FILE
    summary_path = inference_result_dir / "inference_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Shadowsocks Inference Summary ===\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PCAP folder: {pcap_dir}\n")
        f.write(f"Model used: {model_path}\n\n")
        f.write(f"Total flows analyzed: {total}\n")
        f.write(f"Predicted as Shadowsocks: {ss_count}\n")
        f.write(f"Predicted as Non-Shadowsocks: {total - ss_count}\n")
        f.write(f"Percentage Shadowsocks: {(ss_count/total)*100:.1f}%\n\n")
        f.write(f"Output CSV: {predictions_csv_path}\n")
        f.write(f"Distribution Plot: {plot_path}\n")

    logging.info(f"Created summary file: {summary_path}")
    logging.info("Inference completed successfully.")


def extract_features_from_pcaps(pcap_dir: str) -> pd.DataFrame:
    """
    Minimal usage of NFStreamFeatureExtractor to produce a DataFrame of flows
    for unlabeled PCAP files. By default, sets a label='unlabeled' which is dropped.
    """
    # Passing None for output_manager since we only need minimal extraction here.
    extractor = NFStreamFeatureExtractor(
        output_manager=None,
        use_entropy=False,
        min_packets=2
    )
    # Label flows as 'unlabeled' and drop that column afterward.
    df = extractor.extract_features(pcap_dir, label='unlabeled')
    if 'label' in df.columns:
        df.drop(columns=['label'], inplace=True)
    return df


def load_model_pipeline(model_path: str):
    """
    Loads a scikit-learn pipeline (with scaler + model) from the specified .joblib file.
    """
    p = Path(model_path)
    if not p.exists():
        logging.error(f"Model file does not exist: {model_path}")
        return None

    try:
        pipeline = load(p)
        logging.info(f"Model pipeline loaded from: {model_path}")
        return pipeline
    except Exception as e:
        logging.error(f"Error loading pipeline from {model_path}: {e}")
        return None


if __name__ == "__main__":
    main()
