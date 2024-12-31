# file: pcap_inference.py

"""
Loads a trained model pipeline and classifies entire PCAP files as Shadowsocks or not,
using a flow-level majority vote. Saves:
  - A CSV with PCAP-level labels
  - A bar chart of predicted PCAP distribution
  - A text summary

Steps:
  1) Scan pcap_dir for .pcap files
  2) Extract flows via NFStreamFeatureExtractor
  3) Predict each flow (0 or 1)
  4) Group flows by 'filename' and do majority vote => 1 label per PCAP
  5) Save CSV, bar chart, and summary text
  6) Output folder is "<dd-mm-yyyy>_<HH-mm>__pcap_inference"
"""

import logging
import sys
import time
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Use your actual import path:
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor


def setup_logging():
    """Sets up basic logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )


def main():
    """Main entry for PCAP-level classification."""
    # Edit these paths as needed:
    pcap_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/unlabeled/20241229_203756"
    model_path = ("/home/gino/PycharmProjects/myenv/ba/results_training/shadowsocks_traffic_20_sec_youtube_only_port_8388_500_aes_256_12-29_vs_regular_youtube_traffic_on_port_443_20s_500_12-28_30-12-2024_23-21/20241230-232110/trained/best/RandomForest_youtube_20s_500.joblib")
    results_base_dir = "/home/gino/PycharmProjects/myenv/ba/inference_results"

    # Create output folder "<dd-mm-yyyy>_<HH-mm>__pcap_inference"
    timestamp = time.strftime("%d-%m-%Y_%H-%M")
    subfolder_name = f"{timestamp}__pcap_inference"
    inference_dir = Path(results_base_dir) / subfolder_name
    inference_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    logging.info("Starting PCAP-level Shadowsocks inference...")

    # 1) Extract flows for all PCAPs in pcap_dir
    logging.info(f"Extracting flows from: {pcap_dir}")
    df_flows = extract_pcap_flows(pcap_dir)
    if df_flows.empty:
        logging.error("No flows extracted. Exiting.")
        return

    # 2) Load model
    pipeline = load_model_pipeline(model_path)
    if pipeline is None:
        logging.error("Cannot load pipeline. Exiting.")
        return

    # 3) Flow-level predictions
    features_for_model = df_flows.drop(columns=["filename"], errors="ignore")
    predictions = pipeline.predict(features_for_model)
    df_flows["prediction"] = predictions

    # 4) Majority vote per PCAP
    pcap_results = []
    for filename, group in df_flows.groupby("filename"):
        flow_count = len(group)
        shadow_count = np.sum(group["prediction"] == 1)
        if shadow_count > (flow_count / 2.0):
            pcap_label = 1
        else:
            pcap_label = 0
        pcap_results.append({
            "pcap_file": filename,
            "flow_count": flow_count,
            "shadowsocks_flows": shadow_count,
            "pcap_label": pcap_label
        })

    df_pcap = pd.DataFrame(pcap_results)

    # Save CSV
    csv_path = inference_dir / "pcap_level_predictions.csv"
    df_pcap.to_csv(csv_path, index=False)
    logging.info(f"Saved pcap-level CSV: {csv_path}")

    # Basic stats
    total_pcaps = len(df_pcap)
    ss_count = np.sum(df_pcap["pcap_label"] == 1)
    non_ss_count = total_pcaps - ss_count
    logging.info(f"Total PCAPs: {total_pcaps}, Shadowsocks: {ss_count}, Non-Shadowsocks: {non_ss_count}")

    # Bar chart
    logging.info("Creating pcap-level distribution plot...")
    fig_path = inference_dir / "pcap_label_distribution.png"
    labels = ["Non-Shadowsocks (0)", "Shadowsocks (1)"]
    values = [non_ss_count, ss_count]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["skyblue", "salmon"])
    plt.title("PCAP Classification (Majority Vote)")
    plt.xlabel("Label")
    plt.ylabel("Number of PCAPs")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logging.info(f"Distribution plot saved: {fig_path}")

    # Summary text
    txt_path = inference_dir / "inference_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=== PCAP-Level Inference Summary ===\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"PCAP folder: {pcap_dir}\n")
        f.write(f"Model used: {model_path}\n\n")
        f.write(f"Total PCAPs: {total_pcaps}\n")
        f.write(f"Shadowsocks-labeled: {ss_count}\n")
        f.write(f"Non-Shadowsocks-labeled: {non_ss_count}\n")
        f.write(f"CSV output: {csv_path}\n")
        f.write(f"Distribution plot: {fig_path}\n")

    logging.info("PCAP-level inference completed.")


def extract_pcap_flows(pcap_dir: Union[str, Path]) -> pd.DataFrame:
    """Extract flows with NFStreamFeatureExtractor."""
    extractor = NFStreamFeatureExtractor(
        output_manager=None,
        use_entropy=False,
        min_packets=2
    )
    df = extractor.extract_features(pcap_dir, label="unlabeled")
    if "label" in df.columns:
        df.drop(columns=["label"], inplace=True, errors="ignore")
    return df


def load_model_pipeline(path: str):
    """Load scikit-learn pipeline from .joblib."""
    p = Path(path)
    if not p.exists():
        logging.error(f"Model file not found: {path}")
        return None
    try:
        model = load(p)
        logging.info(f"Model pipeline loaded from: {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading pipeline: {e}")
        return None


if __name__ == "__main__":
    main()
