# file: pcab_inference.py

"""
Script to load a pre-trained best model (pipeline) and classify entire PCAP files as
Shadowsocks or not, based on a flow-level majority vote.

Steps:
  1) Finds all .pcap files in pcap_dir
  2) Extracts flows (with NFStreamFeatureExtractor)
  3) Predicts each flow (0 or 1)
  4) Groups flows by 'filename' and does a majority vote => 1 label per PCAP
  5) Saves a CSV, bar chart, and text summary
  6) Output folder is named: "<dd-mm-yyyy>_<HH-mm>__pcap_inference"
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Passen: ggf. an Deinen Projektnamen anpassen:
# from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from nfstream_feature_extractor import NFStreamFeatureExtractor  # <- oder Dein Importpfad

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )

def main():
    """
    Main entry for PCAP-level classification.
    """
    # -- EDIT HIER: Pfade anpassen --
    pcap_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/unlabeled/20241229_203756"
    model_path = (
        "/home/gino/PycharmProjects/myenv/ba/model_training_results/"
        "30-12-2024_00-14__shadowsocks_traffic_20_sec_youtube_only_port_8388_500_"
        "aes_256_12-29_vs_regular_youtube_traffic_on_port_443_20s_500_12-28/"
        "20241230-001442/trained/best/"
        "BEST_RandomForestClassifier_pipeline.joblib"
    )
    results_base_dir = "/home/gino/PycharmProjects/myenv/ba/inference_results"

    # Erzeuge Ausgabeverzeichnis: "<dd-mm-yyyy>_<HH-mm>__pcap_inference"
    date_time_str = time.strftime("%d-%m-%Y_%H-%M")  # z.B. "30-12-2024_13-58"
    subfolder_name = f"{date_time_str}__pcap_inference"
    inference_dir = Path(results_base_dir) / subfolder_name
    inference_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    logging.info("Starting PCAP-level Shadowsocks inference...")

    # (1) Extrahiere Flows aus .pcap (alle .pcap im Ordner)
    logging.info(f"Extracting flows from PCAP directory: {pcap_dir}")
    df_flows = extract_pcap_flows(pcap_dir)
    if df_flows.empty:
        logging.error("No flows extracted. Cannot proceed.")
        return

    # (2) Lade das Pipeline-Modell
    pipeline = load_model_pipeline(model_path)
    if pipeline is None:
        logging.error("Could not load the pipeline. Exiting.")
        return

    # (3) Vorhersagen: Flow-Ebene
    # Wir droppen 'filename' vor dem .predict(), da es meist kein Feature ist
    features_for_model = df_flows.drop(columns=["filename"], errors="ignore")
    predictions = pipeline.predict(features_for_model)
    df_flows["prediction"] = predictions

    # (4) Aggregation pro PCAP (Mehrheitsentscheid)
    pcap_results = []
    for filename, group in df_flows.groupby("filename"):
        flow_count = len(group)
        shadow_flows = np.sum(group["prediction"] == 1)
        # Majority vote
        if shadow_flows > (flow_count / 2.0):
            pcap_label = 1
        else:
            pcap_label = 0

        pcap_results.append({
            "pcap_file": filename,
            "flow_count": flow_count,
            "shadowsocks_flows": shadow_flows,
            "pcap_label": pcap_label
        })

    df_pcap = pd.DataFrame(pcap_results)

    # (5) Speichere CSV
    csv_path = inference_dir / "pcap_level_predictions.csv"
    df_pcap.to_csv(csv_path, index=False)
    logging.info(f"Saved PCAP-level classification CSV: {csv_path}")

    zero_count = np.sum(df_pcap["pcap_label"] == 0)
    one_count = np.sum(df_pcap["pcap_label"] == 1)
    total_pcaps = len(df_pcap)
    logging.info(f"Total PCAPs => {total_pcaps}. "
                 f"Shadowsocks-labeled: {one_count}, Non-Shadowsocks: {zero_count}")

    # (6) Bar Chart der PCAP-Labels
    logging.info("Creating PCAP-level distribution plot...")
    fig_path = inference_dir / "pcap_label_distribution.png"
    label_names = ["Non-Shadowsocks (0)", "Shadowsocks (1)"]

    plt.figure(figsize=(6, 4))
    plt.bar(label_names, [zero_count, one_count], color=["skyblue", "salmon"])
    plt.title("PCAP Classification (Majority Vote)")
    plt.xlabel("Label")
    plt.ylabel("Number of PCAPs")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logging.info(f"Distribution plot saved: {fig_path}")

    # (7) Zusammenfassung als Text-Datei
    txt_path = inference_dir / "inference_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=== PCAP-Level Inference Summary ===\n\n")
        # Kein seconds-level timestamp
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"PCAP folder: {pcap_dir}\n")
        f.write(f"Model used: {model_path}\n\n")
        f.write(f"Total PCAPs: {total_pcaps}\n")
        f.write(f"Shadowsocks-labeled: {one_count}\n")
        f.write(f"Non-Shadowsocks-labeled: {zero_count}\n\n")
        f.write(f"CSV output: {csv_path}\n")
        f.write(f"Distribution plot: {fig_path}\n")

    logging.info(f"Text summary saved: {txt_path}")
    logging.info("PCAP-level inference completed successfully.")

def extract_pcap_flows(pcap_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Minimal usage of NFStreamFeatureExtractor to produce a DataFrame of flows
    for unlabeled PCAP files. By default, sets label='unlabeled'.
    """
    extractor = NFStreamFeatureExtractor(
        output_manager=None,  # crucial: no OutputManager
        use_entropy=False,
        min_packets=2
    )
    df = extractor.extract_features(pcap_dir, label="unlabeled")

    # Drop 'label' Spalte, aber behalte 'filename'
    if 'label' in df.columns:
        df.drop(columns=['label'], inplace=True, errors='ignore')
    return df

def load_model_pipeline(path_to_model: str):
    """
    Loads a scikit-learn pipeline (with scaler + model) from the specified .joblib file.
    """
    p = Path(path_to_model)
    if not p.exists():
        logging.error(f"Model file not found: {path_to_model}")
        return None
    try:
        model = load(p)
        logging.info(f"Model pipeline loaded from: {path_to_model}")
        return model
    except Exception as ex:
        logging.error(f"Error loading pipeline from {path_to_model}: {ex}")
        return None

if __name__ == "__main__":
    main()
