# file: combined_inference.py

"""
Script that performs both:
 1) Flow-level inference (predict each flow),
 2) PCAP-level inference (majority vote per PCAP).

It creates one output folder named:
    "<dd-mm-yyyy>_<HH-mm>__combined_inference"

Inside that folder, it saves:
  - A CSV with flow-level predictions (flow_inference.csv)
  - A bar chart of flow-level distribution (flow_distribution.png)
  - A CSV with PCAP-level predictions (pcap_inference.csv)
  - A bar chart of PCAP-level distribution (pcap_distribution.png)
  - A text file summarizing all results (inference_summary.txt)

Example usage:
    python combined_inference.py
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Passe den Pfad ggf. an, falls dein Code woanders liegt:
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor


def setup_logging():
    """
    Basic logging setup to stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )


def main():
    # ============================
    # 1) Deine Pfade und Dateien
    # ============================
    pcap_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/unlabeled/20241229_203756"
    model_path = ("/home/gino/PycharmProjects/myenv/ba/results_training/shadowsocks_traffic_20_sec_youtube_only_port_8388_500_aes_256_12-29_vs_regular_youtube_traffic_on_port_443_20s_500_12-28_30-12-2024_23-21/20241230-232110/trained/best/BEST_RandomForestClassifier_pipeline.joblib")
    results_base_dir = "/home/gino/PycharmProjects/myenv/ba/inference_results"
    # Ordnername: "<dd-mm-yyyy>_<HH-mm>__combined_inference" (ohne Sekunden)
    date_time_str = time.strftime("%d-%m-%Y_%H-%M")
    combined_subfolder = f"{date_time_str}__combined_inference"
    out_dir = Path(results_base_dir) / combined_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    logging.info("Starting combined (flow + PCAP) inference...")

    # ============================
    # 2) Feature-Extraktion (Flows)
    # ============================
    logging.info(f"Extracting flow features from PCAPs in: {pcap_dir}")
    df_flows = extract_flow_features(pcap_dir)
    if df_flows.empty:
        logging.error("No flows extracted. Aborting.")
        return

    # ============================
    # 3) Vortrainiertes Modell laden
    # ============================
    pipeline = load_model_pipeline(model_path)
    if pipeline is None:
        logging.error("Pipeline could not be loaded. Exiting.")
        return

    # ============================
    # 4) Flow-Level Inference
    # ============================
    logging.info("Predicting each flow => Shadowsocks or not (0/1)...")
    # Spalten wie 'filename' wegwerfen, falls nicht im Training enthalten
    model_input = df_flows.drop(columns=["filename"], errors="ignore")
    flow_preds = pipeline.predict(model_input)
    df_flows["prediction"] = flow_preds

    total_flows = len(flow_preds)
    ss_flows = np.sum(flow_preds == 1)
    logging.info(
        f"Flow-level => {ss_flows} / {total_flows} labeled as Shadowsocks "
        f"({(ss_flows / total_flows) * 100:.1f}% of flows)."
    )

    # 4a) Flows => CSV
    flow_csv_path = out_dir / "flow_inference.csv"
    df_flows.to_csv(flow_csv_path, index=False)
    logging.info(f"Flow-level CSV saved: {flow_csv_path}")

    # 4b) Flow-level Balkendiagramm
    logging.info("Creating flow-level distribution plot...")
    flow_counts = [np.sum(flow_preds == 0), np.sum(flow_preds == 1)]
    flow_labels = ["Non-Shadowsocks (0)", "Shadowsocks (1)"]

    plt.figure(figsize=(6, 4))
    plt.bar(flow_labels, flow_counts, color=["skyblue", "salmon"])
    plt.title("Flow-Level Classification")
    plt.xlabel("Predicted Class")
    plt.ylabel("Number of Flows")
    plt.tight_layout()

    flow_plot_path = out_dir / "flow_distribution.png"
    plt.savefig(flow_plot_path, dpi=300)
    plt.close()
    logging.info(f"Flow-level distribution plot saved: {flow_plot_path}")

    # ============================
    # 5) PCAP-Level Inference
    # ============================
    logging.info("Aggregating flows per PCAP with majority vote...")

    pcap_rows = []
    for filename, group in df_flows.groupby("filename"):
        flow_count = len(group)
        shadow_flows = np.sum(group["prediction"] == 1)
        # Majority
        if shadow_flows > (flow_count / 2):
            pcap_label = 1
        else:
            pcap_label = 0

        pcap_rows.append({
            "pcap_file": filename,
            "flow_count": flow_count,
            "shadowsocks_flows": shadow_flows,
            "pcap_label": pcap_label
        })

    df_pcap = pd.DataFrame(pcap_rows)
    total_pcaps = len(df_pcap)
    pcaps_shadowsocks = np.sum(df_pcap["pcap_label"] == 1)
    pcaps_non_shadow = total_pcaps - pcaps_shadowsocks
    logging.info(
        f"PCAP-level => {pcaps_shadowsocks} / {total_pcaps} labeled as Shadowsocks "
        f"({(pcaps_shadowsocks / total_pcaps) * 100:.1f}% of PCAPs)."
    )

    # 5a) PCAP => CSV
    pcap_csv_path = out_dir / "pcap_inference.csv"
    df_pcap.to_csv(pcap_csv_path, index=False)
    logging.info(f"PCAP-level CSV saved: {pcap_csv_path}")

    # 5b) PCAP-Level Balkendiagramm
    logging.info("Creating pcap-level distribution plot...")
    pcap_counts = [pcaps_non_shadow, pcaps_shadowsocks]
    pcap_labels = ["Non-Shadowsocks (0)", "Shadowsocks (1)"]

    plt.figure(figsize=(6, 4))
    plt.bar(pcap_labels, pcap_counts, color=["skyblue", "salmon"])
    plt.title("PCAP-Level Classification (Majority Vote)")
    plt.xlabel("Label")
    plt.ylabel("Number of PCAPs")
    plt.tight_layout()

    pcap_plot_path = out_dir / "pcap_distribution.png"
    plt.savefig(pcap_plot_path, dpi=300)
    plt.close()
    logging.info(f"PCAP-level distribution plot saved: {pcap_plot_path}")

    # ============================
    # 6) Zusammenfassungsdatei
    # ============================
    summary_txt = out_dir / "inference_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("=== Combined Inference (Flow + PCAP) ===\n\n")
        f.write(f"Timestamp: {time.strftime('%d-%m-%Y %H:%M')}\n")
        f.write(f"PCAP folder: {pcap_dir}\n")
        f.write(f"Model used: {model_path}\n\n")

        # Flow-level summary
        f.write("Flow-level results:\n")
        f.write(f" - Total flows: {total_flows}\n")
        f.write(f" - Shadowsocks flows: {ss_flows}\n")
        f.write(f" - Non-Shadowsocks flows: {total_flows - ss_flows}\n")
        f.write(f" - Percentage Shadowsocks: {(ss_flows / total_flows) * 100:.1f}%\n\n")

        # PCAP-level summary
        f.write("PCAP-level results (majority vote):\n")
        f.write(f" - Total PCAPs: {total_pcaps}\n")
        f.write(f" - Shadowsocks-labeled PCAPs: {pcaps_shadowsocks}\n")
        f.write(f" - Non-Shadowsocks-labeled PCAPs: {pcaps_non_shadow}\n")
        f.write(f" - Percentage Shadowsocks: {(pcaps_shadowsocks / total_pcaps) * 100:.1f}%\n\n")

        # Output references
        f.write("Output files:\n")
        f.write(f" - Flow CSV: {flow_csv_path}\n")
        f.write(f" - Flow distribution plot: {flow_plot_path}\n")
        f.write(f" - PCAP CSV: {pcap_csv_path}\n")
        f.write(f" - PCAP distribution plot: {pcap_plot_path}\n")
        f.write(f" - Summary text: {summary_txt}\n")

    logging.info(f"Summary text written: {summary_txt}")
    logging.info("Combined inference completed successfully.")


def extract_flow_features(pcap_dir: str) -> pd.DataFrame:
    """
    Helper: Extract flows from PCAP(s) using NFStreamFeatureExtractor.
    Keeps 'filename' for grouping, drops 'label' (unlabeled).
    """
    extractor = NFStreamFeatureExtractor(
        output_manager=None,
        use_entropy=False,
        min_packets=2
    )
    df = extractor.extract_features(pcap_dir, label='unlabeled')
    if "label" in df.columns:
        df.drop(columns=["label"], inplace=True, errors="ignore")
    return df


def load_model_pipeline(model_path: str):
    """
    Loads a scikit-learn pipeline from a .joblib file.
    """
    p = Path(model_path)
    if not p.exists():
        logging.error(f"Model not found: {model_path}")
        return None
    try:
        model = load(p)
        logging.info(f"Model pipeline loaded from: {model_path}")
        return model
    except Exception as ex:
        logging.error(f"Error loading pipeline from {model_path}: {ex}")
        return None


if __name__ == "__main__":
    main()
