# file: deployment/flow_inference.py

"""
Script to load a pre-trained best model (pipeline) and use it for classifying new PCAPs
on a flow-by-flow basis. It creates a dedicated output folder named:
    "<dd-mm-yyyy>_<HH-mm>__flow_inference"

Inside that folder, it saves:
  - A CSV file of predictions
  - A text file summarizing paths and stats
  - A bar chart of predicted class distribution

Example usage:
    python flow_inference.py

Workflow:
    1) Extract features from PCAPs (minimally like in your pipeline).
    2) Load the best pipeline from the .joblib file.
    3) Predict each flow (label=1 => "shadowsocks", label=0 => "non-shadowsocks").
    4) Save outputs in a dedicated folder.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# Falls nötig, Pfad anpassen:
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
    # Pfade und Dateien anpassen:
    pcap_dir = "/home/gino/PycharmProjects/myenv/ba/traffic_data/unlabeled/20241229_203756"
    model_path = (
        "/home/gino/PycharmProjects/myenv/ba/model_training_results/"
        "30-12-2024_00-14__shadowsocks_traffic_20_sec_youtube_only_port_8388_500_aes_256_12-29_vs_"
        "regular_youtube_traffic_on_port_443_20s_500_12-28/20241230-001442/trained/best/"
        "BEST_RandomForestClassifier_pipeline.joblib"
    )

    # Ausgabe-Ordner => "<dd-mm-yyyy>_<HH-mm>__flow_inference"
    results_base_dir = "/home/gino/PycharmProjects/myenv/ba/inference_results"
    date_time_str = time.strftime("%d-%m-%Y_%H-%M")  # Tag-Monat-Jahr_Stunde-Minuten (ohne Sekunden)
    output_subfolder_name = f"{date_time_str}__flow_inference"
    inference_result_dir = Path(results_base_dir) / output_subfolder_name
    inference_result_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    logging.info("Starting Shadowsocks flow-based inference script...")

    # 1) Extrahiere Flow-Features aus den PCAPs
    logging.info(f"Extracting flow features from PCAPs in: {pcap_dir}")
    df = extract_features_from_pcaps(pcap_dir)
    if df.empty:
        logging.error("No flows extracted from the provided PCAP folder. Exiting.")
        return

    # 2) Lade das vortrainierte Pipeline-Modell
    pipeline = load_model_pipeline(model_path)
    if pipeline is None:
        logging.error("Could not load pipeline. Exiting.")
        return

    # 3) Vorhersage
    # Falls 'filename' nicht zum Train-Set gehört => droppen
    logging.info("Predicting Shadowsocks vs. Non-Shadowsocks flows...")
    df_for_model = df.drop(columns=["filename"], errors="ignore")
    predictions = pipeline.predict(df_for_model)

    # Zusammenfassung
    ss_count = np.sum(predictions == 1)
    total_flows = len(predictions)
    logging.info(
        f"Shadowsocks predicted flows: {ss_count} / {total_flows} "
        f"({(ss_count/total_flows)*100:.1f}% of flows)"
    )

    # 4) Speichere CSV mit allen Flows und Predictions
    df_out = df.copy()
    df_out['prediction'] = predictions
    csv_path = inference_result_dir / "shadowsocks_predictions.csv"
    df_out.to_csv(csv_path, index=False)
    logging.info(f"Predictions CSV saved to: {csv_path}")

    # 5) Erstelle eine Balkengrafik der Klassenverteilung
    logging.info("Creating class distribution plot...")
    class_labels = ["Non-Shadowsocks (0)", "Shadowsocks (1)"]
    class_counts = [np.sum(predictions == 0), np.sum(predictions == 1)]

    plt.figure(figsize=(6, 4))
    plt.bar(class_labels, class_counts, color=["skyblue", "salmon"])
    plt.title("Flow-Level Predicted Class Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Number of Flows")
    plt.tight_layout()

    plot_path = inference_result_dir / "prediction_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Class distribution plot saved: {plot_path}")

    # 6) Schreibe eine kurze Zusammenfassung
    summary_path = inference_result_dir / "inference_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Flow-Based Shadowsocks Inference Summary ===\n\n")
        f.write(f"Timestamp: {time.strftime('%d-%m-%Y_%H-%M')}\n")  # identisches Format wie Subfolder
        f.write(f"PCAP folder: {pcap_dir}\n")
        f.write(f"Model used: {model_path}\n\n")
        f.write(f"Total flows analyzed: {total_flows}\n")
        f.write(f"Predicted as Shadowsocks: {ss_count}\n")
        f.write(f"Predicted as Non-Shadowsocks: {total_flows - ss_count}\n")
        f.write(f"Percentage Shadowsocks: {(ss_count/total_flows)*100:.1f}%\n\n")
        f.write(f"Output CSV: {csv_path}\n")
        f.write(f"Distribution Plot: {plot_path}\n")

    logging.info(f"Summary file written to: {summary_path}")
    logging.info("Flow-based inference completed successfully.")


def extract_features_from_pcaps(pcap_dir: str) -> pd.DataFrame:
    """
    Uses NFStreamFeatureExtractor (with minimal config) to get a DataFrame of flows
    from the PCAPs. We keep 'filename' so we know from which PCAP each flow came,
    but drop any 'label' column (if present).
    """
    extractor = NFStreamFeatureExtractor(
        output_manager=None,
        use_entropy=False,
        min_packets=2
    )
    # Label flows as 'unlabeled' (relevant for internal usage).
    df = extractor.extract_features(pcap_dir, label='unlabeled')

    # Falls 'label' noch da ist => entfernen
    if "label" in df.columns:
        df.drop(columns=["label"], inplace=True, errors="ignore")

    return df


def load_model_pipeline(path_to_model: str):
    """
    Loads a scikit-learn pipeline (with scaler + model) from the specified .joblib file.
    """
    p = Path(path_to_model)
    if not p.exists():
        logging.error(f"Model file does not exist: {path_to_model}")
        return None

    try:
        pipeline = load(p)
        logging.info(f"Pipeline loaded from: {path_to_model}")
        return pipeline
    except Exception as e:
        logging.error(f"Error loading pipeline from {path_to_model}: {e}")
        return None


if __name__ == "__main__":
    main()
