# file: deployment/predict_shadowsocks.py

"""
Script to load a pre-trained best model (pipeline) and use it for classifying new PCAPs.
It expects two placeholders at the beginning (the path to the PCAP folder, and
the path to the model .joblib file).

Example usage:
    python predict_shadowsocks.py

It will:
    1) Extract features from the PCAPs (minimally like in your pipeline).
    2) Load the best pipeline from your .joblib file.
    3) Predict whether each flow is "shadowsocks" (label=1) or not (label=0).
    4) Print a small report about the results.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import load

# This import assumes NFStreamFeatureExtractor is in traffic_analysis/nfstream_feature_extractor.py
# Adjust if needed to match your project structure.
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor


def setup_logging():
    """
    Basic logging to stdout. You can customize as you wish.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )


def main():
    # 1) DEFINE YOUR TWO PLACEHOLDERS (PCAP folder and trained model path)
    # Adjust these paths as needed:
    pcap_dir = "/path/to/unlabeled_pcaps"    # <-- Place your PCAP directory path here
    model_path = "/path/to/BEST_XGBClassifier_pipeline.joblib"  # <-- Path to your joblib model

    setup_logging()
    logging.info("Starting Shadowsocks inference script...")

    # 2) EXTRACT FEATURES FROM NEW PCAPs
    logging.info(f"Extracting features from PCAPs in: {pcap_dir}")
    df = extract_features_from_pcaps(pcap_dir)
    if df.empty:
        logging.error("No flows extracted from the provided PCAP folder. Exiting.")
        return

    # 3) LOAD THE PRE-TRAINED PIPELINE
    pipeline = load_model_pipeline(model_path)
    if pipeline is None:
        logging.error("Failed to load pipeline. Exiting.")
        return

    # 4) MAKE PREDICTIONS
    # Ensure the extracted DataFrame has the same columns the pipeline expects.
    # If you require the same columns as training, but your DF is missing any,
    # you'd need to handle that (fill them with 0 or drop them).
    # For simplicity, we assume it matches perfectly now.

    logging.info("Predicting Shadowsocks vs. Non-Shadowsocks flows...")
    predictions = pipeline.predict(df)
    # If your best model assigned '1' => shadowsocks, '0' => not shadowsocks

    # 5) PRINT / LOG A SMALL REPORT
    ss_count = np.sum(predictions == 1)
    total = len(predictions)
    logging.info(
        f"Shadowsocks predicted flows: {ss_count} / {total} "
        f"({(ss_count/total)*100:.1f}% of flows)"
    )

    # 6) (Optional) Write out the results to a CSV
    out_file = Path(pcap_dir) / "shadowsocks_predictions.csv"
    df_out = df.copy()
    df_out['prediction'] = predictions
    df_out.to_csv(out_file, index=False)
    logging.info(f"Wrote predictions to {out_file}")

    logging.info("Inference completed successfully.")


def extract_features_from_pcaps(pcap_dir: str) -> pd.DataFrame:
    """
    Minimal usage of NFStreamFeatureExtractor to produce a DataFrame of flows
    for unlabeled PCAP files. By default, sets a label='unlabeled' which we drop.
    """
    # If you don't need OutputManager for this small script, pass None
    extractor = NFStreamFeatureExtractor(
        output_manager=None,
        use_entropy=False,
        min_packets=2
    )
    # We'll treat these PCAPs as "normal" or "proxy"? Actually no label needed;
    # we can just call extract_features once. We'll label them 'unlabeled' and drop it.
    df = extractor.extract_features(pcap_dir, label='unlabeled')
    if 'label' in df.columns:
        df.drop(columns=['label'], inplace=True)

    return df


def load_model_pipeline(model_path: str):
    """
    Loads a scikit-learn pipeline (with scaler + model).
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
