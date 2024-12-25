import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Any

import nfstream
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Add tqdm for progress bars
from tqdm import tqdm


@dataclass
class FlowFeatures:
    duration_seconds: float
    packets_per_second: float
    bytes_per_second: float
    src2dst_packets_per_second: float
    dst2src_packets_per_second: float
    src2dst_bytes_per_second: float
    dst2src_bytes_per_second: float
    packet_size_avg: float
    packet_ratio: float
    byte_ratio: float
    entropy_features: Optional[Dict[str, float]] = None


class NFStreamFeatureExtractor:
    """
    Extracts features from PCAP files in a given directory (proxy or normal)
    using the NFStream library.
    """

    def __init__(self, output_manager, use_entropy: bool = False, min_packets: int = 5):
        """
        Args:
            output_manager: An instance of OutputManager for handling output paths.
            use_entropy: Whether to calculate entropy-based features or not.
            min_packets: Minimum bidirectional packets required to consider a flow valid.
        """
        self.output_manager = output_manager
        self.use_entropy = use_entropy
        self.min_packets = min_packets
        logging.info(f"NFStreamFeatureExtractor initialized with min_packets={min_packets}")

    def prepare_dataset(self, proxy_dir: Union[str, Path], normal_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Orchestrates feature extraction for both proxy and normal directories,
        then combines them into one DataFrame.

        Steps:
          1) Extract features from the proxy directory (label='proxy').
          2) Extract features from the normal directory (label='normal').
          3) Concatenate the two DataFrames.
          4) Drop Inf/NaN values.
          5) Save combined CSV and summary.

        Args:
            proxy_dir: Path to the directory containing .pcap files labeled "proxy".
            normal_dir: Path to the directory containing .pcap files labeled "normal".

        Returns:
            A combined DataFrame of all extracted flows with a 'label' column.
            If both are empty, returns an empty DataFrame.
        """
        try:
            logging.info("Starting dataset preparation...")
            logging.info(f"Processing proxy directory: {proxy_dir}")
            logging.info(f"Processing normal directory: {normal_dir}")

            # Extract features for proxy traffic
            proxy_df = self.extract_features(proxy_dir, label='proxy')
            logging.info(f"Proxy features extracted. Shape: {proxy_df.shape if not proxy_df.empty else 'Empty'}")
            if not proxy_df.empty:
                self._save_csv(proxy_df, "nfstream", "processed", "proxy_features.csv")
                logging.info(f"Sample of proxy data:\n{proxy_df.head()}")
            else:
                logging.warning("No proxy features extracted (DataFrame is empty)!")

            # Extract features for normal traffic
            normal_df = self.extract_features(normal_dir, label='normal')
            logging.info(f"Normal features extracted. Shape: {normal_df.shape if not normal_df.empty else 'Empty'}")
            if not normal_df.empty:
                self._save_csv(normal_df, "nfstream", "processed", "normal_features.csv")
                logging.info(f"Sample of normal data:\n{normal_df.head()}")
            else:
                logging.warning("No normal features extracted (DataFrame is empty)!")

            # Check if both DataFrames are empty
            if proxy_df.empty and normal_df.empty:
                logging.error("Both proxy and normal DataFrames are empty! Returning empty.")
                return pd.DataFrame()

            # Combine the DataFrames
            combined_df = pd.concat([proxy_df, normal_df], ignore_index=True)
            logging.info(f"Combined DataFrame shape: {combined_df.shape}")
            logging.info(f"Label distribution:\n{combined_df['label'].value_counts(dropna=False)}")

            # Remove Inf/NaN values
            original_shape = combined_df.shape
            combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            combined_df.dropna(inplace=True)
            logging.info(f"Cleaned dataset from shape={original_shape} to shape={combined_df.shape}")

            if not combined_df.empty:
                self._save_csv(combined_df, "nfstream", "processed", "complete_dataset.csv")
                self._save_dataset_summary(combined_df)
            else:
                logging.error("Final dataset is empty after cleaning (post-concat dropna)!")

            return combined_df

        except Exception as e:
            logging.error(f"Error in prepare_dataset: {e}", exc_info=True)
            return pd.DataFrame()

    def extract_features(self, pcap_dir: Union[str, Path], label: str) -> pd.DataFrame:
        """
        Extracts features for all PCAP files in the specified directory.

        Args:
            pcap_dir: Path to directory containing .pcap files.
            label: 'proxy' or 'normal' (used in the 'label' column of the output DataFrame).

        Returns:
            A DataFrame with flows from all PCAP files. If no files or no valid flows,
            returns an empty DataFrame.
        """
        pcap_dir = Path(pcap_dir)
        flows_data = []

        logging.info(f"\nExtracting features from directory: {pcap_dir}")
        if not pcap_dir.exists():
            logging.error(f"Directory {pcap_dir} does not exist!")
            return pd.DataFrame()

        pcap_files = list(pcap_dir.glob('*.pcap'))
        if not pcap_files:
            logging.error(f"No PCAP files found in {pcap_dir}!")
            return pd.DataFrame()

        logging.info(
            f"Found {len(pcap_files)} PCAP file(s) in {pcap_dir}. "
            f"Label for extracted flows will be '{label}'."
        )

        # Use tqdm to create a progress bar. That results in exactly one bar for this directory.
        # If you call extract_features() twice (proxy + normal), you get two bars total.
        for pcap_file in tqdm(pcap_files, desc=f"Extracting [{label}]"):
            try:
                logging.info(f"Processing file: {pcap_file}")
                streamer = nfstream.NFStreamer(
                    source=str(pcap_file),
                    decode_tunnels=True,
                    statistical_analysis=True,
                    splt_analysis=10,
                    n_dissections=20,
                    accounting_mode=3
                )

                flow_count, valid_count = 0, 0
                for flow in streamer:
                    flow_count += 1

                    # Extract features for this flow
                    feats = self._extract_flow_features(flow)
                    if feats:
                        valid_count += 1
                        flow_dict = {'label': label}
                        flow_dict.update(self._flow_features_to_dict(feats))
                        flows_data.append(flow_dict)

                logging.info(
                    f"File {pcap_file.name}: total flows = {flow_count}, valid flows = {valid_count}"
                )

            except Exception as e:
                logging.error(f"Error processing {pcap_file}: {e}", exc_info=True)

        if not flows_data:
            logging.error(
                f"No valid flows extracted from {pcap_dir} (label='{label}')! Returning empty DataFrame."
            )
            return pd.DataFrame()

        df = pd.DataFrame(flows_data)
        logging.info(
            f"Extracted features from {pcap_dir} (label='{label}'). "
            f"Resulting DataFrame shape: {df.shape}"
        )
        logging.info(f"Columns in extracted features: {df.columns.tolist()}")

        return df

    def _extract_flow_features(self, flow) -> Optional[FlowFeatures]:
        """
        Extracts FlowFeatures from a single nfstream flow object.
        Returns None if the flow doesn't meet the criteria (e.g., fewer than min_packets).
        """
        try:
            # Minimum packet requirement
            if flow.bidirectional_packets < self.min_packets:
                return None

            duration_sec = flow.bidirectional_duration_ms / 1000.0
            if duration_sec <= 0:
                return None

            features = FlowFeatures(
                duration_seconds=duration_sec,
                packets_per_second=flow.bidirectional_packets / duration_sec,
                bytes_per_second=flow.bidirectional_bytes / duration_sec,
                src2dst_packets_per_second=flow.src2dst_packets / duration_sec,
                dst2src_packets_per_second=flow.dst2src_packets / duration_sec,
                src2dst_bytes_per_second=flow.src2dst_bytes / duration_sec,
                dst2src_bytes_per_second=flow.dst2src_bytes / duration_sec,
                packet_size_avg=(
                    flow.bidirectional_bytes / flow.bidirectional_packets
                    if flow.bidirectional_packets > 0 else 0.0
                ),
                packet_ratio=(
                    flow.src2dst_packets / max(flow.dst2src_packets, 1)
                    if flow.dst2src_packets else 0.0
                ),
                byte_ratio=(
                    flow.src2dst_bytes / max(flow.dst2src_bytes, 1)
                    if flow.dst2src_bytes else 0.0
                ),
                entropy_features=None
            )

            if self.use_entropy:
                features.entropy_features = self._calculate_entropy_features(flow)

            return features

        except Exception as e:
            logging.error(f"Error extracting flow features: {e}")
            return None

    def _flow_features_to_dict(self, features: FlowFeatures) -> Dict[str, Any]:
        """
        Converts a FlowFeatures dataclass into a dict for easy DataFrame construction.
        """
        d = {
            'duration_seconds': features.duration_seconds,
            'packets_per_second': features.packets_per_second,
            'bytes_per_second': features.bytes_per_second,
            'src2dst_packets_per_second': features.src2dst_packets_per_second,
            'dst2src_packets_per_second': features.dst2src_packets_per_second,
            'src2dst_bytes_per_second': features.src2dst_bytes_per_second,
            'dst2src_bytes_per_second': features.dst2src_bytes_per_second,
            'packet_size_avg': features.packet_size_avg,
            'packet_ratio': features.packet_ratio,
            'byte_ratio': features.byte_ratio
        }
        if features.entropy_features:
            d.update(features.entropy_features)
        return d

    def _calculate_entropy_features(self, flow) -> Dict[str, float]:
        """
        Calculates optional entropy-based features (payload entropy, packet-length entropy).
        Returns an empty dict if no relevant data is found.
        """
        results = {}
        try:
            # If the flow object has payload_bytes, compute entropy
            if getattr(flow, 'payload_bytes', None):
                results['payload_entropy'] = self._calculate_entropy(flow.payload_bytes)

            # If the flow object has a packet_lengths attribute
            if hasattr(flow, 'packet_lengths') and flow.packet_lengths:
                length_bytes = bytes(flow.packet_lengths)
                results['packet_length_entropy'] = self._calculate_entropy(length_bytes)

        except Exception as e:
            logging.error(f"Error calculating entropy features: {e}")
        return results

    def _calculate_entropy(self, data: bytes) -> float:
        """
        Computes Shannon entropy for a bytes object.
        """
        if not data:
            return 0.0
        try:
            counts = {}
            for b in data:
                counts[b] = counts.get(b, 0) + 1
            total = len(data)
            probs = [v / total for v in counts.values()]
            return entropy(probs, base=2)
        except Exception as e:
            logging.error(f"Error in _calculate_entropy: {e}")
            return 0.0

    def _save_csv(self, df: pd.DataFrame, category: str, subcategory: str, filename: str):
        """
        Saves a DataFrame to CSV using the OutputManager path structure.
        """
        try:
            path = self.output_manager.get_path(category, subcategory, filename)
            df.to_csv(path, index=False)
            logging.info(f"Saved CSV: {path}")
        except Exception as e:
            logging.error(f"Error saving CSV file {filename}: {e}")

    def _save_dataset_summary(self, df: pd.DataFrame):
        """
        Saves a text summary of the combined dataset (row count, feature list, stats).
        """
        try:
            summary_path = self.output_manager.get_path("nfstream", "summaries", "dataset_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("=== Dataset Summary ===\n\n")
                f.write(f"Total flows: {len(df)}\n")
                if 'label' in df.columns:
                    # The user might still have 'proxy'/'normal' or numeric.
                    # If numeric, label=1 means proxy, label=0 means normal.
                    # This is just an example.
                    # (Optional) If you want a more direct approach,
                    #           count how many 'proxy' vs. 'normal' in the original data
                    #           or show the distribution for 0/1 if already mapped.
                    proxy_count = len(df[df['label'] == 'proxy']) if 'proxy' in df['label'].unique() else 0
                    normal_count = len(df[df['label'] == 'normal']) if 'normal' in df['label'].unique() else 0
                    f.write(f"Proxy flows (label='proxy'): {proxy_count}\n")
                    f.write(f"Normal flows (label='normal'): {normal_count}\n")

                f.write(f"Number of features: {len(df.columns) - 1}\n\n")
                f.write("Available Features:\n")
                for col in sorted(df.columns):
                    if col != 'label':
                        f.write(f" - {col}\n")
                f.write("\nFeature Statistics:\n")
                f.write(df.describe().to_string())
                f.write("\n")

            logging.info(f"Dataset summary saved to {summary_path}")
        except Exception as e:
            logging.error(f"Error saving dataset summary: {e}")
