import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import nfstream
import numpy as np
import pandas as pd
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO)


@dataclass
class FlowFeatures:
    """
    Data class that holds extracted flow features.
    """
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
    Extracts flow-based features from PCAP files using NFStream.
    """

    def __init__(
        self,
        output_manager,
        use_entropy: bool = False,
        min_packets: int = 5
    ):
        """
        Initializes the NFStreamFeatureExtractor.

        Args:
            output_manager: Instance of an output manager (handles file paths).
            use_entropy: Whether to calculate entropy-based features.
            min_packets: Minimum number of bidirectional packets to accept a flow.
        """
        self.output_manager = output_manager
        self.use_entropy = use_entropy
        self.min_packets = min_packets

        # We do not necessarily need to store these excluded features here,
        # unless you plan to filter them out from the final DataFrame.
        # Otherwise we can remove or move them to a config if not used.
        self.exclude_features = {
            'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
            'src_port', 'dst_port', 'protocol', 'ip_version',
            'application_name', 'application_category_name',
            'client_info', 'server_info', 'master_protocol',
            'application_is_guessed', 'requested_server_name',
            'client_fingerprint', 'server_fingerprint'
        }

        logging.info("NFStreamFeatureExtractor initialized.")
        logging.info(f"Entropy calculation: {'enabled' if use_entropy else 'disabled'}.")
        logging.info(f"Minimum packets per flow: {min_packets}.")

    def prepare_dataset(
        self,
        proxy_dir: Union[str, Path],
        normal_dir: Union[str, Path]
    ) -> pd.DataFrame:
        """
        Creates a combined dataset from proxy (Shadowsocks, etc.) and normal traffic directories.

        Args:
            proxy_dir: Directory containing PCAP files for proxy traffic.
            normal_dir: Directory containing PCAP files for normal traffic.

        Returns:
            A pandas DataFrame containing the combined dataset with labels ('proxy'/'normal').
        """
        try:
            logging.info("Starting dataset preparation...")

            # Extract features for proxy traffic
            proxy_df = self.extract_features(proxy_dir, label='proxy')
            if not proxy_df.empty:
                self._save_csv(proxy_df, "nfstream", "processed", "proxy_features.csv")
            else:
                logging.warning("No proxy features extracted.")

            # Extract features for normal traffic
            normal_df = self.extract_features(normal_dir, label='normal')
            if not normal_df.empty:
                self._save_csv(normal_df, "nfstream", "processed", "normal_features.csv")
            else:
                logging.warning("No normal features extracted.")

            # Combine both
            if proxy_df.empty and normal_df.empty:
                logging.warning("No features extracted from either directory.")
                return pd.DataFrame()

            combined_df = pd.concat([proxy_df, normal_df], ignore_index=True)

            # Cleanup any inf/-inf and NaN values
            original_shape = combined_df.shape
            combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            combined_df.dropna(inplace=True)

            logging.info(f"Dataset cleaned from {original_shape} to {combined_df.shape}.")

            if not combined_df.empty:
                self._save_csv(combined_df, "nfstream", "processed", "complete_dataset.csv")
                self._save_dataset_summary(combined_df)
            else:
                logging.warning("Final combined dataset is empty after cleaning.")

            return combined_df

        except Exception as e:
            logging.error(f"Error preparing dataset: {e}")
            return pd.DataFrame()

    def extract_features(
        self,
        pcap_dir: Union[str, Path],
        label: str
    ) -> pd.DataFrame:
        """
        Extracts features from all PCAP files in the specified directory.

        Args:
            pcap_dir: The directory containing PCAP files.
            label: The label to assign to the extracted flows (e.g., 'proxy' or 'normal').

        Returns:
            A pandas DataFrame with extracted flow features.
        """
        pcap_dir = Path(pcap_dir)
        flows_data = []

        logging.info(f"Analyzing directory: {pcap_dir}")
        if not pcap_dir.exists():
            logging.warning(f"Directory {pcap_dir} does not exist.")
            return pd.DataFrame()

        pcap_files = list(pcap_dir.glob('*.pcap'))
        if not pcap_files:
            logging.warning(f"No PCAP files found in {pcap_dir}.")
            return pd.DataFrame()

        logging.info(f"Found {len(pcap_files)} PCAP file(s) in {pcap_dir}.")

        # Process each PCAP file
        for pcap_file in pcap_files:
            logging.info(f"Processing file: {pcap_file}")
            try:
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
                    flow_features = self._extract_flow_features(flow)

                    if flow_features:
                        valid_count += 1
                        flow_dict = {'label': label}
                        flow_dict.update(self._flow_features_to_dict(flow_features))
                        flows_data.append(flow_dict)

                logging.info(f"Flows processed: {flow_count}, valid flows: {valid_count}")

            except Exception as e:
                logging.error(f"Error processing {pcap_file}: {e}")

        if not flows_data:
            logging.warning("No valid flow features found.")
            return pd.DataFrame()

        df = pd.DataFrame(flows_data)
        logging.info(f"Resulting DataFrame shape: {df.shape}")

        return df

    def _extract_flow_features(self, flow) -> Optional[FlowFeatures]:
        """
        Extracts features from a single NFStream flow object.
        Returns a FlowFeatures instance or None if the flow is invalid.

        Args:
            flow: An NFStream flow object.

        Returns:
            FlowFeatures dataclass or None if the flow is invalid.
        """
        try:
            if flow.bidirectional_packets < self.min_packets:
                return None

            duration_sec = flow.bidirectional_duration_ms / 1000.0
            if duration_sec <= 0:
                return None

            # Basic flow features
            features = FlowFeatures(
                duration_seconds=duration_sec,
                packets_per_second=flow.bidirectional_packets / duration_sec,
                bytes_per_second=flow.bidirectional_bytes / duration_sec,
                src2dst_packets_per_second=flow.src2dst_packets / duration_sec,
                dst2src_packets_per_second=flow.dst2src_packets / duration_sec,
                src2dst_bytes_per_second=flow.src2dst_bytes / duration_sec,
                dst2src_bytes_per_second=flow.dst2src_bytes / duration_sec,
                packet_size_avg=flow.bidirectional_bytes / flow.bidirectional_packets,
                packet_ratio=flow.src2dst_packets / max(flow.dst2src_packets, 1),
                byte_ratio=flow.src2dst_bytes / max(flow.dst2src_bytes, 1),
                entropy_features=None
            )

            # Add entropy features if required
            if self.use_entropy:
                entropy_dict = self._calculate_entropy_features(flow)
                features.entropy_features = entropy_dict

            return features

        except Exception as e:
            logging.error(f"Error extracting flow features: {e}")
            return None

    def _flow_features_to_dict(self, features: FlowFeatures) -> dict:
        """
        Converts a FlowFeatures instance into a dictionary for DataFrame insertion.

        Args:
            features: A FlowFeatures object.

        Returns:
            Dictionary of feature_name -> value.
        """
        result = {
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

        # Merge in entropy features if present
        if features.entropy_features:
            result.update(features.entropy_features)

        return result

    def _calculate_entropy_features(self, flow) -> Dict[str, float]:
        """
        Calculates various entropy-based features from the flow payload.

        Args:
            flow: An NFStream flow object containing payload bytes.

        Returns:
            A dictionary of entropy-related features.
        """
        results = {}
        try:
            if hasattr(flow, 'payload_bytes') and flow.payload_bytes:
                # Full payload entropy
                results['payload_entropy'] = self._calculate_entropy(flow.payload_bytes)

                # Optionally, first N bytes
                first_n = 64
                if len(flow.payload_bytes) >= first_n:
                    snippet = flow.payload_bytes[:first_n]
                    results['first_bytes_entropy'] = self._calculate_entropy(snippet)

            # If flow.packet_lengths is available, we can also compute
            # an entropy measure for packet length distribution
            if hasattr(flow, 'packet_lengths') and flow.packet_lengths:
                length_bytes = bytes(flow.packet_lengths)
                results['packet_length_entropy'] = self._calculate_entropy(length_bytes)

        except Exception as e:
            logging.error(f"Error calculating entropy features: {e}")

        return results

    def _calculate_entropy(self, data: bytes) -> float:
        """
        Calculates Shannon entropy for a given byte sequence.

        Args:
            data: The byte sequence for which to compute entropy.

        Returns:
            A float representing the Shannon entropy in bits.
        """
        if not data:
            return 0.0

        try:
            counts = {}
            for b in data:
                counts[b] = counts.get(b, 0) + 1
            total = len(data)
            probabilities = [count / total for count in counts.values()]
            return entropy(probabilities, base=2)
        except Exception as e:
            logging.error(f"Error in _calculate_entropy: {e}")
            return 0.0

    def _save_csv(self, df: pd.DataFrame, category: str, subcategory: str, filename: str):
        """
        Saves a DataFrame to CSV in a given subdirectory managed by output_manager.

        Args:
            df: DataFrame to be saved.
            category: Top-level category (e.g., 'nfstream').
            subcategory: Subcategory inside the top-level directory.
            filename: Name of the output CSV file.
        """
        try:
            csv_path = self.output_manager.get_path(category, subcategory, filename)
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved CSV: {csv_path}")
        except Exception as e:
            logging.error(f"Error saving CSV file {filename}: {e}")

    def _save_dataset_summary(self, df: pd.DataFrame):
        """
        Saves a textual summary of the final dataset.

        Args:
            df: The combined dataset as a pandas DataFrame.
        """
        try:
            summary_path = self.output_manager.get_path("nfstream", "summaries", "dataset_summary.txt")
            with open(summary_path, 'w') as file:
                file.write("=== Dataset Summary ===\n\n")
                file.write(f"Total flows: {len(df)}\n")
                file.write(f"Proxy flows: {len(df[df['label'] == 'proxy'])}\n")
                file.write(f"Normal flows: {len(df[df['label'] == 'normal'])}\n")
                file.write(f"Number of features (excluding label): {len(df.columns) - 1}\n\n")

                file.write("Available Features:\n")
                for col in sorted(df.columns):
                    if col != 'label':
                        file.write(f" - {col}\n")

                file.write("\nFeature Statistics:\n")
                file.write(df.describe().to_string())
                file.write("\n")
            logging.info(f"Dataset summary saved to {summary_path}")

        except Exception as e:
            logging.error(f"Error saving dataset summary: {e}")
