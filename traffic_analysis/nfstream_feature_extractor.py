import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Any

import nfstream
import numpy as np
import pandas as pd
from scipy.stats import entropy


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
    Extracts flow-based features from PCAP files using NFStream.
    """

    def __init__(self, output_manager, use_entropy: bool = False, min_packets: int = 5):
        """
        Args:
            output_manager: Manages output paths.
            use_entropy: Whether to calculate entropy-based features.
            min_packets: Minimum number of packets to keep a flow.
        """
        self.output_manager = output_manager
        self.use_entropy = use_entropy
        self.min_packets = min_packets

        logging.info("NFStreamFeatureExtractor initialized.")
        logging.info(f"Entropy calculation: {'enabled' if use_entropy else 'disabled'}.")
        logging.info(f"Minimum packets per flow: {min_packets}.")

    def prepare_dataset(self, proxy_dir: Union[str, Path], normal_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Creates a combined dataset from proxy and normal traffic directories.
        """
        try:
            logging.info("Starting dataset preparation...")

            proxy_df = self.extract_features(proxy_dir, label='proxy')
            if not proxy_df.empty:
                self._save_csv(proxy_df, "nfstream", "processed", "proxy_features.csv")
            else:
                logging.warning("No proxy features extracted.")

            normal_df = self.extract_features(normal_dir, label='normal')
            if not normal_df.empty:
                self._save_csv(normal_df, "nfstream", "processed", "normal_features.csv")
            else:
                logging.warning("No normal features extracted.")

            if proxy_df.empty and normal_df.empty:
                logging.warning("No features extracted from either directory.")
                return pd.DataFrame()

            combined_df = pd.concat([proxy_df, normal_df], ignore_index=True)

            # Replace inf & drop NaN
            original_shape = combined_df.shape
            combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            combined_df.dropna(inplace=True)
            logging.info(f"Dataset cleaned from {original_shape} to {combined_df.shape}.")

            if not combined_df.empty:
                self._save_csv(combined_df, "nfstream", "processed", "complete_dataset.csv")
                self._save_dataset_summary(combined_df)
            else:
                logging.warning("Final dataset is empty after cleaning.")

            return combined_df

        except Exception as e:
            logging.error(f"Error preparing dataset: {e}")
            return pd.DataFrame()

    def extract_features(self, pcap_dir: Union[str, Path], label: str) -> pd.DataFrame:
        """
        Extracts features from all PCAP files in the specified directory.
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
                    feats = self._extract_flow_features(flow)
                    if feats:
                        valid_count += 1
                        dct = {'label': label}
                        dct.update(self._flow_features_to_dict(feats))
                        flows_data.append(dct)

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
        """
        try:
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
                packet_size_avg=flow.bidirectional_bytes / flow.bidirectional_packets,
                packet_ratio=flow.src2dst_packets / max(flow.dst2src_packets, 1),
                byte_ratio=flow.src2dst_bytes / max(flow.dst2src_bytes, 1),
                entropy_features=None
            )

            if self.use_entropy:
                features.entropy_features = self._calculate_entropy_features(flow)

            return features
        except Exception as e:
            logging.error(f"Error extracting flow features: {e}")
            return None

    def _flow_features_to_dict(self, features: FlowFeatures) -> Dict[str, Any]:
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
        results = {}
        try:
            if getattr(flow, 'payload_bytes', None):
                results['payload_entropy'] = self._calculate_entropy(flow.payload_bytes)

            if hasattr(flow, 'packet_lengths') and flow.packet_lengths:
                length_bytes = bytes(flow.packet_lengths)
                results['packet_length_entropy'] = self._calculate_entropy(length_bytes)

        except Exception as e:
            logging.error(f"Error calculating entropy features: {e}")
        return results

    def _calculate_entropy(self, data: bytes) -> float:
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
        try:
            path = self.output_manager.get_path(category, subcategory, filename)
            df.to_csv(path, index=False)
            logging.info(f"Saved CSV: {path}")
        except Exception as e:
            logging.error(f"Error saving CSV file {filename}: {e}")

    def _save_dataset_summary(self, df: pd.DataFrame):
        try:
            summary_path = self.output_manager.get_path("nfstream", "summaries", "dataset_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("=== Dataset Summary ===\n\n")
                f.write(f"Total flows: {len(df)}\n")
                if 'label' in df.columns:
                    f.write(f"Proxy flows (label=1): {len(df[df['label'] == 'proxy'])}\n")
                    f.write(f"Normal flows (label=0): {len(df[df['label'] == 'normal'])}\n")
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
