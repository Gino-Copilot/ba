import nfstream
import pandas as pd
import numpy as np
import os
from scipy.stats import entropy
import math


class NFStreamFeatureExtractor:
    def __init__(self, output_manager, use_entropy=False):
        """
        Initialize NFStream Feature Extractor

        Args:
            output_manager: Instance of OutputManager for handling output paths
            use_entropy: Boolean to enable/disable entropy features
        """
        self.output_manager = output_manager
        self.use_entropy = use_entropy

        # Features to exclude from analysis
        self.exclude_features = {
            'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
            'src_port', 'dst_port', 'protocol', 'ip_version',
            'application_name', 'application_category_name',
            'client_info', 'server_info', 'master_protocol',
            'application_is_guessed', 'requested_server_name',
            'client_fingerprint', 'server_fingerprint'
        }

    def calculate_byte_entropy(self, data):
        """
        Calculate Shannon entropy of bytes

        Args:
            data: Byte sequence

        Returns:
            float: Entropy value
        """
        if not data:
            return 0

        # Count frequency of each byte
        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1

        # Calculate entropy
        total_bytes = len(data)
        probabilities = [count / total_bytes for count in byte_counts.values()]
        return entropy(probabilities, base=2)

    def calculate_entropy_features(self, flow):
        """
        Calculate various entropy-based features for a flow

        Args:
            flow: NFStream flow object

        Returns:
            dict: Dictionary of entropy features
        """
        entropy_features = {}

        if hasattr(flow, 'payload_bytes') and flow.payload_bytes:
            # Total payload entropy
            entropy_features['payload_entropy'] = self.calculate_byte_entropy(flow.payload_bytes)

            # Entropy of first n bytes
            first_n_bytes = 64
            if len(flow.payload_bytes) >= first_n_bytes:
                entropy_features['first_bytes_entropy'] = self.calculate_byte_entropy(
                    flow.payload_bytes[:first_n_bytes]
                )

            # Entropy of packet size distribution
            if hasattr(flow, 'packet_lengths'):
                entropy_features['packet_length_entropy'] = self.calculate_byte_entropy(
                    bytes(flow.packet_lengths)
                )

        return entropy_features

    def _extract_flow_features(self, flow):
        """
        Extract features from a single flow

        Args:
            flow: NFStream flow object

        Returns:
            dict: Dictionary of flow features or None if flow is too short
        """
        # Skip flows with too few packets
        if flow.bidirectional_packets < 5:
            return None

        flow_features = {}
        duration_sec = flow.bidirectional_duration_ms / 1000 if flow.bidirectional_duration_ms > 0 else 0

        if duration_sec > 0:
            # Basic time-based features
            flow_features.update({
                'duration_seconds': duration_sec,
                'packets_per_second': flow.bidirectional_packets / duration_sec,
                'bytes_per_second': flow.bidirectional_bytes / duration_sec,
            })

            # Directional packet features
            flow_features.update({
                'src2dst_packets_per_second': flow.src2dst_packets / duration_sec,
                'dst2src_packets_per_second': flow.dst2src_packets / duration_sec,
            })

            # Directional byte features
            flow_features.update({
                'src2dst_bytes_per_second': flow.src2dst_bytes / duration_sec,
                'dst2src_bytes_per_second': flow.dst2src_bytes / duration_sec,
            })

            # Statistical features
            flow_features.update({
                'packet_size_avg': flow.bidirectional_bytes / flow.bidirectional_packets,
                'packet_ratio': flow.src2dst_packets / max(flow.dst2src_packets, 1),
                'byte_ratio': flow.src2dst_bytes / max(flow.dst2src_bytes, 1)
            })

            # Add entropy features if enabled
            if self.use_entropy:
                entropy_features = self.calculate_entropy_features(flow)
                flow_features.update(entropy_features)

        return flow_features

    def extract_features(self, pcap_dir, label):
        """
        Extract features from all PCAP files in directory

        Args:
            pcap_dir: Directory containing PCAP files
            label: Label for the traffic class

        Returns:
            DataFrame: Features for all flows
        """
        flows_data = []
        total_files = len([f for f in os.listdir(pcap_dir) if f.endswith('.pcap')])
        processed_files = 0

        # Create directory for intermediate results
        intermediate_dir = self.output_manager.get_path(
            "nfstream", "intermediate", ""
        )
        os.makedirs(intermediate_dir, exist_ok=True)

        for filename in os.listdir(pcap_dir):
            if filename.endswith('.pcap'):
                processed_files += 1
                pcap_path = os.path.join(pcap_dir, filename)
                print(f"Processing file {processed_files}/{total_files}: {filename}")

                try:
                    # Configure NFStream
                    streamer = nfstream.NFStreamer(
                        source=pcap_path,
                        decode_tunnels=True,
                        statistical_analysis=True,
                        splt_analysis=10,
                        n_dissections=20,
                        accounting_mode=3
                    )

                    # Process each flow
                    file_flows = []
                    for flow in streamer:
                        flow_features = self._extract_flow_features(flow)
                        if flow_features:
                            flow_features['label'] = label
                            file_flows.append(flow_features)

                    # Save intermediate results for each file
                    if file_flows:
                        intermediate_path = os.path.join(
                            intermediate_dir,
                            f"features_{filename.replace('.pcap', '.csv')}"
                        )
                        pd.DataFrame(file_flows).to_csv(intermediate_path, index=False)
                        flows_data.extend(file_flows)

                except Exception as e:
                    error_path = self.output_manager.get_path(
                        "nfstream", "errors", f"error_log_{filename}.txt"
                    )
                    with open(error_path, 'w') as f:
                        f.write(f"Error processing {filename}: {str(e)}")
                    print(f"Error processing {filename}: {str(e)}")
                    continue

        if not flows_data:
            print("Warning: No data extracted. Please check the PCAP files in directory:", pcap_dir)
            return pd.DataFrame()

        return pd.DataFrame(flows_data)

    def prepare_dataset(self, proxy_dir, normal_dir):
        """
        Create complete dataset from proxy and normal traffic

        Args:
            proxy_dir: Directory with proxy traffic PCAPs
            normal_dir: Directory with normal traffic PCAPs

        Returns:
            DataFrame: Complete dataset with all features
        """
        print("\nProcessing proxy traffic...")
        proxy_df = self.extract_features(proxy_dir, 'proxy')

        # Save proxy features
        proxy_path = self.output_manager.get_path(
            "nfstream", "processed", "proxy_features.csv"
        )
        proxy_df.to_csv(proxy_path, index=False)

        print("\nProcessing normal traffic...")
        normal_df = self.extract_features(normal_dir, 'normal')

        # Save normal features
        normal_path = self.output_manager.get_path(
            "nfstream", "processed", "normal_features.csv"
        )
        normal_df.to_csv(normal_path, index=False)

        # Combine datasets
        df = pd.concat([proxy_df, normal_df], ignore_index=True)

        # Clean dataset
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        # Save complete dataset
        complete_path = self.output_manager.get_path(
            "nfstream", "processed", "complete_dataset.csv"
        )
        df.to_csv(complete_path, index=False)

        # Create and save dataset summary
        self._save_dataset_summary(df)

        print("\nDataset Statistics:")
        print(f"Total number of flows: {len(df)}")
        print(f"Proxy flows: {len(df[df['label'] == 'proxy'])}")
        print(f"Normal flows: {len(df[df['label'] == 'normal'])}")
        print(f"Number of features: {len(df.columns) - 1}")

        print("\nAvailable features:")
        for column in sorted(df.columns):
            if column != 'label':
                print(f"- {column}")

        return df

    def _save_dataset_summary(self, df):
        """
        Save detailed summary of the dataset

        Args:
            df: Complete dataset DataFrame
        """
        summary_path = self.output_manager.get_path(
            "nfstream", "summaries", "dataset_summary.txt"
        )

        with open(summary_path, 'w') as f:
            f.write("=== Dataset Summary ===\n\n")

            # Basic statistics
            f.write("Dataset Size:\n")
            f.write(f"Total flows: {len(df)}\n")
            f.write(f"Proxy flows: {len(df[df['label'] == 'proxy'])}\n")
            f.write(f"Normal flows: {len(df[df['label'] == 'normal'])}\n")
            f.write(f"Number of features: {len(df.columns) - 1}\n\n")

            # Feature statistics
            f.write("Feature Statistics:\n")
            stats = df.describe()
            stats.to_string(f)

            # Feature list
            f.write("\n\nAvailable Features:\n")
            for column in sorted(df.columns):
                if column != 'label':
                    f.write(f"- {column}\n")