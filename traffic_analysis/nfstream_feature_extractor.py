import nfstream
import pandas as pd
import numpy as np
import os
from scipy.stats import entropy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class FlowFeatures:
    """Datenklasse für extrahierte Flow-Features"""
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
    """Extrahiert Features aus PCAP-Files mittels NFStream"""

    def __init__(self, output_manager, use_entropy: bool = False, min_packets: int = 5):
        """
        Initialisiert den Feature Extractor

        Args:
            output_manager: Instance des OutputManagers
            use_entropy: Aktiviert Entropy-Feature-Berechnung
            min_packets: Minimale Anzahl Packets pro Flow
        """
        self.output_manager = output_manager
        self.use_entropy = use_entropy
        self.min_packets = min_packets

        # Features die von der Analyse ausgeschlossen werden
        self.exclude_features = {
            'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
            'src_port', 'dst_port', 'protocol', 'ip_version',
            'application_name', 'application_category_name',
            'client_info', 'server_info', 'master_protocol',
            'application_is_guessed', 'requested_server_name',
            'client_fingerprint', 'server_fingerprint'
        }

        print("NFStreamFeatureExtractor initialized")
        print(f"Entropy features: {'enabled' if use_entropy else 'disabled'}")
        print(f"Minimum packets per flow: {min_packets}")

    def calculate_entropy(self, data: bytes) -> float:
        """
        Berechnet Shannon-Entropy für Byte-Sequenz

        Args:
            data: Byte-Sequenz

        Returns:
            float: Entropy-Wert
        """
        try:
            if not data:
                return 0.0

            # Zähle Byte-Häufigkeiten
            byte_counts = {}
            for byte in data:
                byte_counts[byte] = byte_counts.get(byte, 0) + 1

            # Berechne Wahrscheinlichkeiten und Entropy
            total_bytes = len(data)
            probabilities = [count / total_bytes for count in byte_counts.values()]
            return entropy(probabilities, base=2)

        except Exception as e:
            print(f"Error calculating entropy: {e}")
            return 0.0

    def calculate_entropy_features(self, flow) -> Dict[str, float]:
        """
        Berechnet verschiedene Entropy-basierte Features

        Args:
            flow: NFStream Flow-Objekt

        Returns:
            Dict[str, float]: Dictionary mit Entropy-Features
        """
        entropy_features = {}

        try:
            if hasattr(flow, 'payload_bytes') and flow.payload_bytes:
                # Gesamt-Payload Entropy
                entropy_features['payload_entropy'] = self.calculate_entropy(
                    flow.payload_bytes
                )

                # Entropy der ersten n Bytes
                first_n_bytes = 64
                if len(flow.payload_bytes) >= first_n_bytes:
                    entropy_features['first_bytes_entropy'] = self.calculate_entropy(
                        flow.payload_bytes[:first_n_bytes]
                    )

                # Entropy der Paketgrößen-Verteilung
                if hasattr(flow, 'packet_lengths'):
                    entropy_features['packet_length_entropy'] = self.calculate_entropy(
                        bytes(flow.packet_lengths)
                    )

        except Exception as e:
            print(f"Error calculating entropy features: {e}")

        return entropy_features

    def _extract_flow_features(self, flow) -> Optional[FlowFeatures]:
        """
        Extrahiert Features aus einem einzelnen Flow

        Args:
            flow: NFStream Flow-Objekt

        Returns:
            Optional[FlowFeatures]: Extrahierte Features oder None bei zu wenig Packets
        """
        try:
            # Überspringe Flows mit zu wenig Packets
            if flow.bidirectional_packets < self.min_packets:
                return None

            duration_sec = flow.bidirectional_duration_ms / 1000 if flow.bidirectional_duration_ms > 0 else 0

            if duration_sec <= 0:
                return None

            # Basis-Features berechnen
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
                byte_ratio=flow.src2dst_bytes / max(flow.dst2src_bytes, 1)
            )

            # Optional: Entropy-Features
            if self.use_entropy:
                features.entropy_features = self.calculate_entropy_features(flow)

            return features

        except Exception as e:
            print(f"Error extracting flow features: {e}")
            return None

    def _convert_features_to_dict(self, features) -> dict:
        """Konvertiert FlowFeatures in ein Dictionary"""
        feature_dict = features.__dict__.copy()
        if features.entropy_features:
            feature_dict.update(features.entropy_features)
        del feature_dict['entropy_features']
        return feature_dict

    def extract_features(self, pcap_dir: Union[str, Path], label: str) -> pd.DataFrame:
        """
        Extrahiert Features aus allen PCAP-Files in einem Verzeichnis

        Args:
            pcap_dir: Verzeichnis mit PCAP-Files
            label: Label für die Flows

        Returns:
            pd.DataFrame: DataFrame mit allen Features
        """
        pcap_dir = Path(pcap_dir)
        flows_data = []

        try:
            print(f"\nAnalyzing directory: {pcap_dir}")
            print(f"Directory exists: {pcap_dir.exists()}")
            print(f"Directory contents: {list(pcap_dir.glob('*'))}")

            pcap_files = list(pcap_dir.glob('*.pcap'))
            total_files = len(pcap_files)

            print(f"Found {total_files} PCAP files")

            if total_files == 0:
                raise ValueError(f"No PCAP files found in {pcap_dir}")

            for pcap_file in pcap_files:
                print(f"\nProcessing file: {pcap_file}")
                print(f"File size: {pcap_file.stat().st_size} bytes")

                try:
                    # Konfiguriere NFStream
                    streamer = nfstream.NFStreamer(
                        source=str(pcap_file),
                        decode_tunnels=True,
                        statistical_analysis=True,
                        splt_analysis=10,
                        n_dissections=20,
                        accounting_mode=3
                    )

                    # Verarbeite jeden Flow
                    flow_count = 0
                    valid_flow_count = 0
                    for flow in streamer:
                        flow_count += 1
                        features = self._extract_flow_features(flow)
                        if features:
                            valid_flow_count += 1
                            flow_dict = {
                                'label': label,
                                **self._convert_features_to_dict(features)
                            }
                            flows_data.append(flow_dict)

                    print(f"Total flows processed: {flow_count}")
                    print(f"Valid flows extracted: {valid_flow_count}")

                except Exception as e:
                    print(f"Error processing file {pcap_file}: {str(e)}")
                    continue

            if not flows_data:
                print("Warning: No features extracted from any files")
                return pd.DataFrame()

            df = pd.DataFrame(flows_data)
            print(f"\nFinal dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            print(f"Error extracting features from {pcap_dir}: {str(e)}")
            return pd.DataFrame()

    def prepare_dataset(self, proxy_dir: Union[str, Path], normal_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Erstellt kompletten Datensatz aus Proxy- und Normal-Traffic

        Args:
            proxy_dir: Verzeichnis mit Proxy-Traffic
            normal_dir: Verzeichnis mit Normal-Traffic

        Returns:
            pd.DataFrame: Kompletter Datensatz
        """
        try:
            # Verarbeite Proxy-Traffic
            print("\nProcessing proxy traffic...")
            proxy_df = self.extract_features(proxy_dir, 'proxy')
            if not proxy_df.empty:
                print(f"Proxy dataset shape: {proxy_df.shape}")
                proxy_df.to_csv(self.output_manager.get_path(
                    "nfstream", "processed", "proxy_features.csv"
                ), index=False)
            else:
                print("No proxy features extracted!")

            # Verarbeite Normal-Traffic
            print("\nProcessing normal traffic...")
            normal_df = self.extract_features(normal_dir, 'normal')
            if not normal_df.empty:
                print(f"Normal dataset shape: {normal_df.shape}")
                normal_df.to_csv(self.output_manager.get_path(
                    "nfstream", "processed", "normal_features.csv"
                ), index=False)
            else:
                print("No normal features extracted!")

            # Kombiniere Datensätze
            if proxy_df.empty and normal_df.empty:
                print("No features extracted from either dataset!")
                return pd.DataFrame()

            df = pd.concat([proxy_df, normal_df], ignore_index=True)

            # Bereinige Datensatz
            original_shape = df.shape
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            print(f"\nDataset cleaned: {original_shape} -> {df.shape}")

            # Speichere kompletten Datensatz
            if not df.empty:
                df.to_csv(self.output_manager.get_path(
                    "nfstream", "processed", "complete_dataset.csv"
                ), index=False)
                self._save_dataset_summary(df)
                print("\nDataset saved successfully")
            else:
                print("Warning: Final dataset is empty!")

            return df

        except Exception as e:
            print(f"Error preparing dataset: {str(e)}")
            return pd.DataFrame()

    def _save_dataset_summary(self, df: pd.DataFrame):
        """
        Speichert detaillierte Zusammenfassung des Datensatzes

        Args:
            df: Kompletter Datensatz als DataFrame
        """
        try:
            summary_path = self.output_manager.get_path(
                "nfstream", "summaries", "dataset_summary.txt"
            )

            with open(summary_path, 'w') as f:
                f.write("=== Dataset Summary ===\n\n")

                # Basis-Statistiken
                f.write("Dataset Size:\n")
                f.write(f"Total flows: {len(df)}\n")
                f.write(f"Proxy flows: {len(df[df['label'] == 'proxy'])}\n")
                f.write(f"Normal flows: {len(df[df['label'] == 'normal'])}\n")
                f.write(f"Number of features: {len(df.columns) - 1}\n\n")

                # Feature-Liste und Statistiken
                f.write("Available Features:\n")
                for column in sorted(df.columns):
                    if column != 'label':
                        f.write(f"- {column}\n")

                # Detaillierte Statistiken
                f.write("\nFeature Statistics:\n")
                f.write(df.describe().to_string())

            print(f"Dataset summary saved to {summary_path}")

        except Exception as e:
            print(f"Error saving dataset summary: {str(e)}")


if __name__ == "__main__":
    # Test-Code
    from pathlib import Path

    test_dir = Path("test_pcaps")
    if test_dir.exists():
        extractor = NFStreamFeatureExtractor(None)
        df = extractor.extract_features(test_dir, "test")
        print(f"Extracted features shape: {df.shape}")