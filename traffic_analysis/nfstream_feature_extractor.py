import nfstream
import pandas as pd
import numpy as np
import os
from scipy.stats import entropy
import math


class NFStreamFeatureExtractor:
    def __init__(self, use_entropy=False):
        self.use_entropy = use_entropy
        self.exclude_features = {
            'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
            'src_port', 'dst_port', 'protocol', 'ip_version',
            'application_name', 'application_category_name',
            'client_info', 'server_info', 'master_protocol',
            'application_is_guessed', 'requested_server_name',
            'client_fingerprint', 'server_fingerprint'
        }

    def calculate_byte_entropy(self, data):
        """Berechnet die Shannon-Entropie der Bytes"""
        if not data:
            return 0

        # Zähle die Häufigkeit jedes Bytes
        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1

        # Berechne die Entropie
        total_bytes = len(data)
        probabilities = [count / total_bytes for count in byte_counts.values()]
        return entropy(probabilities, base=2)

    def calculate_entropy_features(self, flow):
        """Berechnet verschiedene Entropie-basierte Features"""
        entropy_features = {}

        if hasattr(flow, 'payload_bytes') and flow.payload_bytes:
            # Gesamte Payload-Entropie
            entropy_features['payload_entropy'] = self.calculate_byte_entropy(flow.payload_bytes)

            # Getrennte Entropie für die ersten n Bytes
            first_n_bytes = 64  # Anzahl der zu analysierenden Bytes
            if len(flow.payload_bytes) >= first_n_bytes:
                entropy_features['first_bytes_entropy'] = self.calculate_byte_entropy(
                    flow.payload_bytes[:first_n_bytes]
                )

            # Entropie der Paketgrößenverteilung
            if hasattr(flow, 'packet_lengths'):
                entropy_features['packet_length_entropy'] = self.calculate_byte_entropy(
                    bytes(flow.packet_lengths)
                )

        return entropy_features

    def _extract_flow_features(self, flow):
        flow_features = {}
        if flow.bidirectional_packets < 5:
            return None

        duration_sec = flow.bidirectional_duration_ms / 1000 if flow.bidirectional_duration_ms > 0 else 0
        if duration_sec > 0:
            # Basis-Features
            flow_features.update({
                'duration_seconds': duration_sec,
                'packets_per_second': flow.bidirectional_packets / duration_sec,
                'bytes_per_second': flow.bidirectional_bytes / duration_sec,
                'src2dst_packets_per_second': flow.src2dst_packets / duration_sec,
                'dst2src_packets_per_second': flow.dst2src_packets / duration_sec,
                'src2dst_bytes_per_second': flow.src2dst_bytes / duration_sec,
                'dst2src_bytes_per_second': flow.dst2src_bytes / duration_sec,
                'packet_size_avg': flow.bidirectional_bytes / flow.bidirectional_packets,
                'packet_ratio': flow.src2dst_packets / max(flow.dst2src_packets, 1),
                'byte_ratio': flow.src2dst_bytes / max(flow.dst2src_bytes, 1)
            })

            # Entropie-Features hinzufügen wenn aktiviert
            if self.use_entropy:
                entropy_features = self.calculate_entropy_features(flow)
                flow_features.update(entropy_features)

        return flow_features

    def extract_features(self, pcap_dir, label):
        flows_data = []
        total_files = len([f for f in os.listdir(pcap_dir) if f.endswith('.pcap')])
        processed_files = 0

        for filename in os.listdir(pcap_dir):
            if filename.endswith('.pcap'):
                processed_files += 1
                pcap_path = os.path.join(pcap_dir, filename)
                print(f"Verarbeite Datei {processed_files}/{total_files}: {filename}")

                try:
                    streamer = nfstream.NFStreamer(
                        source=pcap_path,
                        decode_tunnels=True,
                        statistical_analysis=True,
                        splt_analysis=10,
                        n_dissections=20,
                        accounting_mode=3
                    )

                    for flow in streamer:
                        flow_features = self._extract_flow_features(flow)
                        if flow_features:
                            flow_features['label'] = label
                            flows_data.append(flow_features)

                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von {filename}: {str(e)}")
                    continue

        if not flows_data:
            print("Warnung: Keine Daten extrahiert. Bitte überprüfen Sie die PCAP-Dateien im Verzeichnis:", pcap_dir)

        return pd.DataFrame(flows_data)

    def prepare_dataset(self, proxy_dir, normal_dir):
        """Erstellt den vollständigen Datensatz aus beiden Verzeichnissen"""
        print("\nVerarbeite Proxy-Traffic...")
        proxy_df = self.extract_features(proxy_dir, 'proxy')

        print("\nVerarbeite normalen Traffic...")
        normal_df = self.extract_features(normal_dir, 'normal')

        # Kombiniere und bereinige Datensätze
        df = pd.concat([proxy_df, normal_df], ignore_index=True)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        if 'label' not in df.columns:
            print("Fehler: Die Spalte 'label' fehlt im kombinierten DataFrame!")
        else:
            print("Spalte 'label' erfolgreich hinzugefügt.")

        print("\nDatensatz-Statistiken:")
        print(f"Gesamtanzahl Flows: {len(df)}")
        print(f"Proxy Flows: {len(df[df['label'] == 'proxy'])}")
        print(f"Normal Flows: {len(df[df['label'] == 'normal'])}")
        print(f"Anzahl Features: {len(df.columns) - 1}")

        print("\nVerfügbare Features:")
        for column in sorted(df.columns):
            if column != 'label':
                print(f"- {column}")

        return df