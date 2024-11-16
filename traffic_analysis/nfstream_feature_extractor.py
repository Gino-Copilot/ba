import nfstream
import pandas as pd
import numpy as np
import os


class NFStreamFeatureExtractor:
    def __init__(self):
        self.exclude_features = {
            'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
            'src_port', 'dst_port', 'protocol', 'ip_version',
            'application_name', 'application_category_name',
            'client_info', 'server_info', 'master_protocol',
            'application_is_guessed', 'requested_server_name',
            'client_fingerprint', 'server_fingerprint'
        }

    def _extract_flow_features(self, flow):
        flow_features = {}
        if flow.bidirectional_packets < 5:
            return None

        duration_sec = flow.bidirectional_duration_ms / 1000 if flow.bidirectional_duration_ms > 0 else 0
        if duration_sec > 0:
            # Basis-Zeitfeatures
            flow_features.update({
                'duration_seconds': duration_sec,

                # Paket-basierte Features
                'packets_per_second': flow.bidirectional_packets / duration_sec,
                'src2dst_packets_per_second': flow.src2dst_packets / duration_sec,
                'dst2src_packets_per_second': flow.dst2src_packets / duration_sec,
                'packet_ratio': flow.src2dst_packets / (flow.dst2src_packets + 1),
                'packet_size_avg': flow.bidirectional_bytes / (flow.bidirectional_packets + 1),

                # Byte-basierte Features
                'bytes_per_second': flow.bidirectional_bytes / duration_sec,
                'src2dst_bytes_per_second': flow.src2dst_bytes / duration_sec,
                'dst2src_bytes_per_second': flow.dst2src_bytes / duration_sec,
                'byte_ratio': flow.src2dst_bytes / (flow.dst2src_bytes + 1),

                # Statistische Features
                'iat_avg': flow.bidirectional_duration_ms / (flow.bidirectional_packets + 1),
                'src2dst_iat_avg': flow.src2dst_duration_ms / (flow.src2dst_packets + 1),
                'dst2src_iat_avg': flow.dst2src_duration_ms / (flow.dst2src_packets + 1),
            })

            # Erweiterte Features (mit Verfügbarkeitsprüfung)
            if hasattr(flow, 'bidirectional_min_piat_ms') and hasattr(flow, 'bidirectional_max_piat_ms'):
                flow_features.update({
                    'min_iat_ms': flow.bidirectional_min_piat_ms,
                    'max_iat_ms': flow.bidirectional_max_piat_ms,
                })

            # TCP-spezifische Features
            if hasattr(flow, 'src2dst_tcp_flags') and hasattr(flow, 'dst2src_tcp_flags'):
                flow_features.update({
                    'tcp_flags_ratio': flow.src2dst_tcp_flags / (flow.dst2src_tcp_flags + 1),
                })

            # Burst Features (falls verfügbar)
            if all(hasattr(flow, attr) for attr in ['src2dst_burst_packets', 'dst2src_burst_packets',
                                                    'src2dst_burst_bytes', 'dst2src_burst_bytes']):
                flow_features.update({
                    'burst_packets_ratio': flow.src2dst_burst_packets / (flow.dst2src_burst_packets + 1),
                    'burst_bytes_ratio': flow.src2dst_burst_bytes / (flow.dst2src_burst_bytes + 1),
                })

            # TLS-spezifische Features (optional)
            if hasattr(flow, 'tls_version'):
                flow_features.update({
                    'tls_sni_length': len(flow.requested_server_name) if flow.requested_server_name else 0,
                })
                if hasattr(flow, 'tls_client_hello_length'):
                    flow_features.update({
                        'tls_client_hello_length': flow.tls_client_hello_length
                    })
                if hasattr(flow, 'tls_server_hello_length'):
                    flow_features.update({
                        'tls_server_hello_length': flow.tls_server_hello_length
                    })

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
                            # Stelle sicher, dass das Label korrekt hinzugefügt wird
                            flow_features['label'] = label
                            flows_data.append(flow_features)

                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von {filename}: {str(e)}")
                    continue

        # Überprüfe, ob Daten extrahiert wurden
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

        # Debug-Ausgabe, um sicherzustellen, dass die Spalte 'label' vorhanden ist
        if 'label' not in df.columns:
            print("Fehler: Die Spalte 'label' fehlt im kombinierten DataFrame!")
        else:
            print("Spalte 'label' erfolgreich hinzugefügt.")

        print("\nDatensatz-Statistiken:")
        print(f"Gesamtanzahl Flows: {len(df)}")
        print(f"Proxy Flows: {len(df[df['label'] == 'proxy'])}")
        print(f"Normal Flows: {len(df[df['label'] == 'normal'])}")
        print(f"Anzahl Features: {len(df.columns) - 1}")

        # Zusätzliche Feature-Statistiken
        print("\nVerfügbare Features:")
        for column in sorted(df.columns):
            if column != 'label':
                print(f"- {column}")

        return df