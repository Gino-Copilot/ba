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
            flow_features.update({
                'duration_seconds': duration_sec,
                'packets_per_second': flow.bidirectional_packets / duration_sec,
                'bytes_per_second': flow.bidirectional_bytes / duration_sec,
                'src2dst_packets_per_second': flow.src2dst_packets / duration_sec,
                'dst2src_packets_per_second': flow.dst2src_packets / duration_sec,
                'src2dst_bytes_per_second': flow.src2dst_bytes / duration_sec,
                'dst2src_bytes_per_second': flow.dst2src_bytes / duration_sec
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

        return df
