import unittest
import os
import pandas as pd
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor


class TestNFStreamFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Setup für den Test: Beispiel-Ordner mit PCAP-Dateien, falls erforderlich
        self.extractor = NFStreamFeatureExtractor()
        self.mock_pcap_dir = "tests/mock_pcap_files"

        # Erstelle das Verzeichnis und füge Beispiel-PCAP-Dateien hinzu (falls noch nicht vorhanden)
        if not os.path.exists(self.mock_pcap_dir):
            os.makedirs(self.mock_pcap_dir)
            # Hier könntest du eine kleine Beispiel-PCAP-Datei hinzufügen, falls vorhanden

    def test_feature_extraction(self):
        # Überprüfe, ob die Feature-Extraktion funktioniert und einen DataFrame zurückgibt
        df = self.extractor.extract_features(self.mock_pcap_dir, 'test')
        self.assertIsInstance(df, pd.DataFrame)
        print("Feature extraction funktioniert korrekt und liefert einen DataFrame zurück.")

    def tearDown(self):
        # Entferne die Beispiel-PCAP-Dateien nach dem Test
        if os.path.exists(self.mock_pcap_dir):
            for filename in os.listdir(self.mock_pcap_dir):
                file_path = os.path.join(self.mock_pcap_dir, filename)
                os.remove(file_path)
            os.rmdir(self.mock_pcap_dir)


if __name__ == "__main__":
    unittest.main()
