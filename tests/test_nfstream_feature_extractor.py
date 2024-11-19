import unittest
import os
import pandas as pd
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor


class TestNFStreamFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Initialize the extractor and set up test paths
        self.extractor = NFStreamFeatureExtractor()
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.pcap_dir = os.path.join(self.test_dir, "test_data")

    def test_feature_extraction(self):
        # Verify PCAP files exist in test directory
        pcap_files = [f for f in os.listdir(self.pcap_dir) if f.endswith('.pcap')]
        self.assertGreater(len(pcap_files), 0, "No PCAP files found in test directory")

        # Extract features from PCAP files
        df = self.extractor.extract_features(self.pcap_dir, 'test')
        self.assertIsInstance(df, pd.DataFrame)

        # Verify features and flows were extracted
        self.assertGreater(len(df.columns), 0, "No features were extracted")
        self.assertGreater(len(df), 0, "No flows were extracted from PCAP files")

        # Print test results
        print("\nTest results:")
        print(f"- Found {len(pcap_files)} PCAP files")
        print(f"- Created DataFrame with {len(df.columns)} features")
        print(f"- Number of flows: {len(df)}")

        # Display all extracted features
        print("\nExtracted features:")
        for column in df.columns:
            print(f"- {column}")


if __name__ == "__main__":
    unittest.main()