# file: data_inspector.py

import logging
import shutil
from pathlib import Path


class DataInspector:
    """
    Checks .pcap files for a minimum size requirement, copies only those above
    the threshold into a 'clean' folder, and provides a method to inspect
    DataFrame properties.
    """

    def __init__(self, min_file_size_bytes=50000, min_flow_count=10):
        """
        Args:
            min_file_size_bytes: PCAP files smaller than this threshold are considered invalid.
            min_flow_count: If a DataFrame has fewer rows than this, a warning is logged.
        """
        self.min_file_size_bytes = min_file_size_bytes
        self.min_flow_count = min_flow_count

    def copy_valid_pcaps(self, source_dir: str, target_dir: str) -> None:
        """
        Copies only valid PCAP files (>= self.min_file_size_bytes) from source_dir
        into target_dir. Logs any file that is too small but does not delete the original.

        Args:
            source_dir: The folder containing the original PCAPs.
            target_dir: The folder where valid PCAPs are copied.
        """
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        if not source_path.exists():
            logging.error(f"Source directory does not exist: {source_dir}")
            return

        target_path.mkdir(parents=True, exist_ok=True)
        pcap_files = list(source_path.glob("*.pcap"))

        if not pcap_files:
            logging.warning(f"No PCAP files found in {source_dir}. Nothing to copy.")
            return

        total_files = len(pcap_files)
        valid_count = 0
        skipped_count = 0

        for pcap in pcap_files:
            size = pcap.stat().st_size
            if size >= self.min_file_size_bytes:
                valid_count += 1
                destination = target_path / pcap.name
                shutil.copy2(str(pcap), str(destination))
                logging.info(
                    f"Copied: {pcap.name} ({size} bytes) -> {destination}"
                )
            else:
                skipped_count += 1
                logging.info(
                    f"Skipped (too small): {pcap.name} ({size} bytes), "
                    f"threshold={self.min_file_size_bytes}"
                )

        logging.info(
            f"DataInspector: Copied {valid_count}, skipped {skipped_count} out of {total_files} PCAPs "
            f"from {source_dir} to {target_dir}."
        )

    def check_flow_dataframe(self, df) -> None:
        """
        Logs the shape, checks for missing columns, and warns if the flow count is below the threshold.

        Args:
            df: A pandas DataFrame representing extracted flow data.
        """
        row_count, col_count = df.shape
        logging.info(f"Flow-DataFrame has {row_count} rows and {col_count} columns.")

        # Example of required columns
        required_cols = [
            "packets_per_second",
            "src2dst_bytes_per_second",
            "dst2src_bytes_per_second",
            "duration_seconds",
            "bytes_per_second",
            "packet_size_avg",
            "packet_ratio",
            "byte_ratio"
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logging.warning(f"Missing columns: {missing_cols}")

        if row_count < self.min_flow_count:
            logging.warning(
                f"DataFrame has only {row_count} flows, under threshold {self.min_flow_count}."
            )

