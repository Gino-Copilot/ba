import logging
import pandas as pd
from pathlib import Path

class DataInspector:
    """
    Checks DataFrame or PCAP properties before/after extraction.
    In particular, it deletes PCAP files that are too small
    and logs the percentage of deletions.
    Also warns if certain columns or enough flows are missing in the DataFrame.
    """

    def __init__(self, min_file_size_bytes=50_000, min_flow_count=10):
        """
        :param min_file_size_bytes: PCAP files smaller than this threshold are physically deleted.
        :param min_flow_count: If a DataFrame has fewer rows than this, a warning is logged.
        """
        self.min_file_size_bytes = min_file_size_bytes
        self.min_flow_count = min_flow_count

    def remove_small_pcaps(self, pcap_dir: str) -> None:
        """
        Scans the directory for small PCAP files (under self.min_file_size_bytes),
        and physically deletes them so the FeatureExtractor won't see them later.
        Logs how many files were deleted and the percentage.

        :param pcap_dir: Path to the directory containing .pcap files.
        """
        pcap_path = Path(pcap_dir)
        pcap_files = list(pcap_path.glob("*.pcap"))
        total_files = len(pcap_files)
        deleted_files = 0

        if total_files == 0:
            logging.info(f"No PCAP files found in: {pcap_dir}")
            return

        for pcap_file in pcap_files:
            size = pcap_file.stat().st_size
            if size < self.min_file_size_bytes:
                try:
                    pcap_file.unlink()  # Physically delete
                    deleted_files += 1
                    logging.warning(
                        f"Deleted {pcap_file.name} ({size} bytes) in {pcap_dir} "
                        f"(threshold={self.min_file_size_bytes})."
                    )
                except Exception as e:
                    logging.error(f"Error deleting {pcap_file.name}: {e}")

        if deleted_files > 0:
            percent_deleted = 100.0 * deleted_files / total_files
            logging.info(
                f"Deleted {deleted_files} out of {total_files} PCAPs in {pcap_dir} "
                f"({percent_deleted:.1f}%)."
            )
        else:
            logging.info(
                f"All {total_files} PCAP files in {pcap_dir} "
                f"are above {self.min_file_size_bytes} bytes."
            )

    def check_flow_dataframe(self, df: pd.DataFrame) -> None:
        """
        Logs the shape, missing columns, or flow count checks.
        Warns if important columns are missing or if too few flows are present.
        """
        row_count, col_count = df.shape
        logging.info(f"Flow-DataFrame has {row_count} rows and {col_count} columns.")

        # Falls du bestimmte Spalten erwartest:
        required_cols = ["bidirectional_packets", "src2dst_bytes", "dst2src_bytes"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logging.warning(f"Missing columns: {missing_cols}")

        # Mindestanzahl an Flows
        if row_count < self.min_flow_count:
            logging.warning(
                f"DataFrame has only {row_count} flows, under threshold {self.min_flow_count}."
            )
            # Optional: könntest hier return False machen, wenn du das verarbeiten willst
