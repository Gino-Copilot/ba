# file: data_cleaner.py

import logging
import numpy as np
import pandas as pd


class DataCleaner:
    """
    Cleans a flow-based DataFrame by:
      - Dropping rows with fewer packets_per_second than a threshold
      - Optionally imputing NaN values in numeric columns with the median
    """

    def __init__(self, min_packet_threshold=5, impute_with_median=True):
        """
        Args:
            min_packet_threshold: Rows with 'packets_per_second' below this are removed.
            impute_with_median: Whether to replace NaN in numeric columns with the column median.
        """
        self.min_packet_threshold = min_packet_threshold
        self.impute_with_median = impute_with_median
        self.packet_column = "packets_per_second"

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logging.warning("DataCleaner: Received an empty DataFrame. Returning as is.")
            return df

        df_cleaned = df.copy()

        # Step 1: drop rows that have 'packets_per_second' < threshold
        df_cleaned = self._drop_few_packets(df_cleaned)

        # Step 2: optionally impute missing numeric columns with median
        if self.impute_with_median:
            df_cleaned = self._impute_missing_median(df_cleaned)

        return df_cleaned

    def _drop_few_packets(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.packet_column not in df.columns:
            logging.warning(
                f"DataCleaner: Column '{self.packet_column}' not found. Skipping drop based on packets."
            )
            return df

        before_count = len(df)
        df = df[df[self.packet_column] >= self.min_packet_threshold]
        after_count = len(df)
        dropped = before_count - after_count

        if dropped > 0:
            logging.info(
                f"DataCleaner: Dropped {dropped} rows where '{self.packet_column}' < {self.min_packet_threshold}."
            )
        return df

    def _impute_missing_median(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                median_val = df[col].median(skipna=True)
                df[col].fillna(median_val, inplace=True)
                logging.info(
                    f"DataCleaner: Filled {null_count} NaNs in column '{col}' with median={median_val:.3f}"
                )
        return df
