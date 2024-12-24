import logging
import numpy as np
import pandas as pd

class DataCleaner:
    """
    Cleans the dataset by:
      1) Checking if certain columns exist before dropping rows.
      2) Dropping rows with too few packets (if a reference column exists).
      3) Optionally imputing missing values in numeric columns via median.
    """

    def __init__(self, min_packet_threshold=5, impute_with_median=True):
        """
        :param min_packet_threshold: Minimum number of packets required to keep a row.
        :param impute_with_median: Whether to fill missing numeric values with the median.
        """
        self.min_packet_threshold = min_packet_threshold
        self.impute_with_median = impute_with_median

        # Columns that might not exist, but if they do, we use them for dropping rows.
        self.packet_column = "bidirectional_packets"
        # If your dataset does not produce 'bidirectional_packets', adjust this column name
        # or remove the dropping logic if you do not need it.

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the cleaning steps in the correct order:
        1) If self.packet_column exists, drops rows with fewer than min_packet_threshold.
        2) Imputes missing values in numeric columns if impute_with_median is True.

        :param df: The raw DataFrame to clean.
        :return: A cleaned DataFrame. Might become empty if rows are heavily filtered.
        """
        if df.empty:
            logging.warning("DataCleaner: Received an empty DataFrame. Returning as is.")
            return df

        df_cleaned = df.copy()

        # Step 1: Drop rows based on packet_column if it exists
        df_cleaned = self._drop_few_packets(df_cleaned)

        # Step 2: Optional median imputation
        if self.impute_with_median:
            df_cleaned = self._impute_missing_median(df_cleaned)

        return df_cleaned

    def _drop_few_packets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows where the packet_column is below min_packet_threshold, if packet_column exists.
        Otherwise, it logs a warning and returns df unchanged.
        """
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
                f"DataCleaner: Dropped {dropped} rows with '{self.packet_column}' < {self.min_packet_threshold}."
            )
        return df

    def _impute_missing_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills NaN in numeric columns with the median of each column.
        """
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
