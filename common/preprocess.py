# preprocess.py
# --- Data loading and preprocessing steps ---
# preprocess.py
import pandas as pd
import logging
from typing import Optional
import pyarrow.parquet as pq

class Preprocessor:
    @staticmethod
    def load_parquet(file_path: str, columns=None) -> pd.DataFrame:
        """Step 1: Load LinkedIn parquet file"""
        if columns is None:
            columns = [
                'job_key', 'state', 'zipcode', 'correct_date',
                'first_scrape_timestamp', 'scraped_location', 'src'
            ]
        logging.info(f"Loading data from {file_path} ...")
        df = pd.read_parquet(file_path, columns=columns)
        logging.info(f"Loaded {len(df)} rows from {file_path}")
        return df

    @staticmethod
    def read_parquet_in_batches(file_path: str, batch_size: int = 50000, columns=None):
        """Read parquet file in batches for memory-efficient processing"""
        if columns is None:
            columns = [
                'job_key', 'state', 'zipcode', 'correct_date',
                'first_scrape_timestamp', 'scraped_location', 'src'
            ]
        
        logging.info(f"Reading parquet file in batches of {batch_size:,} rows from {file_path}")
        
        # Get total rows for progress tracking
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        logging.info(f"Total rows to process: {total_rows:,}")
        
        # Read in batches using pyarrow
        batch_num = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            batch_num += 1
            batch_df = batch.to_pandas()
            logging.info(f"Reading batch {batch_num} with {len(batch_df):,} rows")
            yield batch_df

    @staticmethod
    def rebuild_correct_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Build correct_date_filled from first_scrape_timestamp"""
        df['correct_date_filled'] = pd.to_datetime(df['first_scrape_timestamp'], errors='coerce')
        missing = df['correct_date_filled'].isna().sum()
        logging.info(f"Missing correct_date_filled after first parse: {missing}")
        return df

    @staticmethod
    def fill_null_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Step 3: Fill nulls in correct_date_filled using correct_date"""
        df['correct_date'] = pd.to_datetime(df['correct_date'])
        nulls_before = df['correct_date_filled'].isnull().sum()
        df['correct_date_filled'] = df['correct_date_filled'].fillna(df['correct_date'])
        nulls_after = df['correct_date_filled'].isnull().sum()
        logging.info(f"Filled {nulls_before - nulls_after} nulls in correct_date_filled using correct_date. Remaining nulls: {nulls_after}")
        return df

   