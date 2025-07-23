from common.preprocess import Preprocessor
from soc_groupings import rollup_soc_code, is_seasonal_burst
from common.mongodb_hist import HistoryManager
from detect import AnomalyDetector
import logging
import pandas as pd
from common.utils.memlogger import log_memory
import os

class SOCLevelAnomalyPipeline:
    def __init__(self, file_path, out_csv, batch_size=50000):
        self.file_path = file_path
        self.out_csv = out_csv
        self.batch_size = batch_size

    def read_parquet_in_batches(self):
        return Preprocessor.read_parquet_in_batches(self.file_path, self.batch_size)

    def process_batch(self, batch_df):
        logger = logging.getLogger(__name__)
        batch_df = batch_df.copy()
        batch_df = Preprocessor.rebuild_correct_timestamp(batch_df)
        batch_df = Preprocessor.fill_null_dates(batch_df)
        # Ensure job_count exists for grouping
        if 'job_count' not in batch_df.columns:
            batch_df['job_count'] = 1
        if 'nlp_soc_code' in batch_df.columns:
            mean_counts = batch_df.groupby('nlp_soc_code')['job_count'].mean()
            mean_counts = pd.Series(mean_counts.values, index=mean_counts.index)
            low_volume_6d = mean_counts[mean_counts < 50].index
            batch_df['nlp_soc_code_rolled'] = batch_df['nlp_soc_code'].apply(
                lambda x: rollup_soc_code(x, level=1) if x in low_volume_6d else x
            )
        else:
            batch_df['nlp_soc_code_rolled'] = ''
        log_memory("process_batch")
        return batch_df

    def aggregate_batch_to_daily_counts(self, batch_df):
        logger = logging.getLogger(__name__)
        batch_df = batch_df.copy()
        batch_df['correct_date_filled'] = pd.to_datetime(batch_df['correct_date_filled'])
        batch_df['date'] = batch_df['correct_date_filled'].dt.strftime('%Y-%m-%d')
        grouped = batch_df.groupby(['date', 'nlp_soc_code_rolled', 'src'])['job_count'].sum().reset_index()
        logger.info(f"Aggregated batch to {len(grouped)} daily count records")
        return grouped

    def run_with_batching(self):
        logger = logging.getLogger(__name__)
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://anomaly-detection-mongodb:27017')
        history_manager = HistoryManager(
            mongo_uri=mongo_uri,
            db_name="anomaly_detection",
            collection_name="soc_level_data",
            group_field='nlp_soc_code_rolled'
        )
        log_memory("run_with_batching start")
        logger.info(f"Reading parquet file in batches of {self.batch_size:,} rows from {self.file_path}")
        total_rows = 0
        batch_num = 0
        for batch_df in self.read_parquet_in_batches():
            batch_num += 1
            total_rows += len(batch_df)
            processed_batch = self.process_batch(batch_df)
            daily_counts = self.aggregate_batch_to_daily_counts(processed_batch)
            history_manager.add_daily_counts_batch(daily_counts)
            log_memory(f"batch_{batch_num}")
            logger.info(f"Completed {total_rows:,} rows total")
            del batch_df, processed_batch, daily_counts
        logger.info(f"All batches processed. Total rows: {total_rows:,}")
        logger.info("Loading complete history for anomaly detection...")
        df_hist = history_manager.load_complete_history()
        if len(df_hist) == 0:
            logger.error("No history data available for anomaly detection")
            return None, None, None
        logger.info("Running anomaly detection...")
        detector = AnomalyDetector()
        df_anom, pivot, summary = detector.detect_anomalies(df_hist, csv_path=self.out_csv)
        return df_anom, pivot, summary

    def run(self):
        return self.run_with_batching()
