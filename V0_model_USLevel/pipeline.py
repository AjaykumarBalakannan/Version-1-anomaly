from common.preprocess import Preprocessor
from state_impute import StateImputer
from common.mongodb_hist import HistoryManager
from detect import AnomalyDetectorV1, detect_anomalies_v1
import logging
import pandas as pd
from common.utils.memlogger import log_memory
import os


class USLevelAnomalyPipeline:
    def __init__(self, file_path, mapping_path, zipcode_to_state_path, out_csv, batch_size=50000):
        self.file_path = file_path
        self.mapping_path = mapping_path
        self.zipcode_to_state_path = zipcode_to_state_path
        self.out_csv = out_csv
        self.batch_size = batch_size

    def read_parquet_in_batches(self):
        """Read parquet file in batches to manage memory"""
        return Preprocessor.read_parquet_in_batches(self.file_path, self.batch_size)

    def process_batch(self, batch_df, zipcode_to_state):
        """Process a single batch through preprocessing and state imputation"""
        logger = logging.getLogger(__name__)
        
        # Create a copy to avoid modifying the original batch
        batch_df = batch_df.copy()
        
        # Step 1: Preprocess the batch
        batch_df = Preprocessor.rebuild_correct_timestamp(batch_df)
        batch_df = Preprocessor.fill_null_dates(batch_df)
        
        # Step 2: State imputation
        batch_df = StateImputer.correct_short_zipcodes_fast_safe(batch_df, self.mapping_path)
        batch_df = StateImputer.impute_states(batch_df, zipcode_to_state)
        
        # Step 3: Assign US level tag
        batch_df['geo_level'] = 'US'
        
        log_memory(f"processed batch - RSS memory")
        return batch_df

    def aggregate_batch_to_daily_counts(self, batch_df):
        """Aggregate batch to daily job counts per segment"""
        logger = logging.getLogger(__name__)
        
        # Create a copy to avoid modifying the original batch
        batch_df = batch_df.copy()
        
        # Ensure date is in proper format
        batch_df['correct_date_filled'] = pd.to_datetime(batch_df['correct_date_filled'])
        batch_df['date'] = batch_df['correct_date_filled'].dt.strftime('%Y-%m-%d')
        
        # Aggregate to daily counts per segment
        daily_counts = (
            batch_df.groupby(['date', 'geo_level', 'src'])
                   .size()
                   .reset_index()
                   .rename(columns={0: 'job_count'})
        )
        
        logger.info(f"Aggregated batch to {len(daily_counts)} daily count records")
        return daily_counts

    def run_with_batching(self):
        """Run the pipeline with batch processing and daily count aggregation"""
        logger = logging.getLogger(__name__)
        
        # Load zipcode to state mapping once
        zipcode_to_state = StateImputer.get_zipcode_to_state(self.zipcode_to_state_path)
        log_memory("get_zipcode_to_state - RSS memory")
        
        # Initialize history manager
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://anomaly-detection-mongodb:27017')
        history_manager = HistoryManager(
            mongo_uri=mongo_uri,
            db_name="anomaly_detection",
            collection_name="us_job_history",
            group_field='geo_level'
        )
        log_memory("HistoryManager - RSS memory")
        
        logger.info(f"Reading parquet file in batches of {self.batch_size:,} rows from {self.file_path}")
        
        total_rows = 0
        batch_num = 0
        
        # Process each batch
        for batch_df in self.read_parquet_in_batches():
            batch_num += 1
            total_rows += len(batch_df)
            
            # Process the batch
            processed_batch = self.process_batch(batch_df, zipcode_to_state)
            
            # Aggregate to daily counts
            daily_counts = self.aggregate_batch_to_daily_counts(processed_batch)
            
            # Store daily counts to MongoDB with upsert logic
            history_manager.add_daily_counts_batch(daily_counts)
            
            log_memory(f"stored batch {batch_num} - RSS memory")
            logger.info(f"Completed {total_rows:,} rows total")
            
            # Clear batch from memory
            del batch_df, processed_batch, daily_counts
        
        logger.info(f"All batches processed. Total rows: {total_rows:,}")
        
        # Load complete history for anomaly detection
        logger.info("Loading complete history for anomaly detection...")
        df_hist = history_manager.load_complete_history()
        
        if len(df_hist) == 0:
            logger.error("No history data available for anomaly detection")
            return None, None, None
        
        # Run anomaly detection
        logger.info("Running V1 anomaly detection...")
        logger.info("V1 Models: iforest, hbos, copod, knn, svm, ocsvm, cblof")
        logger.info("V1 Features: rolling means (7,14,28), std, mad, pct_change, z-scores, temporal features")
        
        df_anom, pivot, summary = detect_anomalies_v1(
            df_hist, 
            csv_path=self.out_csv,
            models=("iforest", "hbos", "copod", "knn", "svm", "ocsvm", "cblof"),
            alert_budget_per_week=2,
            cooldown_days=1,
            quantile_bounds=(0.95, 0.999),
            verbose=True
        )
        
        # Add V1 specific logging
        logger.info(f"V1 Anomaly Detection completed:")
        logger.info(f"  Total records: {summary.get('total_records', 0)}")
        logger.info(f"  Total anomalies: {summary.get('total_anomalies', 0)}")
        logger.info(f"  Anomaly rate: {summary.get('anomaly_rate', 0):.4f}")
        logger.info(f"  Segments: {summary.get('segments', 0)}")
        logger.info(f"  Quantile bounds: {summary.get('quantile_bounds', (0, 0))}")
        
        return df_anom, pivot, summary

    def run(self):
        """Main entry point"""
        return self.run_with_batching()
