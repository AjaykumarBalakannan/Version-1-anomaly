# pipeline.py
import logging
import os
import pandas as pd
from common.preprocess import Preprocessor
from common.mongodb_hist import HistoryManager
from common.utils.memlogger import log_memory
from state_impute import StateImputer
from detect import StateAnomalyDetector
from config import US_STATES, US_STATES_FULL

class StateLevelAnomalyPipeline:
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
        
        log_memory(f"processed batch - RSS memory")
        return batch_df

    def aggregate_batch_to_daily_counts(self, batch_df):
        """Aggregate batch to daily job counts per state"""
        logger = logging.getLogger(__name__)
        
        # Create a copy to avoid modifying the original batch
        batch_df = batch_df.copy()
        
        # Ensure date is in proper format
        batch_df['correct_date_filled'] = pd.to_datetime(batch_df['correct_date_filled'])
        batch_df['date'] = batch_df['correct_date_filled'].dt.strftime('%Y-%m-%d')
        batch_df['state'] = batch_df['correct_state']
        
        # Aggregate to daily counts per state
        daily_counts = (
            batch_df.groupby(['date', 'state', 'src'])
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
        
        # Initialize history manager with environment variables
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://anomaly-detection-mongodb:27017')
        db_name = os.getenv('APP_DB', 'state_anomaly_db')
        collection_name = "state_job_history"
        
        history_manager = HistoryManager(
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name=collection_name,
            group_field='state'
        )
        log_memory("StateHistoryManager - RSS memory")
        
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
        logger.info("Loading complete state-level history for anomaly detection...")
        df_hist = history_manager.load_complete_history()
        
        if len(df_hist) == 0:
            logger.error("No state-level history data available for anomaly detection")
            return None, None, None
        
        # Run anomaly detection
        logger.info("Running state-level anomaly detection...")
        detector = StateAnomalyDetector()
        df_anom, pivot, summary = detector.detect_anomalies(df_hist, csv_path=self.out_csv)
        
        return df_anom, pivot, summary

    def run(self):
        """Main entry point"""
        return self.run_with_batching()
