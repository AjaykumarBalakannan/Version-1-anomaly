# mongodb_hist.py
# --- Maintain rolling window of historical data in MongoDB ---

import pandas as pd
import logging
import time
from pymongo import ReplaceOne, MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, BulkWriteError
from common.mongo_utils import get_mongo_collection
import os

# Determine the grouping field based on model type (default to geo_level, use nlp_soc_code_rolled for SOC)
# GROUP_FIELD = os.getenv('SOC_MODEL', '0') == '1' and 'nlp_soc_code_rolled' or 'geo_level'

class HistoryManager:
    """
    MongoDB History Manager for US Level Anomaly Detection
    
    Manages daily job count data in MongoDB with rolling window functionality.
    Provides methods for upserting new data and loading historical data for analysis.
    """
    
    def __init__(self, mongo_uri, db_name, collection_name, group_field='geo_level'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.group_field = group_field
        self.date_col = 'date'
        self.group_cols = [self.group_field, 'src']
        self.job_count_col = 'job_count'
        
        # Initialize MongoDB connection
        self.collection = self._initialize_connection()
    
    # ============================================================================
    # CONNECTION MANAGEMENT
    # ============================================================================
    
    def _initialize_connection(self, max_retries=3, retry_delay=5):
        """
        Initialize MongoDB connection with retry logic
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retry attempts in seconds
            
        Returns:
            MongoDB collection object
            
        Raises:
            ConnectionError: If connection fails after all retries
        """
        logger = logging.getLogger(__name__)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting MongoDB connection (attempt {attempt + 1}/{max_retries}) to {self.mongo_uri}")
                
                # Create MongoDB client and test connection
                client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=10000)
                client.admin.command('ping')
                
                # Get database and collection
                db = client[self.db_name]
                collection = db[self.collection_name]
                
                logger.info("MongoDB connection successful")
                
                # Setup indexes for optimal performance
                self._setup_indexes(collection)
                
                return collection
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"MongoDB connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to MongoDB after {max_retries} attempts")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error connecting to MongoDB: {e}")
                raise
        
        # This should never be reached, but just in case
        raise ConnectionError("Failed to establish MongoDB connection")
    
    def _setup_indexes(self, collection):
        """
        Setup necessary indexes for optimal query performance
        
        Args:
            collection: MongoDB collection object
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Create compound index on the query fields used in upsert operations
            # This will convert COLLSCAN to IXSCAN for much faster queries
            index_name = "date_geo_src_idx"
            index_fields = [("date", 1), (self.group_field, 1), ("src", 1)]
            
            # Check if index already exists
            existing_indexes = collection.list_indexes()
            index_exists = any(idx.get('name') == index_name for idx in existing_indexes)
            
            if not index_exists:
                logger.info(f"Creating compound index: {index_name} on fields: {index_fields}")
                collection.create_index(index_fields, name=index_name)
                logger.info(f"Successfully created index: {index_name}")
            else:
                logger.info(f"Index {index_name} already exists, skipping creation")
                
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
            # Don't fail the entire connection if index creation fails
            pass
    
    # ============================================================================
    # DATA OPERATIONS - INSERT/UPSERT
    # ============================================================================
    
    def add_daily_counts_batch(self, daily_counts_df: pd.DataFrame):
        """
        Add daily job counts batch to MongoDB with upsert logic
        
        Args:
            daily_counts_df: DataFrame containing daily counts with columns:
                           ['date', 'geo_level', 'src', 'job_count']
        """
        logger = logging.getLogger(__name__)
        
        # Validation checks
        if self.collection is None:
            logger.error("MongoDB collection is not available")
            return
            
        if len(daily_counts_df) == 0:
            logger.warning("Empty daily counts batch, skipping")
            return
            
        try:
            # Log the data being processed
            self._log_batch_data(daily_counts_df)
            
            # Prepare upsert operations
            operations = self._prepare_upsert_operations(daily_counts_df)
            
            # Execute bulk upsert operations
            self._execute_bulk_upserts(operations)
            
            logger.info(f"Successfully processed daily counts batch with {len(daily_counts_df)} records")
            
        except Exception as e:
            logger.error(f"Error adding daily counts batch to MongoDB: {e}")
    
    def _log_batch_data(self, daily_counts_df: pd.DataFrame):
        """Log the data being upserted for debugging"""
        logger = logging.getLogger(__name__)
        logger.info("Upserting the following (date, geo_level, src, job_count) keys:")
        for _, row in daily_counts_df.iterrows():
            logger.info(f"  date={row['date']}, geo_level={row[self.group_field]}, src={row['src']}, job_count={row['job_count']}")
    
    def _prepare_upsert_operations(self, daily_counts_df: pd.DataFrame):
        """
        Prepare upsert operations for bulk write
        
        Args:
            daily_counts_df: DataFrame with daily counts
            
        Returns:
            List of UpdateOne operations
        """
        operations = []
        
        for _, row in daily_counts_df.iterrows():
            # Create unique _id for each (date, geo_level, src) combination
            unique_id = f"{row[self.group_field]}_{row['src']}_{row['date']}"
            
            # Create upsert operation to increment job_count
            operation = UpdateOne(
                {
                    'date': row['date'],
                    self.group_field: row[self.group_field], 
                    'src': row['src']
                },
                {
                    '$inc': {'job_count': row['job_count']},
                    '$setOnInsert': {
                        '_id': unique_id,
                        'date': row['date'],
                        self.group_field: row[self.group_field],
                        'src': row['src']
                    }
                },
                upsert=True
            )
            operations.append(operation)
        
        return operations
    
    def _execute_bulk_upserts(self, operations, chunk_size=10000):
        """
        Execute bulk upsert operations in chunks
        
        Args:
            operations: List of UpdateOne operations
            chunk_size: Number of operations per chunk
        """
        logger = logging.getLogger(__name__)
        
        if not operations:
            return
        
        total_chunks = (len(operations) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(operations), chunk_size):
            chunk = operations[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            logger.info(f"Upserting daily counts chunk {chunk_num}/{total_chunks} ({len(chunk)} operations)")

            try:
                result = self.collection.bulk_write(chunk, ordered=False)
                logger.info(f"Daily counts chunk {chunk_num} result: {result.upserted_count} upserts, {result.modified_count} modifications")
            except Exception as e:
                logger.error(f"Error upserting daily counts chunk {chunk_num}: {e}")
                continue
    
    # ============================================================================
    # DATA OPERATIONS - QUERY/LOAD
    # ============================================================================
    
    def load_complete_history(self, chunk_size: int = 100000) -> pd.DataFrame:
        """
        Load complete history from MongoDB in chunks and apply rolling window logic
        
        Args:
            chunk_size: Number of documents to load per chunk
            
        Returns:
            DataFrame with historical data after rolling window filtering
        """
        logger = logging.getLogger(__name__)
        
        if self.collection is None:
            logger.error("MongoDB collection is not available")
            return pd.DataFrame()
            
        try:
            logger.info("Loading complete history from MongoDB in chunks...")
            
            # Get total count and validate
            total_docs = self.collection.count_documents({})
            logger.info(f"Total documents to load: {total_docs}")
            
            if total_docs == 0:
                logger.warning("No history found in MongoDB")
                return pd.DataFrame()
            
            # Load data in chunks
            all_chunks = self._load_data_in_chunks(total_docs, chunk_size)
            
            # Combine and process data
            df_hist = self._combine_and_process_chunks(all_chunks)
            
            # Apply rolling window logic
            df_hist_filtered = self._apply_rolling_window(df_hist)
            
            return df_hist_filtered
            
        except Exception as e:
            logger.error(f"Error loading history from MongoDB: {e}")
            return pd.DataFrame()
    
    def _load_data_in_chunks(self, total_docs: int, chunk_size: int):
        """
        Load data from MongoDB in chunks
        
        Args:
            total_docs: Total number of documents to load
            chunk_size: Number of documents per chunk
            
        Returns:
            List of DataFrames, one per chunk
        """
        logger = logging.getLogger(__name__)
        all_chunks = []
        processed_docs = 0
        
        for skip in range(0, total_docs, chunk_size):
            chunk_num = (skip // chunk_size) + 1
            total_chunks = (total_docs + chunk_size - 1) // chunk_size
            
            logger.info(f"Loading chunk {chunk_num}/{total_chunks} (skip={skip}, limit={chunk_size})")
            
            # Load chunk
            cursor = self.collection.find({}).skip(skip).limit(chunk_size)
            chunk_df = pd.DataFrame(list(cursor))
            
            if len(chunk_df) == 0:
                break
            
            # Process chunk
            chunk_df = self._process_chunk_data(chunk_df)
            
            all_chunks.append(chunk_df)
            processed_docs += len(chunk_df)
            
            logger.info(f"Loaded chunk {chunk_num}: {len(chunk_df)} rows, total processed: {processed_docs}")
        
        return all_chunks
    
    def _process_chunk_data(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a chunk of data from MongoDB
        
        Args:
            chunk_df: Raw chunk DataFrame from MongoDB
            
        Returns:
            Processed chunk DataFrame
        """
        # Convert date columns back to proper format
        chunk_df[self.date_col] = pd.to_datetime(chunk_df[self.date_col], errors='coerce', format='mixed')
        chunk_df['date_only'] = chunk_df[self.date_col].dt.date
        
        # Drop MongoDB _id column
        if '_id' in chunk_df.columns:
            chunk_df = chunk_df.drop(columns=['_id'])
        
        return chunk_df
    
    def _combine_and_process_chunks(self, all_chunks):
        """
        Combine all chunks into a single DataFrame
        
        Args:
            all_chunks: List of chunk DataFrames
            
        Returns:
            Combined DataFrame
        """
        logger = logging.getLogger(__name__)
        
        if not all_chunks:
            logger.warning("No data loaded from MongoDB")
            return pd.DataFrame()
        
        df_hist = pd.concat(all_chunks, ignore_index=True)
        return df_hist
    
    def _apply_rolling_window(self, df_hist: pd.DataFrame, n_days: int = 45) -> pd.DataFrame:
        """
        Apply rolling window logic to keep only the last N unique dates per group
        
        Args:
            df_hist: Historical data DataFrame
            n_days: Number of days to keep in rolling window
            
        Returns:
            Filtered DataFrame with rolling window applied
        """
        logger = logging.getLogger(__name__)
        logger.info("Applying rolling window logic to history...")
        
        def keep_last_n_unique_dates(group, n=n_days):
            """Keep only the last N unique dates for each group"""
            unique_dates = sorted(group['date_only'].unique(), reverse=True)
            if len(unique_dates) > n:
                cutoff_date = unique_dates[n-1]
                return group[group['date_only'] >= cutoff_date]
            return group
        
        # Apply rolling window per geo_level and src
        df_hist_filtered = df_hist.groupby(self.group_cols).apply(keep_last_n_unique_dates).reset_index(drop=True)
        
        logger.info(f"History loaded: {len(df_hist_filtered)} records after rolling window")
        logger.info(f"Date range: {df_hist_filtered['date'].min()} to {df_hist_filtered['date'].max()}")
        
        return df_hist_filtered
    
    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================
    
    def clear_old_data(self, days_to_keep=90):
        """
        Clear old data from MongoDB
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        logger = logging.getLogger(__name__)
        
        try:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_to_keep)
            result = self.collection.delete_many({
                'inserted_at': {'$lt': cutoff_date}
            })
            logger.info(f"Cleared {result.deleted_count} old records from MongoDB")
        except Exception as e:
            logger.error(f"Error clearing old data: {e}")
    
    def get_collection_stats(self):
        """
        Get collection statistics
        
        Returns:
            Dictionary with collection statistics
        """
        logger = logging.getLogger(__name__)
        
        try:
            total_docs = self.collection.count_documents({})
            latest_date = self.collection.find().sort('date', -1).limit(1)
            latest_record = list(latest_date)
            stats = {
                'total_documents': total_docs,
                'latest_date': latest_record[0]['date'] if latest_record else None
            }
            logger.info(f"Collection stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection"""
        logger = logging.getLogger(__name__)
        logger.info("MongoDB connection will be closed when process ends")
