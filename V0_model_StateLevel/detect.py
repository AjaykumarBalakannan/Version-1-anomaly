# detect.py
import pandas as pd
import logging

class StateAnomalyDetector:
    @staticmethod
    def detect_anomalies(
        df,
        window_size=30,
        anomaly_threshold=3,
        csv_path=None
    ):
        """
        Runs windowed Z-score anomaly detection for job posting volumes at state x source x date level.
        Works with pre-aggregated MongoDB data that already contains daily counts per state.
        """
        logger = logging.getLogger(__name__)
        df = df.copy()
        
        # MongoDB data already has 'date', 'state', 'src', 'job_count' columns
        # Create date_only from the existing 'date' column
        df['date_only'] = pd.to_datetime(df['date']).dt.date
        
        # The data is already aggregated, so we don't need to group again
        # Just rename for consistency with the rest of the code
        job_counts = df[['date_only', 'state', 'src', 'job_count']].copy()
        
        # Create pivot table from the pre-aggregated data
        pivot = job_counts.pivot_table(
            index=['state', 'src'],
            columns='date_only',
            values='job_count'
        ).sort_index()
        
        logger.info("Pivot table (state, src x date):\n%s", pivot.head())
        logger.info(f"Pivot shape: {pivot.shape}")
        logger.info(f"Date range: {pivot.columns.min()} to {pivot.columns.max()}")
        
        # Calculate rolling statistics
        rolling_mean = pivot.rolling(window=window_size, min_periods=7, axis=1).mean()
        rolling_std = pivot.rolling(window=window_size, min_periods=7, axis=1).std()
        expected_high = rolling_mean + anomaly_threshold * rolling_std
        expected_low = rolling_mean - anomaly_threshold * rolling_std
        z_scores = (pivot - rolling_mean) / rolling_std
        
        # Anomaly flags
        high_anomalies = z_scores > anomaly_threshold
        low_anomalies = z_scores < -anomaly_threshold
        
        logger.info(f"Anomaly threshold: +/-{anomaly_threshold}")
        logger.info(f"High anomalies found: {high_anomalies.sum().sum()}")
        logger.info(f"Low anomalies found: {low_anomalies.sum().sum()}")
        
        # Top 5 (state,src) with Most High/Low Anomalies
        high_anomaly_counts = high_anomalies.sum(axis=1).sort_values(ascending=False)
        top5_high = high_anomaly_counts.head(5)
        low_anomaly_counts = low_anomalies.sum(axis=1).sort_values(ascending=False)
        top5_low = low_anomaly_counts.head(5)
        
        logger.info(f"Top 5 State-Source with Most High Anomalies:\n{top5_high}")
        logger.info(f"Top 5 State-Source with Most Low Anomalies:\n{top5_low}")
        
        # Flatten to long-form DataFrame
        df_anom = z_scores.stack().reset_index()
        df_anom.columns = ['state', 'src', 'date', 'Z_score']
        
        # Add rolling statistics
        for name, table in [
            ("high_expected_job_count", expected_high),
            ("low_expected_job_count", expected_low),
            ("rolling_mean", rolling_mean),
            ("rolling_std", rolling_std)
        ]:
            temp = table.stack().reset_index()
            temp.columns = ['state', 'src', 'date', name]
            df_anom = df_anom.merge(temp, on=['state', 'src', 'date'], how='left')
        
        # Merge with actual job counts
        df_anom = df_anom.merge(
            job_counts.rename(columns={'date_only': 'date'}),
            on=['state', 'src', 'date'],
            how='left'
        )
        
        # Add anomaly flags
        df_anom['is_high_anomaly'] = (
            (df_anom['job_count'] > df_anom['high_expected_job_count']) &
            (df_anom['Z_score'] > anomaly_threshold)
        )
        df_anom['is_low_anomaly'] = (
            (df_anom['job_count'] < df_anom['low_expected_job_count']) &
            (df_anom['Z_score'] < -anomaly_threshold)
        )
        df_anom['is_anomaly'] = df_anom['is_high_anomaly'] | df_anom['is_low_anomaly']
        df_anom['anomaly_type'] = 'none'
        df_anom.loc[df_anom['is_high_anomaly'], 'anomaly_type'] = 'high'
        df_anom.loc[df_anom['is_low_anomaly'], 'anomaly_type'] = 'low'
        df_anom['date_only'] = pd.to_datetime(df_anom['date']).dt.date
        
        # Focus: Filter anomalies for ONLY the most recent date (today)
        today_date = df['date_only'].max()
        today_anomalies = df_anom[
            (df_anom['date_only'] == today_date) &
            (df_anom['is_anomaly'])
        ].copy()
        
        if not today_anomalies.empty:
            logger.warning(f"Anomalies detected for {today_date}:\n{today_anomalies}")
        else:
            logger.info(f"No anomalies detected for {today_date}")
        
        # Summary
        summary = {
            'total_high': int(high_anomalies.sum().sum()),
            'total_low': int(low_anomalies.sum().sum()),
            'top5_high_state_src': top5_high.to_dict(),
            'top5_low_state_src': top5_low.to_dict(),
            'today_anomalies': today_anomalies
        }
        
        if csv_path:
            df_anom_to_save = df_anom.copy()
            if "low_expected_job_count" in df_anom_to_save.columns:
                df_anom_to_save["low_expected_job_count"] = df_anom_to_save["low_expected_job_count"].clip(lower=0)
            df_anom_to_save.to_csv(csv_path, index=False)
            logger.info(f"Anomaly CSV saved at {csv_path}")
        
        return df_anom, pivot, summary
