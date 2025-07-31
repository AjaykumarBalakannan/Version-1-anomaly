# detect.py
# --- Traditional anomaly detection logic for US-level data ---

import pandas as pd
import numpy as np
import logging

class AnomalyDetector:
    @staticmethod
    def detect_anomalies(
        df,
        window_size=30,
        anomaly_threshold=3,
        mad_multiplier=3.5,
        csv_path=None
    ):
        logger = logging.getLogger(__name__)
        df = df.copy()

        # The input df now contains daily job counts, not individual job records
        # Expected columns: ['date', 'nlp_soc_code_rolled', 'src', 'job_count']
        
        # Convert date to proper format
        df['date'] = pd.to_datetime(df['date'])
        df['date_only'] = df['date'].dt.date

        logger.info("Daily job counts data:\n%s", df.head())
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Step 1: Pivot (nlp_soc_code_rolled, src) x date matrix
        pivot = df.pivot_table(
            index=['nlp_soc_code_rolled', 'src'],
            columns='date',
            values='job_count'
        ).sort_index()

        logger.info("Pivot table (nlp_soc_code_rolled, src x date):\n%s", pivot.head())

        # Step 2: Compute rolling stats
        rolling_mean = pivot.rolling(window=window_size, min_periods=7, axis=1).mean()
        rolling_std = pivot.rolling(window=window_size, min_periods=7, axis=1).std()
        rolling_median = pivot.rolling(window=window_size, min_periods=7, axis=1).median()
        rolling_mad = pivot.rolling(window=window_size, min_periods=7, axis=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )

        # Step 3: Expected thresholds
        expected_high = rolling_mean + anomaly_threshold * rolling_std
        expected_low = rolling_mean - anomaly_threshold * rolling_std
        expected_low_mad = rolling_median - mad_multiplier * rolling_mad  # Consistent MAD scale

        # Step 4: Z-score
        z_scores = (pivot - rolling_mean) / rolling_std

        # Step 5: Flatten to long form
        df_anom = z_scores.stack().reset_index()
        df_anom.columns = ['nlp_soc_code_rolled', 'src', 'date', 'Z_score']

        # Flatten and merge thresholds
        for name, table in [
            ("high_expected_job_count", expected_high),
            ("low_expected_job_count", expected_low),
            ("low_expected_job_count_mad", expected_low_mad),
            ("rolling_mean", rolling_mean),
            ("rolling_std", rolling_std),
            ("rolling_median", rolling_median),
            ("rolling_mad", rolling_mad)
        ]:
            temp = table.stack().reset_index()
            temp.columns = ['nlp_soc_code_rolled', 'src', 'date', name]
            df_anom = df_anom.merge(temp, on=['nlp_soc_code_rolled', 'src', 'date'], how='left')

        # Merge actual job counts
        df_anom = df_anom.merge(
            df[['nlp_soc_code_rolled', 'src', 'date', 'job_count']],
            on=['nlp_soc_code_rolled', 'src', 'date'],
            how='left'
        )

        # Step 6: Detect anomalies
        df_anom['is_high_anomaly'] = (
            (df_anom['job_count'] > df_anom['high_expected_job_count']) &
            (df_anom['Z_score'] > anomaly_threshold)
        )

        df_anom['is_low_anomaly'] = (
            (df_anom['job_count'] < df_anom['low_expected_job_count']) |
            (df_anom['job_count'] < df_anom['low_expected_job_count_mad'])
        )

        df_anom['is_anomaly'] = df_anom['is_high_anomaly'] | df_anom['is_low_anomaly']

        # Step 7: Add date info
        df_anom['date_only'] = pd.to_datetime(df_anom['date']).dt.date

        # Step 8: Summary statistics
        summary = {
            'total_records': len(df_anom),
            'high_anomalies': df_anom['is_high_anomaly'].sum(),
            'low_anomalies': df_anom['is_low_anomaly'].sum(),
            'total_anomalies': df_anom['is_anomaly'].sum(),
            'anomaly_rate': df_anom['is_anomaly'].mean(),
            'date_range': f"{df_anom['date'].min()} to {df_anom['date'].max()}",
            'segments': len(df_anom[['nlp_soc_code_rolled', 'src']].drop_duplicates())
        }

        logger.info("Anomaly detection summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        # Step 9: Save results
        if csv_path:
            df_anom.to_csv(csv_path, index=False)
            logger.info(f"Anomaly detection results saved to {csv_path}")

        return df_anom, pivot, summary
