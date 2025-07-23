# main.py
# --- Entrypoint for running the SOC-level pipeline, with logging setup ---

import logging
import warnings
import os
from datetime import datetime
from pipeline import SOCLevelAnomalyPipeline

warnings.filterwarnings("ignore")

# --- Logging setup: both file and console, log file named as today ---
today_str = datetime.now().strftime("%Y-%m-%d")
log_file_path = f"logs/V0_SOCLevel_anomaly_pipeline_{today_str}.log"

# Remove old logging handlers (avoid duplicate logs if re-run)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logging.info("V0 SOC Level Model pipeline started. Test: This should appear in both log file and console.")

if __name__ == "__main__":
    # --- Set up all file/data paths here using environment variables ---
    file_path = os.getenv('SOC_INPUT_PARQUET', 'input-files/linkedin_jan_2025.parquet')
    out_csv = os.getenv('SOC_OUTPUT_CSV', 'output/anomalies_SOC.csv')
    batch_size = int(os.getenv('SOC_BATCH_SIZE', '50000'))

    # Log configuration
    logging.info(f"V0 SOC Level Model pipeline configuration:")
    logging.info(f"  Input file: {file_path}")
    logging.info(f"  Output file: {out_csv}")
    logging.info(f"  Batch size: {batch_size:,}")

    # --- Run the batch processing pipeline ---
    pipeline = SOCLevelAnomalyPipeline(
        file_path,
        out_csv,
        batch_size=batch_size
    )
    
    logging.info(f"Starting batch processing with batch size: {batch_size:,}")
    df_anom, pivot, summary = pipeline.run()
    
    if summary is not None and 'today_anomalies' in summary:
        print("Today's anomalies:")
        print(summary['today_anomalies'])
        if summary['today_anomalies'] is not None:
            today_str = datetime.now().strftime("%Y-%m-%d")
            summary['today_anomalies'].to_csv(f"output/anomalies_today_SOC_{today_str}.csv", index=False)
    else:
        print("No anomaly summary available.")

