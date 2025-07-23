# main.py
# --- Entrypoint for running the US-level pipeline, with logging setup ---

import logging
import warnings
import os
from datetime import datetime
from pipeline import USLevelAnomalyPipeline

warnings.filterwarnings("ignore")

# --- Logging setup: both file and console, log file named as today ---
today_str = datetime.now().strftime("%Y-%m-%d")
log_file_path = f"logs/V0_USLevel_anomaly_pipeline_{today_str}.log"

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

logging.info("V0 US Level Model pipeline started. Test: This should appear in both log file and console.")

if __name__ == "__main__":
    # --- Set up all file/data paths here using environment variables ---
    file_path = os.getenv('US_INPUT_PARQUET', 'input-files/linkedin_jan_2025.parquet')
    mapping_path = os.getenv('US_MAPPING_XLSX', 'data/Full_zipcodeFix.xlsx')
    zipcode_to_state_path = os.getenv('US_ZIPCODE_XLSX', 'data/Canaria_Zipcodes.xlsx')
    out_csv = os.getenv('US_OUTPUT_CSV', 'output/anomalies_US.csv')
    batch_size = int(os.getenv('US_BATCH_SIZE', '50000'))

    # Log configuration
    logging.info(f"V0 US Level Model pipeline configuration:")
    logging.info(f"  Input file: {file_path}")
    logging.info(f"  Mapping file: {mapping_path}")
    logging.info(f"  Zipcode file: {zipcode_to_state_path}")
    logging.info(f"  Output file: {out_csv}")
    logging.info(f"  Batch size: {batch_size:,}")

    # --- Run the batch processing pipeline ---
    pipeline = USLevelAnomalyPipeline(
        file_path, 
        mapping_path, 
        zipcode_to_state_path, 
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
            summary['today_anomalies'].to_csv(f"output/anomalies_today_US_{today_str}.csv", index=False)
    else:
        print("No anomaly summary available.")

