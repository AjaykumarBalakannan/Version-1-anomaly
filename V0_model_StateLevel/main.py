# main.py

import logging
import warnings
import os
from datetime import datetime
from pipeline import StateLevelAnomalyPipeline

warnings.filterwarnings("ignore")

today_str = datetime.now().strftime("%Y-%m-%d")
log_file_path = f"logs/V1_StateLevel_anomaly_pipeline_{today_str}.log"

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

logging.info("V1 State Level Model pipeline started. This should appear in both log file and console.")

if __name__ == "__main__":
    # Set up all file/data paths here (use env vars or relative paths)
    file_path = os.getenv('STATE_INPUT_PARQUET', 'input-files/linkedin_jan_2025.parquet')
    mapping_path = os.getenv('STATE_MAPPING_XLSX', 'data/Full_zipcodeFix.xlsx')
    zipcode_to_state_path = os.getenv('STATE_ZIPCODE_XLSX', 'data/Canaria_Zipcodes.xlsx')
    out_csv = os.getenv('STATE_OUTPUT_CSV', 'output/anomalies_state_V1.csv')
    batch_size = int(os.getenv('STATE_BATCH_SIZE', '50000'))

    # Log configuration
    logging.info(f"V1 State Level Model pipeline configuration:")
    logging.info(f"  Input file: {file_path}")
    logging.info(f"  Mapping file: {mapping_path}")
    logging.info(f"  Zipcode file: {zipcode_to_state_path}")
    logging.info(f"  Output file: {out_csv}")
    logging.info(f"  Batch size: {batch_size:,}")

    pipeline = StateLevelAnomalyPipeline(
        file_path,
        mapping_path,
        zipcode_to_state_path,
        out_csv,
        batch_size=batch_size
    )
    df_anom, pivot, summary = pipeline.run()
    if summary is not None and 'today_anomalies' in summary:
        print("Today's anomalies:")
        print(summary['today_anomalies'])
        if summary['today_anomalies'] is not None:
            today_str = datetime.now().strftime("%Y-%m-%d")
            summary['today_anomalies'].to_csv(f"output/anomalies_today_State_V1_{today_str}.csv", index=False)
    else:
        print("No anomaly summary available.")
