# State-Level Job Posting Anomaly Detection

This module detects anomalies in state-level job posting data using a batch processing pipeline and MongoDB for history storage. It is designed to run in a containerized environment with a dedicated MongoDB instance.

## Features
- Batch processing of large parquet files
- State-level aggregation and anomaly detection
- Rolling mean/std-based anomaly logic (no median/MAD)
- MongoDB-based history management (separate container)
- Configurable via environment variables

## Folder Structure
```
V0_Model_statexsrc/
  - main.py              # Entrypoint for the pipeline
  - pipeline.py          # Batch pipeline logic
  - detect.py            # State-level anomaly detection
  - history.py           # MongoDB history manager
  - preprocess.py        # Data loading and preprocessing
  - state_impute.py      # State imputation utilities
  - config.py            # US state codes and names
```

## Environment Variables
| Variable                | Description                                 | Default                        |
|-------------------------|---------------------------------------------|--------------------------------|
| STATE_INPUT_PARQUET     | Path to input parquet file                  | data/linkedin_may_2025.parquet |
| STATE_MAPPING_XLSX      | Path to zipcode fix Excel file              | data/Full_zipcodeFix.xlsx      |
| STATE_ZIPCODE_XLSX      | Path to zipcode-to-state Excel file         | data/Canaria_Zipcodes.xlsx     |
| STATE_OUTPUT_CSV        | Output CSV for anomalies                    | output/anomalies_state.csv     |
| STATE_BATCH_SIZE        | Batch size for processing                   | 50000                          |
| MONGO_URI               | MongoDB URI (state-level)                   | mongodb://state-mongodb:27017  |

## Docker Compose Example
Add the following service to your `docker-compose.yml`:

```yaml
services:
  state-anomaly-app:
    build: ./V0_Model_statexsrc
    container_name: state-anomaly-app
    depends_on:
      - state-mongodb
    environment:
      - MONGO_URI=mongodb://state-mongodb:27017
      - STATE_INPUT_PARQUET=data/linkedin_may_2025.parquet
      - STATE_MAPPING_XLSX=data/Full_zipcodeFix.xlsx
      - STATE_ZIPCODE_XLSX=data/Canaria_Zipcodes.xlsx
      - STATE_OUTPUT_CSV=output/anomalies_state.csv
      - STATE_BATCH_SIZE=50000
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./logs:/app/logs

  state-mongodb:
    image: mongo:8.0
    container_name: state-mongodb
    restart: always
    ports:
      - "27018:27017"
    volumes:
      - state_mongodb_data:/data/db

volumes:
  state_mongodb_data:
```

## How to Run
1. Build and start the containers:
   ```sh
   docker-compose up --build state-anomaly-app state-mongodb
   ```
2. Logs and output will be available in the `logs/` and `output/` folders.

## Notes
- The state-level model uses its own MongoDB container and collection (`state_job_history`).
- Make sure the input data and mapping files are available in the mounted `data/` directory.
- Adjust batch size and file paths as needed via environment variables. 