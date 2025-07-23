# US Job Anomaly Detection Pipeline

This project implements a scalable, memory-efficient pipeline for detecting anomalies in US-level and State-level job posting data using daily job counts per segment stored in MongoDB. The pipeline is designed for large-scale data (millions of records) and is fully containerized with Docker.

## Features
- **Batch processing** of large parquet files for memory efficiency
- **Daily aggregation**: Only daily job counts per segment are stored in MongoDB
- **Rolling window**: Keeps only the last 45 days of history per segment
- **Fast upserts**: Uses compound indexes for efficient MongoDB operations
- **Anomaly detection**: Rolling statistics and Z-score based detection
- **Full Docker support**: Reproducible, isolated, and production-ready
- **Dual models**: US-level and State-level anomaly detection

## Project Structure
```
V0_model_USLevel/        # US Level Model
  main.py                # Entrypoint for US-level pipeline
  pipeline.py            # US-level pipeline logic
  mongodb_hist.py        # MongoDB history manager
  preprocess.py          # Data loading and preprocessing
  state_impute.py        # State imputation logic
  detect.py              # US-level anomaly detection logic
  utils/
    memlogger.py         # Memory logging utility

V0_model_StateLevel/     # State Level Model
  main.py                # Entrypoint for state-level pipeline
  pipeline.py            # State-level pipeline logic
  history.py             # State-level MongoDB history manager
  preprocess.py          # Data loading and preprocessing
  state_impute.py        # State imputation logic
  detect.py              # State-level anomaly detection logic
  utils/
    memlogger.py         # Memory logging utility

Dockerfile               # Build the model images
docker-compose.yml       # Orchestration for both models
requirements.txt         # Python dependencies
output/                  # Output CSVs and logs
logs/                    # Log files
```

## Setup Instructions

### 1. Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop) (recommended)
- Docker Compose (comes with Docker Desktop)
- (Optional) Python 3.9+ for local runs

### 2. Build and Run with Docker

#### Build the images
```
docker-compose build
```

#### Start MongoDB in the background
```
docker-compose up -d anomaly-detection-mongodb
```

#### Run both models
```
docker-compose up
```

#### Run specific model
```
# US Level Model
docker-compose up anomaly-detection-us-level

# State Level Model
docker-compose up anomaly-detection-state-level
```

### 3. Data Files
- Place your input parquet file(s) in the `data/` directory.
- Update the `file_path` in the respective model's `main.py` to point to your data file.
- Ensure mapping files (`Full_zipcodeFix.xlsx`, `Canaria_Zipcodes.xlsx`) are present in `data/`.

### 4. Output
- Results and logs are written to the `output/` and `logs/` directories.
- US-level anomalies are saved as CSV in `output/anomalies_US.csv` (default).
- State-level anomalies are saved as CSV in `output/anomalies_state.csv` (default).

## Configuration
- **MongoDB URI**: Set via the `MONGO_URI` environment variable (default is `mongodb://anomaly-detection-mongodb:27017` in Docker).
- **Batch size**: Configurable in each model's `main.py` (`batch_size` variable).
- **Rolling window**: Last 45 days per segment are kept for anomaly detection.
- **Environment variables**: See `docker-compose.yml` for state-level model configuration.

## Production Notes
- The pipeline is robust to large data and MongoDB restarts.
- Only daily counts are stored, not raw job records, for scalability.
- All logs are written to both console and `logs/` for traceability.
- For production, consider enabling MongoDB authentication and network restrictions.

## Troubleshooting
- If you see MongoDB version errors, remove the Docker volume:
  ```
  docker-compose down -v
  docker volume rm anomalydetection_mongodb_data
  ```
- If you change the data file, clear MongoDB before rerunning for clean results.

## License
MIT