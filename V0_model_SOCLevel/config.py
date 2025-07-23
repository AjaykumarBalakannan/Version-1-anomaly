import os
from datetime import datetime

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'anomaly_detection')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'soc_level_data')

# Data Processing Configuration
DATA_DIR = os.getenv('DATA_DIR', 'data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
LOG_DIR = os.getenv('LOG_DIR', 'logs')

# SOC Aggregation Configuration
MIN_VOLUME_THRESHOLD = int(os.getenv('MIN_VOLUME_THRESHOLD', '10'))
VARIANCE_THRESHOLD = float(os.getenv('VARIANCE_THRESHOLD', '0.5'))
MAX_SOC_LENGTH = int(os.getenv('MAX_SOC_LENGTH', '6'))

# Anomaly Detection Configuration
Z_SCORE_THRESHOLD = float(os.getenv('Z_SCORE_THRESHOLD', '2.5'))
DYNAMIC_THRESHOLD_MULTIPLIER = float(os.getenv('DYNAMIC_THRESHOLD_MULTIPLIER', '1.5'))
LOOKBACK_DAYS = int(os.getenv('LOOKBACK_DAYS', '30'))

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Model Configuration
MODEL_NAME = 'V0_SOC_Level'
MODEL_VERSION = '1.0.0'
CREATED_DATE = datetime.now().strftime('%Y-%m-%d')

# File naming conventions
ANOMALY_OUTPUT_FILE = f'anomalies_soc_{datetime.now().strftime("%Y%m%d")}.csv'
LOG_FILE = f'{MODEL_NAME}_{datetime.now().strftime("%Y%m%d")}.log' 