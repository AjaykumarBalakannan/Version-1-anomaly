# CANARIA Anomaly Detection System

A comprehensive job market anomaly detection system that identifies unusual patterns in job posting data across three levels: US Level, State Level, and SOC (Standard Occupational Classification) Level.

## 📋 **Table of Contents**

- [🏗️ System Architecture](#️-system-architecture)
- [🔄 How It Works](#-how-it-works)
- [📊 Data Flow & Rolling Window](#-data-flow--rolling-window)
- [🧠 V1 Model Architecture](#-v1-model-architecture)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [🔧 Detailed Setup Instructions](#-detailed-setup-instructions)
- [🧠 Model Details](#-model-details)
- [📊 Output Format](#-output-format)
- [🔍 Common Files & Functions](#-common-files--functions)
- [🚨 Troubleshooting](#-troubleshooting)
- [📈 Monitoring & Logging](#-monitoring--logging)
- [🔄 Upgrade Paths](#-upgrade-paths)
- [🔧 Advanced Customization Guide](#-advanced-customization-guide)
- [📞 Support & Maintenance](#-support--maintenance)
- [🎯 Quick Reference Commands](#-quick-reference-commands)

## 🏗️ **System Architecture**

This system consists of three independent anomaly detection models, each analyzing job data at different granularities:

- **US Level Model**: Detects anomalies across geographic regions (geo_level × source)
- **State Level Model**: Detects anomalies at state/territory level (state × source)  
- **SOC Level Model**: Detects anomalies by job categories (SOC code × source)

All models use **Version 1 (V1) ML-based ensemble detection** with advanced feature engineering and multiple anomaly detection algorithms.

## 🔄 **How It Works**

### **End-to-End Process Flow**

The CANARIA system follows a sophisticated pipeline to detect anomalies in job posting data:

```
Input Data (Parquet) → Preprocessing → MongoDB Storage → Feature Engineering → ML Models → Anomaly Detection → Results
```

#### **1. Data Ingestion & Preprocessing**
- **Input**: Large parquet files containing daily job posting data
- **Batch Processing**: Data is processed in configurable batches (default: 50,000 rows) to manage memory efficiently
- **Data Cleaning**: Timestamps are rebuilt, null dates are filled, and geographic data is imputed
- **Validation**: Ensures required columns (`date`, `geo_level`/`state`/`nlp_soc_code`, `src`, `job_count`) are present

#### **Detailed Preprocessing Steps:**

**Step 1: File Reading & Validation**
```python
# System reads parquet files in configurable batches
def read_parquet_in_batches(file_path, batch_size=50000):
    # CUSTOMIZE BATCH SIZE HERE for memory management
    # Default: 50,000 rows per batch
    # High-memory systems: 100,000+ rows per batch
    # Low-memory systems: 25,000 rows per batch
```

**Step 2: Data Cleaning & Validation**
```python
# Required columns validation
required_columns = ['date', 'geo_level', 'src', 'job_count']  # US Level
required_columns = ['date', 'state', 'src', 'job_count']      # State Level  
required_columns = ['date', 'nlp_soc_code', 'src', 'job_count'] # SOC Level

# Data type validation and conversion
df['date'] = pd.to_datetime(df['date'])           # Convert to datetime
df['job_count'] = pd.to_numeric(df['job_count'])  # Convert to numeric
df['src'] = df['src'].astype(str)                 # Convert to string
```

**Step 3: Geographic Data Imputation**
```python
# State Level: Impute missing state information
def impute_states(df):
    # Uses zipcode mapping files to fill missing states
    # Maps: Canaria_Zipcodes.xlsx, Full_zipcodeFix.xlsx
    # CUSTOMIZE: Add your own mapping files or logic

# SOC Level: Roll up SOC codes for better grouping
def rollup_soc_code(soc_code):
    # Groups similar SOC codes together
    # CUSTOMIZE: Modify SOC grouping logic as needed
```

**Step 4: Data Quality Checks**
```python
# Automatic data quality validation
def validate_data_quality(df):
    # Check for missing values
    # Validate date ranges
    # Ensure job counts are positive
    # Verify geographic/SOC code validity
    # CUSTOMIZE: Add your own validation rules
```

#### **2. MongoDB Storage with Rolling Window**
- **Daily Aggregation**: Only daily job counts per segment are stored (not raw job records)
- **Rolling Window**: Maintains a configurable window of historical data (default: 45 days)
- **Automatic Cleanup**: Oldest data is automatically removed when new data arrives
- **Efficient Indexing**: Compound indexes on `(group_field, src, date)` for fast queries

#### **3. Feature Engineering Pipeline**
- **Rolling Statistics**: 7, 14, 28-day rolling means, standard deviations, MAD
- **Temporal Features**: Day of week, week of year, month patterns
- **Statistical Measures**: Z-scores, percentage changes, rolling min/max
- **Long-term Patterns**: Expanding means, share vs long-term averages

#### **4. ML-Based Anomaly Detection**
- **Ensemble Approach**: Combines 7 different anomaly detection algorithms
- **Per-Segment Thresholds**: Each (geo_level/state/SOC × source) gets its own adaptive threshold
- **Smart Calibration**: Uses quantile-based thresholds (default: 95th-99.9th percentile)
- **Cooldown Logic**: Prevents alert spam with configurable cooldown periods

#### **5. Results & Output**
- **Anomaly Scores**: Individual scores from each model + ensemble average
- **Binary Classification**: Clear anomaly flags (1 = anomaly, 0 = normal)
- **Detailed Logging**: Complete execution logs with performance metrics
- **Multiple Formats**: CSV output with comprehensive anomaly information

## 📊 **Data Flow & Rolling Window**

### **Rolling Window Mechanism**

The system implements a sophisticated rolling window approach that ensures optimal performance and memory management:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rolling Window (45 days)                     │
├─────────────────────────────────────────────────────────────────┤
│ Day 1 │ Day 2 │ Day 3 │ ... │ Day 44 │ Day 45 │ NEW DAY      │
│        │       │       │     │        │        │ ↓             │
│        │       │       │     │        │        │ Day 46       │
│        │       │       │     │        │        │ (Day 1      │
│        │       │       │     │        │        │  deleted)   │
└─────────────────────────────────────────────────────────────────┘
```

#### **How Rolling Window Works:**

1. **Initial Load**: System loads the last 45 days of historical data
2. **Daily Updates**: When new data arrives:
   - New day is added to the window
   - Oldest day (beyond 45 days) is automatically removed
   - Window maintains exactly 45 days of data
3. **Memory Efficiency**: Only relevant historical data is kept in memory
4. **Performance**: Faster queries and processing with limited data

#### **Rolling Window Benefits:**

- **🔄 Always Fresh**: Data is never older than the configured window
- **💾 Memory Efficient**: Prevents unlimited data accumulation
- **⚡ Fast Processing**: Smaller datasets process faster
- **🎯 Relevant History**: Focuses on recent patterns that matter
- **🔧 Configurable**: Window size can be adjusted per model

#### **Rolling Window Configuration:**

```python
# In mongodb_hist.py - customize rolling window size
class HistoryManager:
    def __init__(self, mongo_uri, db_name, collection_name, group_field='geo_level'):
        # CUSTOMIZE ROLLING WINDOW HERE
        self.rolling_window_days = 45  # Change to desired retention days
        
        # Examples:
        # self.rolling_window_days = 30   # 30 days for short-term analysis
        # self.rolling_window_days = 90   # 90 days for long-term patterns
        # self.rolling_window_days = 180  # 180 days for seasonal analysis
```

### **Data Storage Strategy**

#### **MongoDB Collections Structure:**

```
us_anomaly_db/
├── us_job_history/
│   ├── {geo_level: "US", src: "linkedin_us", date: "2025-01-01", job_count: 15000}
│   ├── {geo_level: "US", src: "linkedin_us", date: "2025-01-02", job_count: 15200}
│   └── ... (45 days of data)

state_anomaly_db/
├── state_job_history/
│   ├── {state: "CA", src: "linkedin_us", date: "2025-01-01", job_count: 5000}
│   ├── {state: "CA", src: "linkedin_us", date: "2025-01-02", job_count: 5100}
│   └── ... (45 days of data)

soc_anomaly_db/
├── soc_level_data/
│   ├── {nlp_soc_code_rolled: "15-1132", src: "linkedin_us", date: "2025-01-01", job_count: 800}
│   ├── {nlp_soc_code_rolled: "15-1132", src: "linkedin_us", date: "2025-01-02", job_count: 820}
│   └── ... (45 days of data)
```

#### **Automatic Data Management:**

- **Upsert Logic**: New data automatically updates existing records or creates new ones
- **Date-based Cleanup**: Old records beyond rolling window are automatically removed
- **Index Optimization**: Compound indexes ensure fast queries and updates
- **Data Integrity**: Automatic validation and error handling

## 🧠 **V1 Model Architecture**

### **Core Architecture Components**

The V1 models use a sophisticated multi-layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    V1 Anomaly Detection Architecture            │
├─────────────────────────────────────────────────────────────────┤
│  Input Data Layer                                              │
│  ├── Raw Job Counts (Parquet)                                 │
│  ├── Preprocessing & Validation                               │
│  └── MongoDB Storage (Rolling Window)                         │
├─────────────────────────────────────────────────────────────────┤
│  Feature Engineering Layer                                     │
│  ├── Rolling Statistics (7, 14, 28 days)                     │
│  ├── Temporal Features (DOW, Week, Month)                     │
│  ├── Statistical Measures (Z-scores, MAD)                     │
│  └── Long-term Patterns (Expanding means)                     │
├─────────────────────────────────────────────────────────────────┤
│  ML Model Ensemble Layer                                       │
│  ├── PyCaret Models (iforest, knn, svm)                       │
│  ├── PyOD Models (hbos, copod, ocsvm, cblof)                 │
│  └── Ensemble Scoring & Normalization                          │
├─────────────────────────────────────────────────────────────────┤
│  Threshold & Alert Layer                                       │
│  ├── Per-Segment Thresholds (Quantile-based)                  │
│  ├── Cooldown Logic (Prevent Alert Spam)                      │
│  └── Anomaly Classification                                    │
├─────────────────────────────────────────────────────────────────┤
│  Output & Monitoring Layer                                     │
│  ├── CSV Results with Scores                                   │
│  ├── Detailed Logging & Metrics                                │
│  └── Performance Monitoring                                    │
└─────────────────────────────────────────────────────────────────┘
```

### **Model Training & Inference Process**

#### **Training Phase:**
1. **Data Preparation**: Load historical data from MongoDB rolling window
2. **Feature Engineering**: Generate 18 engineered features per record
3. **Model Training**: Train 7 different anomaly detection models
4. **Threshold Calibration**: Calculate per-segment thresholds using quantiles

#### **Inference Phase:**
1. **Feature Generation**: Generate features for new data
2. **Model Scoring**: Get anomaly scores from all 7 models
3. **Ensemble Aggregation**: Average scores for robust detection
4. **Threshold Comparison**: Compare scores against calibrated thresholds
5. **Anomaly Classification**: Mark records as normal or anomalous

### **Key Architectural Benefits:**

- **🔄 Rolling Updates**: Models automatically adapt to new data patterns
- **⚖️ Ensemble Robustness**: Multiple models reduce false positives/negatives
- **🎯 Adaptive Thresholds**: Each segment gets its own sensitivity level
- **💾 Memory Efficient**: Rolling window prevents unlimited data accumulation
- **🔧 Highly Configurable**: Every aspect can be customized for specific needs

### **Daily Workflow & Rolling Window Operation**

#### **What Happens Every Day:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Daily Execution Workflow                     │
├─────────────────────────────────────────────────────────────────┤
│  Morning (Data Ingestion)                                      │
│  ├── New parquet file arrives with yesterday's job data        │
│  ├── System processes data in configurable batches             │
│  ├── Data is cleaned, validated, and preprocessed             │
│  └── Daily job counts are aggregated per segment              │
├─────────────────────────────────────────────────────────────────┤
│  MongoDB Update (Rolling Window)                               │
│  ├── New day's data is inserted/updated in MongoDB            │
│  ├── System checks rolling window size (default: 45 days)     │
│  ├── Oldest day beyond 45 days is automatically deleted       │
│  └── Rolling window maintains exactly 45 days of data         │
├─────────────────────────────────────────────────────────────────┤
│  Anomaly Detection (ML Models)                                 │
│  ├── System loads 45 days of historical data from MongoDB     │
│  ├── Feature engineering generates 18 features per record     │
│  ├── 7 ML models are trained on the 45-day dataset            │
│  ├── Per-segment thresholds are calibrated                    │
│  └── Anomalies are detected and classified                    │
├─────────────────────────────────────────────────────────────────┤
│  Results & Cleanup                                             │
│  ├── Anomaly results are saved to CSV files                   │
│  ├── Detailed logs are written with performance metrics       │
│  ├── Memory is cleaned up (rolling window prevents overflow)  │
│  └── System is ready for next day's execution                 │
└─────────────────────────────────────────────────────────────────┘
```

#### **Rolling Window in Action:**

**Day 1-45 (Initial Window):**
```
MongoDB contains: [Day 1, Day 2, Day 3, ..., Day 44, Day 45]
Total records: 45 days × number of segments
```

**Day 46 (New Data Arrives):**
```
1. NEW DATA: Day 46 job counts are processed and added
2. ROLLING WINDOW: System checks window size (45 days)
3. CLEANUP: Day 1 data is automatically deleted
4. RESULT: Window now contains [Day 2, Day 3, ..., Day 45, Day 46]
```

**Day 47 (Next Day):**
```
1. NEW DATA: Day 47 job counts are processed and added
2. ROLLING WINDOW: System checks window size (45 days)
3. CLEANUP: Day 2 data is automatically deleted
4. RESULT: Window now contains [Day 3, Day 4, ..., Day 46, Day 47]
```

#### **Why Rolling Window is Critical:**

1. **🔄 Always Current**: Models always work with the most recent 45 days of data
2. **💾 Memory Management**: Prevents unlimited data accumulation that would crash the system
3. **⚡ Performance**: Smaller datasets process faster and use less memory
4. **🎯 Pattern Recognition**: Focuses on recent market patterns that are most relevant
5. **🔧 Scalability**: System can handle years of data without performance degradation

#### **Rolling Window Configuration Examples:**

```python
# Short-term analysis (30 days)
self.rolling_window_days = 30  # Faster processing, recent patterns only

# Medium-term analysis (45 days) - DEFAULT
self.rolling_window_days = 45  # Balanced performance and pattern recognition

# Long-term analysis (90 days)
self.rolling_window_days = 90  # Slower processing, seasonal patterns included

# Seasonal analysis (180 days)
self.rolling_window_days = 180 # Long-term trends, seasonal effects captured
```

## 📁 **Project Structure**

```
CANARIA_PROJECT_DEPLOYMENT/
├── lib/                                    # Main project directory
│   ├── common/                            # Shared functionality
│   │   ├── mongo_utils.py                 # MongoDB connection utilities
│   │   ├── mongodb_hist.py               # Historical data management
│   │   ├── preprocess.py                 # Data preprocessing utilities
│   │   └── utils/
│   │       └── memlogger.py              # Memory usage logging
│   ├── V0_model_USLevel/                 # US Level anomaly detection
│   │   ├── config.py                     # Configuration settings
│   │   ├── detect.py                     # V1 ML-based anomaly detection
│   │   ├── main.py                       # Entry point for US Level
│   │   └── pipeline.py                   # Data processing pipeline
│   ├── V0_model_StateLevel/              # State Level anomaly detection
│   │   ├── config.py                     # Configuration settings
│   │   ├── detect.py                     # V1 ML-based anomaly detection
│   │   ├── main.py                       # Entry point for State Level
│   │   ├── pipeline.py                   # Data processing pipeline
│   │   └── state_impute.py               # State data imputation
│   ├── V0_model_SOCLevel/                # SOC Level anomaly detection
│   │   ├── config.py                     # Configuration settings
│   │   ├── detect.py                     # V1 ML-based anomaly detection
│   │   ├── main.py                       # Entry point for SOC Level
│   │   ├── pipeline.py                   # Data processing pipeline
│   │   └── soc_groupings.py              # SOC code rollup logic
│   ├── data/                             # Data files
│   │   ├── Canaria_Zipcodes.xlsx         # Zipcode mapping data
│   │   └── Full_zipcodeFix.xlsx          # Zipcode correction data
│   ├── input-files/                      # Input data directory
│   ├── output/                           # Output results directory
│   ├── logs/                             # Log files directory
│   ├── docker-compose.yml                # Docker orchestration
│   ├── Dockerfile                        # Application container
│   └── requirements.txt                  # Python dependencies
```

## 🚀 **Quick Start Guide**

### **Prerequisites**
- Docker and Docker Compose installed
- Git (to clone the repository)
- At least 8GB RAM available

### **1. Clone and Setup**
```bash
# Clone the repository (replace with actual repo URL)
git clone <REPOSITORY_URL>
cd CANARIA_PROJECT_DEPLOYMENT/lib

# Build and start the containers
docker-compose up -d --build

# Wait for MongoDB to fully start (about 10-15 seconds)
sleep 15
```

### **2. Test Individual Models**

#### **US Level Model**
```bash
docker exec anomaly-detection python V0_model_USLevel/main.py
```
**Output**: `output/anomalies_US_V1.csv`

#### **State Level Model**
```bash
docker exec anomaly-detection python V0_model_StateLevel/main.py
```
**Output**: `output/anomalies_state_V1.csv`

#### **SOC Level Model**
```bash
docker exec anomaly-detection python V0_model_SOCLevel/main.py
```
**Output**: `output/anomalies_SOC_V1.csv`

### **3. Run All Models Sequentially**
```bash
# Run US Level
docker exec anomaly-detection python V0_model_USLevel/main.py

# Run State Level  
docker exec anomaly-detection python V0_model_StateLevel/main.py

# Run SOC Level
docker exec anomaly-detection python V0_model_SOCLevel/main.py
```

## 🔧 **Detailed Setup Instructions**

### **Docker Environment Setup**

The system uses Docker containers for:
- **anomaly-detection**: Main application container with Python environment
- **anomaly-detection-mongodb**: MongoDB database for historical data

#### **Container Management**
```bash
# Start containers
docker-compose up -d

# Stop containers
docker-compose down

# Stop and remove all data (fresh start)
docker-compose down -v

# View logs
docker-compose logs -f

# Rebuild containers
docker-compose up -d --build
```

### **Environment Variables & Configuration**

#### **Docker Environment Variables**
Edit `docker-compose.yml` to customize:

```yaml
environment:
  - MONGO_URI=mongodb://anomaly-detection-mongodb:27017
  - BATCH_SIZE=50000                    # Memory management
  - ROLLING_WINDOW_DAYS=45              # Historical data retention
  - LOG_LEVEL=INFO                       # Logging verbosity
  - MAX_RETRIES=3                        # MongoDB connection retries
  - RETRY_DELAY=5                        # Retry delay in seconds
```

#### **Model-Specific Environment Variables**
Each model can have custom configurations:

```yaml
# US Level Model
- US_MODEL_BATCH_SIZE=50000
- US_MODEL_ROLLING_WINDOW=45

# State Level Model  
- STATE_MODEL_BATCH_SIZE=50000
- STATE_MODEL_ROLLING_WINDOW=45

# SOC Level Model
- SOC_MODEL_BATCH_SIZE=50000
- SOC_MODEL_ROLLING_WINDOW=45
```

#### **MongoDB Access**
```bash
# Access MongoDB shell
docker exec -it anomaly-detection-mongodb mongosh

# Clear all data (if needed)
docker exec anomaly-detection-mongodb mongosh --eval "
  db.getSiblingDB('us_anomaly_db').us_job_history.deleteMany({}); 
  db.getSiblingDB('state_anomaly_db').state_job_history.deleteMany({}); 
  db.getSiblingDB('soc_anomaly_db').soc_level_data.deleteMany({}); 
  print('All collections cleared successfully');
"
```

### **Data Management**

#### **Input Data Requirements**
- Place your job posting data files in the `input-files/` directory
- Supported formats: Parquet files
- Required columns: `date`, `geo_level`/`state`/`nlp_soc_code`, `src`, `job_count`

#### **Customizing Input/Output Paths**
Edit the `main.py` file in each model directory:

```python
# US Level Model - V0_model_USLevel/main.py
class USLevelAnomalyPipeline:
    def __init__(self, file_path, out_csv, batch_size=50000):
        self.file_path = file_path          # CUSTOMIZE INPUT PATH HERE
        self.out_csv = out_csv              # CUSTOMIZE OUTPUT PATH HERE
        self.batch_size = batch_size        # CUSTOMIZE BATCH SIZE HERE

# State Level Model - V0_model_StateLevel/main.py  
class StateLevelAnomalyPipeline:
    def __init__(self, file_path, out_csv, batch_size=50000):
        self.file_path = file_path          # CUSTOMIZE INPUT PATH HERE
        self.out_csv = out_csv              # CUSTOMIZE OUTPUT PATH HERE
        self.batch_size = batch_size        # CUSTOMIZE BATCH SIZE HERE

# SOC Level Model - V0_model_SOCLevel/main.py
class SOCLevelAnomalyPipeline:
    def __init__(self, file_path, out_csv, batch_size=50000):
        self.file_path = file_path          # CUSTOMIZE INPUT PATH HERE
        self.out_csv = out_csv              # CUSTOMIZE OUTPUT PATH HERE
        self.batch_size = batch_size        # CUSTOMIZE BATCH SIZE HERE
```

**Path Customization Examples:**
```python
# Custom input paths
file_path = "data/custom_job_data.parquet"
file_path = "/absolute/path/to/your/data.parquet"
file_path = "s3://bucket/path/to/data.parquet"  # If using cloud storage

# Custom output paths
out_csv = "results/anomalies_custom_name.csv"
out_csv = "/custom/output/directory/anomalies.csv"
out_csv = "output/anomalies_$(date +%Y%m%d).csv"  # Dynamic naming

# Custom batch sizes for memory management
batch_size = 100000  # For high-memory systems
batch_size = 25000   # For low-memory systems
batch_size = 50000   # Default balanced setting
```

#### **Output Files**
- **Anomaly Results**: CSV files with anomaly scores and predictions
- **Logs**: Detailed execution logs in `logs/` directory
- **Today's Anomalies**: Filtered results for the most recent date

## 🧠 **Model Details**

### **V1 Anomaly Detection Features**

All three models use the same advanced V1 detection system:

#### **Ensemble Models (7 algorithms)**
1. **Isolation Forest** - Tree-based outlier detection
2. **HBOS** - Histogram-based outlier detection
3. **COPOD** - Copula-based outlier detection
4. **KNN** - K-Nearest Neighbors outlier detection
5. **SVM** - Support Vector Machine outlier detection
6. **OCSVM** - One-Class SVM outlier detection
7. **CBLOF** - Cluster-based local outlier factor

#### **Customizing ML Models**
Edit the `models` parameter in each model's `detect.py`:

```python
# In detect.py - AnomalyDetectorV1.__init__()
def __init__(
    self,
    models=("iforest", "hbos", "copod", "knn", "svm", "ocsvm", "cblof"),  # CUSTOMIZE HERE
    alert_budget_per_week=2,                                                # CUSTOMIZE HERE
    cooldown_days=1,                                                        # CUSTOMIZE HERE
    quantile_bounds=(0.95, 0.999),                                         # CUSTOMIZE HERE
    verbose=False,
):
```

**Available Model Options:**
- **PyCaret Models**: `"iforest"`, `"knn"`, `"svm"`
- **PyOD Models**: `"hbos"`, `"copod"`, `"ocsvm"`, `"cblof"`
- **Custom Models**: Add your own models by extending the class

**Model Parameters to Customize:**
- **`alert_budget_per_week`**: Maximum alerts per week (default: 2)
- **`cooldown_days`**: Days between alerts (default: 1)
- **`quantile_bounds`**: Sensitivity range (default: 95th-99.9th percentile)

#### **Feature Engineering (18 features)**
- **Rolling Statistics**: 7, 14, 28-day rolling means, standard deviations, MAD
- **Percentage Changes**: 1-day and 7-day changes
- **Z-Scores**: 7-day and 28-day z-scores
- **Temporal Features**: Day of week, week of year, month
- **Long-term Patterns**: Expanding means, share vs long-term average

#### **Customizing Feature Engineering**
Edit the `_add_roll_feats` method in each model's `detect.py`:

```python
@staticmethod
def _add_roll_feats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    
    # CUSTOMIZE ROLLING WINDOWS HERE
    g["mean_7"] = g["job_count"].rolling(7, min_periods=3).mean()      # Change 7 to desired days
    g["mean_14"] = g["job_count"].rolling(14, min_periods=5).mean()    # Change 14 to desired days
    g["mean_28"] = g["job_count"].rolling(28, min_periods=7).mean()    # Change 28 to desired days
    
    # CUSTOMIZE STANDARD DEVIATION WINDOWS
    g["std_7"] = g["job_count"].rolling(7, min_periods=3).std(ddof=0)  # Change 7 to desired days
    g["std_28"] = g["job_count"].rolling(28, min_periods=7).std(ddof=0) # Change 28 to desired days
    
    # CUSTOMIZE MAD WINDOW
    g["mad_28"] = g["job_count"].rolling(28, min_periods=7).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=False
    )  # Change 28 to desired days
    
    # CUSTOMIZE PERCENTAGE CHANGE WINDOWS
    g["pct_chg_1"] = g["job_count"].pct_change(1)  # Change 1 to desired days
    g["pct_chg_7"] = g["job_count"].pct_change(7)  # Change 7 to desired days
    
    # CUSTOMIZE Z-SCORE WINDOWS
    g["z_7"] = (g["job_count"] - g["mean_7"]) / (g["std_7"].replace(0, np.nan))
    g["z_28"] = (g["job_count"] - g["mean_28"]) / (g["std_28"].replace(0, np.nan))
    
    # CUSTOMIZE ROLLING MIN/MAX WINDOWS
    g["roll_min_14"] = g["job_count"].rolling(14, min_periods=5).min()  # Change 14 to desired days
    g["roll_max_14"] = g["job_count"].rolling(14, min_periods=5).max()  # Change 14 to desired days
    
    # CUSTOMIZE LONG-TERM PATTERNS
    g["lt_mean"] = g["job_count"].expanding(min_periods=14).mean()      # Change 14 to desired days
    g["share_vs_lt"] = g["job_count"] / (g["lt_mean"].replace(0, np.nan))
    
    return g
```

**Feature Customization Options:**
- **Rolling Windows**: Change 7, 14, 28 to any number of days
- **Min Periods**: Adjust minimum data points required for calculations
- **New Features**: Add custom features like:
  - Seasonal patterns: `g["season"] = g["date"].dt.quarter`
  - Holiday effects: `g["is_holiday"] = g["date"].isin(holiday_dates)`
  - Market indicators: `g["market_volatility"] = external_volatility_data`

#### **Smart Thresholding**
- **Per-segment thresholds**: Each (geo_level/state/SOC × source) gets its own threshold
- **Quantile-based**: Configurable sensitivity (default: 95th percentile)
- **Cooldown periods**: Prevents alert spam (default: 1 day)

### **Model-Specific Details**

#### **US Level Model** (`V0_model_USLevel/`)
- **Grouping**: `geo_level × src`
- **Use Case**: National/regional anomaly detection
- **Data Source**: Geographic region-based job counts
- **Output**: `anomalies_US_V1.csv`

#### **State Level Model** (`V0_model_StateLevel/`)
- **Grouping**: `state × src`
- **Use Case**: State-specific anomaly detection
- **Data Source**: State-level job counts
- **Output**: `anomalies_state_V1.csv`

#### **SOC Level Model** (`V0_model_SOCLevel/`)
- **Grouping**: `nlp_soc_code_rolled × src`
- **Use Case**: Job category anomaly detection
- **Data Source**: SOC code-based job counts
- **Output**: `anomalies_SOC_V1.csv`

## 📊 **Output Format**

All models produce CSV files with the following columns:

```csv
geo_level/state/nlp_soc_code_rolled,src,date,job_count,
score_iforest,score_hbos,score_copod,score_knn,score_svm,
score_ocsvm,score_cblof,score_ens,date_only,threshold,is_anomaly
```

### **Key Output Fields**
- **Individual Scores**: Anomaly scores from each model (0-1 scale)
- **Ensemble Score**: Average of all individual scores
- **Threshold**: Per-segment anomaly threshold
- **is_anomaly**: Binary flag (1 = anomaly, 0 = normal)

## 🔍 **Common Files & Functions**

### **`common/mongo_utils.py`**
- **Purpose**: MongoDB connection management
- **Key Functions**:
  - `get_mongo_collection()`: Get MongoDB collection with retry logic
  - Connection pooling and error handling

### **`common/mongodb_hist.py`**
- **Purpose**: Historical data management with rolling windows
- **Key Functions**:
  - `upsert_data()`: Add/update daily job counts
  - `load_historical_data()`: Load data for analysis
  - Rolling window filtering for optimal performance

#### **Customizing MongoDB Configuration**
Edit the `mongodb_hist.py` file to customize database settings:

```python
class HistoryManager:
    def __init__(self, mongo_uri, db_name, collection_name, group_field='geo_level'):
        self.mongo_uri = mongo_uri                    # CUSTOMIZE MONGODB URI HERE
        self.db_name = db_name                        # CUSTOMIZE DATABASE NAME HERE
        self.collection_name = collection_name        # CUSTOMIZE COLLECTION NAME HERE
        self.group_field = group_field                # CUSTOMIZE GROUPING FIELD HERE
        
        # CUSTOMIZE ROLLING WINDOW HERE
        self.rolling_window_days = 45                 # Change 45 to desired retention days
        
        # CUSTOMIZE CONNECTION PARAMETERS HERE
        self.max_retries = 3                          # Change 3 to desired retry attempts
        self.retry_delay = 5                          # Change 5 to desired retry delay seconds
        self.server_selection_timeout = 10000         # Change 10000 to desired timeout (ms)
```

**MongoDB Customization Options:**
```python
# Custom MongoDB URIs
mongo_uri = "mongodb://localhost:27017"              # Local MongoDB
mongo_uri = "mongodb://user:pass@host:27017"        # Authenticated MongoDB
mongo_uri = "mongodb://host1:27017,host2:27017"     # Replica set
mongo_uri = "mongodb+srv://cluster.mongodb.net"     # MongoDB Atlas

# Custom Database Names
db_name = "custom_anomaly_db"                        # Custom database name
db_name = "anomaly_detection_prod"                   # Production database
db_name = "anomaly_detection_dev"                    # Development database

# Custom Collection Names
collection_name = "custom_job_history"               # Custom collection name
collection_name = "job_data_2025"                    # Year-specific collection
collection_name = "anomaly_input_data"               # Descriptive collection name

# Custom Grouping Fields
group_field = "custom_geo_field"                     # Custom geographic field
group_field = "business_unit"                        # Business unit grouping
group_field = "market_segment"                       # Market segment grouping
```

### **`common/preprocess.py`**
- **Purpose**: Data preprocessing utilities
- **Key Functions**:
  - `read_parquet_in_batches()`: Memory-efficient file reading
  - `rebuild_timestamps()`: Fix timestamp formatting
  - `fill_null_dates()`: Handle missing dates

### **`common/utils/memlogger.py`**
- **Purpose**: Memory usage monitoring
- **Key Functions**:
  - `log_memory()`: Log current memory usage
  - Performance monitoring for large datasets

## 🚨 **Troubleshooting**

### **Common Issues**

#### **MongoDB Connection Errors**
```bash
# Check if MongoDB is running
docker ps | grep mongodb

# Restart MongoDB container
docker restart anomaly-detection-mongodb

# Wait for startup and retry
sleep 15
```

#### **Memory Issues**
- Reduce batch size in pipeline configuration
- Monitor memory usage in logs
- Ensure sufficient RAM (8GB+ recommended)

#### **Model Training Failures**
- Check if all dependencies are installed
- Verify data quality and sufficient history
- Check logs for specific error messages

### **Performance Optimization**

#### **For Large Datasets**
- Increase batch size gradually
- Monitor memory usage
- Use rolling window filtering in MongoDB

#### **For Faster Processing**
- Reduce feature set if needed
- Use fewer ensemble models
- Adjust quantile bounds for sensitivity

## 📈 **Monitoring & Logging**

### **Log Files**
- **Location**: `logs/` directory
- **Naming**: `V1_[Model]Level_anomaly_pipeline_YYYY-MM-DD.log`
- **Content**: Detailed execution logs, memory usage, model training status

#### **Customizing Logging Configuration**
Edit the `main.py` file in each model directory:

```python
# Customize logging level and format
logging.basicConfig(
    level=logging.INFO,                              # CUSTOMIZE LOG LEVEL HERE
    format="%(asctime)s [%(levelname)s] %(message)s", # CUSTOMIZE FORMAT HERE
    handlers=[
        logging.FileHandler(log_file_path),           # CUSTOMIZE LOG FILE PATH HERE
        logging.StreamHandler()                       # CUSTOMIZE CONSOLE OUTPUT HERE
    ]
)

# Customize log file naming
today_str = datetime.now().strftime("%Y-%m-%d")
log_file_path = f"logs/CUSTOM_PREFIX_{today_str}.log"  # CUSTOMIZE PREFIX HERE
log_file_path = f"logs/anomaly_detection_{today_str}_{model_type}.log"  # MODEL-SPECIFIC NAMING
log_file_path = f"/custom/log/path/anomalies_{today_str}.log"  # CUSTOM LOG PATH
```

**Log Level Options:**
- `logging.DEBUG`: Most verbose, includes all details
- `logging.INFO`: Standard information level (default)
- `logging.WARNING`: Only warnings and errors
- `logging.ERROR`: Only errors
- `logging.CRITICAL`: Only critical errors

**Custom Log Format Examples:**
```python
# Simple format
format="%(levelname)s: %(message)s"

# Detailed format with function names
format="%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s"

# JSON format for log aggregation
format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'

# Custom format with process ID
format="%(asctime)s [%(levelname)s] PID:%(process)d - %(message)s"
```

### **Key Log Information**
- Data processing progress
- Model training status for each algorithm
- Anomaly detection results and statistics
- Performance metrics and timing

## 🔄 **Upgrade Paths**

### **From V0 to V1**
The system has been upgraded from traditional statistical methods (V0) to ML-based ensemble detection (V1):

- **V0**: Z-score, rolling statistics, simple thresholds
- **V1**: Ensemble ML models, advanced features, adaptive thresholds

### **Future Enhancements**
- Additional anomaly detection algorithms
- Real-time streaming capabilities
- Advanced visualization dashboards
- API endpoints for integration

## 📞 **Support & Maintenance**

### **Regular Maintenance**
- Monitor log files for errors
- Clear old log files periodically
- Update dependencies as needed
- Backup MongoDB data regularly

### **Performance Monitoring**
- Track anomaly detection rates
- Monitor processing times
- Check memory usage patterns
- Validate threshold effectiveness

---

## 🎯 **Quick Reference Commands**

```bash
# Start fresh
docker-compose down -v && docker-compose up -d --build

# Run all models
docker exec anomaly-detection python V0_model_USLevel/main.py
docker exec anomaly-detection python V0_model_StateLevel/main.py  
docker exec anomaly-detection python V0_model_SOCLevel/main.py

# Check status
docker ps
docker-compose logs -f

# Access data
docker exec -it anomaly-detection-mongodb mongosh
```

---

## 🔧 **Advanced Customization Guide**

### **Adding New Anomaly Detection Models**

To add your own custom anomaly detection model:

```python
# 1. Create custom model class
class CustomAnomalyModel:
    def __init__(self, custom_param1, custom_param2):
        self.param1 = custom_param1
        self.param2 = custom_param2
    
    def fit(self, X):
        # Your custom training logic
        pass
    
    def predict(self, X):
        # Your custom prediction logic
        pass

# 2. Integrate into AnomalyDetectorV1
def _train_models(self, df_feat: pd.DataFrame, feature_cols):
    # ... existing code ...
    
    # Add your custom model
    if "custom" in self.models:
        try:
            custom_mdl = CustomAnomalyModel(param1=value1, param2=value2)
            custom_mdl.fit(X_pyod)
            fitted["custom"] = ("custom", custom_mdl)
            self.logger.info("Trained custom model successfully")
        except Exception as e:
            self.logger.error(f"Failed to train custom model: {e}")
```

### **Custom Data Sources**

To integrate with different data sources:

```python
# 1. Database connections (PostgreSQL, MySQL, etc.)
import psycopg2
import mysql.connector

# 2. Cloud storage (AWS S3, Google Cloud Storage)
import boto3
from google.cloud import storage

# 3. API endpoints
import requests
import aiohttp

# 4. Stream processing (Kafka, RabbitMQ)
from kafka import KafkaConsumer
import pika

# Example: Custom data loader
class CustomDataLoader:
    def __init__(self, source_type, connection_params):
        self.source_type = source_type
        self.connection_params = connection_params
    
    def load_data(self):
        if self.source_type == "postgresql":
            return self._load_from_postgresql()
        elif self.source_type == "s3":
            return self._load_from_s3()
        elif self.source_type == "api":
            return self._load_from_api()
```

### **Performance Optimization**

#### **Memory Management**
```python
# Customize batch processing
class CustomPipeline:
    def __init__(self, memory_limit_gb=8):
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.batch_size = self._calculate_optimal_batch_size()
    
    def _calculate_optimal_batch_size(self):
        # Dynamic batch size based on available memory
        available_memory = psutil.virtual_memory().available
        return min(50000, int(available_memory * 0.1 / 1024))  # Use 10% of available memory
```

#### **Parallel Processing**
```python
# Enable parallel processing for multiple models
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def _train_models_parallel(self, df_feat: pd.DataFrame, feature_cols):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for model_name in self.models:
            future = executor.submit(self._train_single_model, model_name, df_feat, feature_cols)
            futures.append((model_name, future))
        
        # Collect results
        for model_name, future in futures:
            try:
                result = future.result()
                if result:
                    self.fitted_models_[model_name] = result
            except Exception as e:
                self.logger.error(f"Model {model_name} training failed: {e}")
```

### **Production Deployment**

#### **Environment-Specific Configs**
```python
# config.py - Environment-based configuration
import os

class Config:
    # Development
    DEV = {
        'batch_size': 25000,
        'rolling_window': 30,
        'log_level': 'DEBUG',
        'mongo_uri': 'mongodb://localhost:27017'
    }
    
    # Production
    PROD = {
        'batch_size': 100000,
        'rolling_window': 90,
        'log_level': 'WARNING',
        'mongo_uri': 'mongodb://prod-cluster:27017'
    }
    
    # Get current environment
    @staticmethod
    def get_config():
        env = os.getenv('ENVIRONMENT', 'DEV').upper()
        return getattr(Config, env, Config.DEV)
```

#### **Health Checks & Monitoring**
```python
# health_check.py - System health monitoring
import psutil
import time

class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def check_system_health(self):
        health_status = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time,
            'mongodb_connection': self._check_mongodb(),
            'model_status': self._check_models()
        }
        return health_status
    
    def _check_mongodb(self):
        try:
            # Test MongoDB connection
            return True
        except:
            return False
```

### **Custom Output Formats**

#### **JSON Output**
```python
# Custom JSON output with metadata
def export_json_results(self, df_out, output_path):
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_version': 'V1',
            'total_records': len(df_out),
            'anomaly_count': df_out['is_anomaly'].sum(),
            'anomaly_rate': float(df_out['is_anomaly'].mean())
        },
        'anomalies': df_out[df_out['is_anomaly'] == 1].to_dict('records'),
        'summary': self._generate_summary_stats(df_out)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
```

#### **Database Output**
```python
# Store results in database
def store_results_in_db(self, df_out, db_connection):
    # Create results table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS anomaly_results (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP,
        geo_level VARCHAR(100),
        src VARCHAR(100),
        date DATE,
        job_count INTEGER,
        anomaly_score FLOAT,
        is_anomaly BOOLEAN,
        threshold FLOAT
    )
    """
    
    # Insert results
    for _, row in df_out.iterrows():
        insert_sql = """
        INSERT INTO anomaly_results 
        (timestamp, geo_level, src, date, job_count, anomaly_score, is_anomaly, threshold)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (datetime.now(), row['geo_level'], row['src'], row['date'], 
                 row['job_count'], row['score_ens'], row['is_anomaly'], row['threshold'])
        db_connection.execute(insert_sql, values)
```

---

**Note**: Replace `<REPOSITORY_URL>` with your actual GitHub repository URL before cloning.