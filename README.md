# CANARIA Anomaly Detection System

A comprehensive job market anomaly detection system that identifies unusual patterns in job posting data across three levels: US Level, State Level, and SOC (Standard Occupational Classification) Level.

## 🏗️ **System Architecture**

This system consists of three independent anomaly detection models, each analyzing job data at different granularities:

- **US Level Model**: Detects anomalies across geographic regions (geo_level × source)
- **State Level Model**: Detects anomalies at state/territory level (state × source)  
- **SOC Level Model**: Detects anomalies by job categories (SOC code × source)

All models use **Version 1 (V1) ML-based ensemble detection** with advanced feature engineering and multiple anomaly detection algorithms.

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

#### **Feature Engineering (18 features)**
- **Rolling Statistics**: 7, 14, 28-day rolling means, standard deviations, MAD
- **Percentage Changes**: 1-day and 7-day changes
- **Z-Scores**: 7-day and 28-day z-scores
- **Temporal Features**: Day of week, week of year, month
- **Long-term Patterns**: Expanding means, share vs long-term average

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

**Note**: Replace `<REPOSITORY_URL>` with your actual GitHub repository URL before cloning.