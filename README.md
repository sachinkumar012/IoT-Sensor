# IoT Sensor Data RAG for Smart Buildings

A comprehensive Retrieval-Augmented Generation (RAG) system that processes IoT sensor data, maintenance manuals, and building specifications to provide predictive maintenance insights and operational optimization for smart buildings.

## 🏗️ System Overview

This system combines real-time IoT sensor data processing with intelligent document retrieval to provide:
- **Real-time sensor monitoring** and data ingestion
- **Predictive maintenance** insights using ML algorithms
- **Anomaly detection** and alert systems
- **Operational efficiency** optimization recommendations
- **Intelligent querying** of maintenance manuals and building specs

## 🚀 Key Features

### IoT Data Processing
- Real-time sensor data streaming via MQTT
- Multi-sensor data fusion and correlation
- Historical data analysis and trending
- Sensor health monitoring

### RAG System
- Vector database for maintenance manuals and building specifications
- Semantic search and retrieval
- Context-aware response generation
- Relevance scoring and ranking

### Predictive Analytics
- Equipment failure prediction models
- Energy efficiency optimization
- Maintenance scheduling recommendations
- Performance trend analysis

### Anomaly Detection
- Real-time anomaly identification
- Alert system with severity levels
- Automated response recommendations
- Historical anomaly tracking

## 🛠️ Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IoT Sensors  │    │  Data Pipeline  │    │   RAG Engine   │
│                │───▶│                │───▶│                │
│ • Temperature  │    │ • MQTT Client   │    │ • Vector DB    │
│ • Humidity     │    │ • Data Fusion  │    │ • Embeddings   │
│ • Pressure     │    │ • Preprocessing │    │ • Retrieval    │
│ • Energy       │    │ • Storage      │    │ • Generation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Analytics     │    │  Web Interface  │
                       │                │    │                │
                       │ • ML Models    │    │ • Streamlit App │
                       │ • Predictions  │    │ • Real-time     │
                       │ • Optimization │    │ • Dashboards    │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
IoT Sensor/
├── app/                    # Main application
│   ├── main.py            # Streamlit app entry point
│   ├── components/         # UI components
│   ├── pages/             # App pages
│   └── utils/             # Utility functions
├── core/                   # Core system components
│   ├── data_ingestion.py  # IoT data ingestion
│   ├── rag_engine.py      # RAG system core
│   ├── ml_models.py       # ML algorithms
│   └── anomaly_detection.py # Anomaly detection
├── data/                   # Data storage
│   ├── sensors/           # Sensor data
│   ├── manuals/           # Maintenance manuals
│   └── specs/             # Building specifications
├── models/                 # Trained models
├── config/                 # Configuration files
├── tests/                  # Test files
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd IoT-Sensor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Run the Application
```bash
streamlit run app/main.py
```

### 5. Access the Web Interface
Open your browser and navigate to `http://localhost:8501`

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for advanced language models
- `MQTT_BROKER`: MQTT broker address for sensor data
- `MONGODB_URI`: MongoDB connection string
- `CHROMA_PERSIST_DIR`: ChromaDB persistence directory

### Sensor Configuration
Configure your IoT sensors in `config/sensors.yaml`:
```yaml
sensors:
  temperature:
    id: "temp_001"
    type: "temperature"
    location: "HVAC_room"
    unit: "celsius"
  humidity:
    id: "hum_001"
    type: "humidity"
    location: "HVAC_room"
    unit: "percentage"
```

## 📊 Data Sources

### IoT Sensors
- Temperature sensors
- Humidity sensors
- Pressure sensors
- Energy consumption meters
- Air quality sensors
- Motion detectors

### Documents
- Equipment maintenance manuals
- Building specifications
- Operating procedures
- Safety guidelines
- Energy efficiency reports

## 🧠 Machine Learning Models

### Predictive Maintenance
- Random Forest for equipment failure prediction
- LSTM networks for time series forecasting
- Isolation Forest for anomaly detection

### Energy Optimization
- Linear regression for consumption prediction
- Clustering for usage pattern analysis
- Optimization algorithms for efficiency

## 📈 Performance Metrics

- **Retrieval Accuracy**: Precision@K, Recall@K
- **Response Latency**: Average query response time
- **Anomaly Detection**: F1-score, Precision, Recall
- **Predictive Accuracy**: MAE, RMSE for maintenance predictions

## 🚀 Deployment

### Local Development
```bash
streamlit run app/main.py
```

### Production Deployment
```bash
# Using Docker
docker build -t iot-sensor-rag .
docker run -p 8501:8501 iot-sensor-rag

# Using Streamlit Cloud
# Deploy directly to Streamlit Cloud
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for language models
- HuggingFace for sentence transformers
- ChromaDB for vector database
- Streamlit for web interface framework

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Contact the development team
- Check the documentation in the `docs/` folder

---

**Built with ❤️ for Smart Building Innovation**
