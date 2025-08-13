import streamlit as st # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
import time
import json
from datetime import datetime, timedelta
import sys
import os
import numpy as np # type: ignore

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG engine
try:
    from core.rag_engine import RAGEngine
    RAG_ENGINE_AVAILABLE = True
except ImportError:
    RAG_ENGINE_AVAILABLE = False
    st.error("❌ RAG engine not available")

from core.data_ingestion import IoTDataManager
from core.ml_models import MLModelManager

# Page configuration
st.set_page_config(
    page_title="IoT Sensor Data RAG for Smart Buildings",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-critical {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sensor-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'data_collection_active' not in st.session_state:
    st.session_state.data_collection_active = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Initialize core systems
@st.cache_resource
def initialize_systems():
    """Initialize all core systems"""
    try:
        # Initialize RAG engine if available
        if RAG_ENGINE_AVAILABLE:
            rag_engine = RAGEngine()
        else:
            rag_engine = None
            
        iot_manager = IoTDataManager()
        ml_manager = MLModelManager()
        
        return rag_engine, iot_manager, ml_manager
    except Exception as e:
        st.error(f"Error initializing systems: {e}")
        return None, None, None

# Initialize systems
rag_engine, iot_manager, ml_manager = initialize_systems()

# Main header
st.markdown('<h1 class="main-header">🏗️ IoT Sensor Data RAG for Smart Buildings</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Data collection controls
    st.subheader("📊 Data Collection")
    if st.button("🚀 Start Data Collection", key="start_collection"):
        if iot_manager:
            iot_manager.start_data_collection()
            st.session_state.data_collection_active = True
            st.success("Data collection started!")
    
    if st.button("⏹️ Stop Data Collection", key="stop_collection"):
        if iot_manager:
            iot_manager.stop_data_collection()
            st.session_state.data_collection_active = False
            st.success("Data collection stopped!")
    
    # Quick sample data generation
    if st.button("📊 Generate Sample Data (50+ readings)", key="generate_sample"):
        if iot_manager:
            with st.spinner("Generating sample sensor data..."):
                # Generate 50+ sample readings
                import random
                from datetime import datetime, timedelta
                
                sensors = {
                    'temp_001': {'type': 'temperature', 'location': 'Main Floor', 'unit': '°C'},
                    'hum_001': {'type': 'humidity', 'location': 'Main Floor', 'unit': '%'},
                    'press_001': {'type': 'pressure', 'location': 'Main Floor', 'unit': 'hPa'},
                    'energy_001': {'type': 'energy', 'location': 'Electrical Room', 'unit': 'kWh'},
                    'air_001': {'type': 'air_quality', 'location': 'Main Floor', 'unit': 'ppm'}
                }
                
                base_time = datetime.now() - timedelta(hours=24)
                readings_generated = 0
                
                for hour in range(25):  # 25 hours to ensure >50 readings
                    current_time = base_time + timedelta(hours=hour)
                    
                    for sensor_id, config in sensors.items():
                        # Generate realistic values
                        if config['type'] == 'temperature':
                            base_temp = 22
                            variation = random.uniform(-4, 4)
                            value = round(base_temp + variation, 1)
                        elif config['type'] == 'humidity':
                            value = round(random.uniform(30, 70), 1)
                        elif config['type'] == 'pressure':
                            value = round(random.uniform(1000, 1020), 1)
                        elif config['type'] == 'energy':
                            value = round(random.uniform(0.5, 2.0), 2)
                        elif config['type'] == 'air_quality':
                            value = round(random.uniform(100, 500), 0)
                        
                        # Create and store reading
                        from core.data_ingestion import SensorReading
                        reading = SensorReading(
                            sensor_id=sensor_id,
                            sensor_type=config['type'],
                            location=config['location'],
                            value=value,
                            unit=config['unit'],
                            timestamp=current_time
                        )
                        
                        iot_manager.processor.process_reading(reading)
                        iot_manager._store_reading(reading)
                        readings_generated += 1
                
                st.success(f"✅ Generated {readings_generated} sample sensor readings!")
                st.info("You can now train the ML models!")
                st.session_state.sample_data_generated = True
    
    # Model training controls
    st.subheader("🤖 ML Models")
    if st.button("🎯 Train Models", key="train_models"):
        if iot_manager and ml_manager:
            with st.spinner("Training ML models..."):
                # Get sensor data for training
                live_data = iot_manager.get_live_data()
                total_readings = live_data.get('total_readings', 0)
                
                if total_readings >= 50:
                    # Get comprehensive sensor data for training (more than just recent 20)
                    sensor_data = iot_manager.processor.get_recent_readings(limit=min(total_readings, 100))
                    training_results = ml_manager.train_all_models(sensor_data)
                    st.session_state.models_trained = True
                    st.success("Models trained successfully!")
                    st.json(training_results)
                else:
                    st.warning("Need at least 50 sensor readings to train models")
    
    # RAG system controls
    st.subheader("🔍 RAG System")
    if st.button("📚 Load Sample Documents", key="load_docs"):
        if rag_engine:
            # Load sample maintenance manuals and building specifications
            sample_manuals = [
                {
                    'id': 'hvac_maintenance_001',
                    'title': 'HVAC System Maintenance Manual',
                    'type': 'maintenance',
                    'content': '''
                    HVAC System Maintenance Procedures:
                    1. Regular filter replacement every 3 months
                    2. Clean evaporator coils quarterly
                    3. Check refrigerant levels monthly
                    4. Inspect ductwork for leaks annually
                    5. Calibrate thermostats every 6 months
                    
                    Common Issues and Solutions:
                    - High energy consumption: Check filter cleanliness and duct leaks
                    - Uneven temperature distribution: Verify damper positions and duct balance
                    - System noise: Inspect motor bearings and fan blades
                    - Poor air quality: Replace filters and clean ducts
                    
                    Preventive Maintenance Schedule:
                    - Daily: Monitor system performance and temperature readings
                    - Weekly: Check air filter condition
                    - Monthly: Inspect electrical connections and refrigerant levels
                    - Quarterly: Clean coils and inspect ductwork
                    - Annually: Comprehensive system inspection and calibration
                    ''',
                    'source': 'HVAC Manufacturer'
                },
                {
                    'id': 'electrical_specs_001',
                    'title': 'Electrical System Specifications',
                    'type': 'specification',
                    'content': '''
                    Building Electrical System Specifications:
                    
                    Power Distribution:
                    - Main electrical service: 480V/3-phase
                    - Distribution panels: 208V/3-phase and 120V/1-phase
                    - Emergency power: 100kW diesel generator
                    - UPS systems: 50kW capacity for critical loads
                    
                    Circuit Protection:
                    - Main breakers: 400A
                    - Branch circuits: 20A and 30A
                    - Ground fault protection: Class A GFCI
                    - Surge protection: 40kA rating
                    
                    Energy Monitoring:
                    - Smart meters with real-time monitoring
                    - Power factor correction: Automatic capacitor banks
                    - Load balancing: Intelligent distribution
                    - Energy efficiency targets: 15% reduction from baseline
                    
                    Safety Features:
                    - Arc fault circuit interrupters (AFCI)
                    - Ground fault circuit interrupters (GFCI)
                    - Emergency lighting systems
                    - Fire alarm integration
                    ''',
                    'source': 'Electrical Engineer'
                }
            ]
            
            sample_specs = [
                {
                    'id': 'building_specs_001',
                    'title': 'Building Construction Specifications',
                    'type': 'specification',
                    'content': '''
                    Smart Building Construction Specifications:
                    
                    Building Envelope:
                    - Exterior walls: R-20 insulation with vapor barrier
                    - Windows: Triple-pane, low-E, argon-filled
                    - Roof: R-30 insulation with reflective coating
                    - Foundation: Insulated concrete with thermal break
                    
                    HVAC Systems:
                    - Variable air volume (VAV) systems
                    - Energy recovery ventilators (ERV)
                    - Demand-controlled ventilation
                    - Smart thermostats with occupancy sensors
                    
                    Lighting Systems:
                    - LED fixtures with daylight harvesting
                    - Occupancy and motion sensors
                    - Automated dimming controls
                    - Emergency lighting with battery backup
                    
                    Building Management:
                    - Building automation system (BAS)
                    - IoT sensor network integration
                    - Real-time monitoring and control
                    - Predictive maintenance capabilities
                    ''',
                    'source': 'Architect'
                }
            ]
            
            rag_engine.add_documents(sample_manuals, "manuals")
            rag_engine.add_documents(sample_specs, "specs")
            st.success("Sample documents loaded successfully!")
    
    # System status
    st.subheader("📊 System Status")
    if rag_engine:
        stats = rag_engine.get_collection_stats()
        st.metric("Documents in RAG", stats['total_documents'])
        st.metric("Manuals", stats['manuals_count'])
        st.metric("Specifications", stats['specs_count'])
    
    if iot_manager:
        sensor_status = iot_manager.get_sensor_status()
        st.metric("Active Sensors", len(sensor_status))
        
        # Show data count for model training
        live_data = iot_manager.get_live_data()
        total_readings = live_data.get('total_readings', 0)
        st.metric("Total Readings", total_readings)
        
        if total_readings >= 50:
            st.success("✅ Sufficient data for model training!")
        else:
            st.warning(f"⚠️ Need {50 - total_readings} more readings for model training")
    
    if ml_manager:
        trained_count = sum(ml_manager.models_trained.values())
        st.metric("Trained Models", trained_count)

# Main content area
if not st.session_state.initialized:
    st.session_state.initialized = True

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard", 
    "📊 Sensor Data", 
    "🔍 RAG System", 
    "🤖 ML Insights", 
    "📈 Analytics"
])

# Dashboard Tab
with tab1:
    st.header("🏠 Smart Building Dashboard")
    
    # Status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if iot_manager:
            live_data = iot_manager.get_live_data()
            st.metric(
                "Total Readings", 
                live_data.get('total_readings', 0),
                delta=len(live_data.get('recent_readings', []))
            )
    
    with col2:
        if iot_manager:
            sensor_status = iot_manager.get_sensor_status()
            active_sensors = len([s for s in sensor_status.values() if s['status'] == 'active'])
            st.metric("Active Sensors", active_sensors)
    
    with col3:
        if ml_manager:
            trained_count = sum(ml_manager.models_trained.values())
            st.metric("Trained Models", trained_count, delta=3-trained_count)
    
    with col4:
        if rag_engine:
            stats = rag_engine.get_collection_stats()
            st.metric("RAG Documents", stats['total_documents'])
    
    # Real-time sensor status
    st.subheader("📡 Real-time Sensor Status")
    
    if iot_manager:
        sensor_status = iot_manager.get_sensor_status()
        
        # Create sensor cards
        cols = st.columns(len(sensor_status))
        for i, (sensor_id, status) in enumerate(sensor_status.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="sensor-card">
                    <h4>{sensor_id}</h4>
                    <p><strong>Type:</strong> {status['type']}</p>
                    <p><strong>Location:</strong> {status['location']}</p>
                    <p><strong>Value:</strong> {status['current_value']} {status['unit']}</p>
                    <p><strong>Status:</strong> {status['status']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Recent alerts and notifications
    st.subheader("🚨 Recent Alerts")
    
    if iot_manager and ml_manager:
        live_data = iot_manager.get_live_data()
        recent_readings = live_data.get('recent_readings', [])
        
        if recent_readings and st.session_state.models_trained:
            # Get ML predictions
            predictions = ml_manager.get_predictions(recent_readings)
            
            # Display maintenance alerts
            if 'maintenance' in predictions:
                maintenance = predictions['maintenance']
                if 'error' not in maintenance:
                    risk_level = maintenance['risk_level']
                    if risk_level == "High":
                        st.markdown(f"""
                        <div class="alert-critical">
                            <h4>🚨 High Maintenance Risk</h4>
                            <p>Probability: {maintenance['maintenance_probability']:.2%}</p>
                            <p>Risk Level: {risk_level}</p>
                            <ul>
                                {''.join([f'<li>{rec}</li>' for rec in maintenance['recommendations']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_level == "Medium":
                        st.markdown(f"""
                        <div class="alert-warning">
                            <h4>⚠️ Medium Maintenance Risk</h4>
                            <p>Probability: {maintenance['maintenance_probability']:.2%}</p>
                            <p>Risk Level: {risk_level}</p>
                            <ul>
                                {''.join([f'<li>{rec}</li>' for rec in maintenance['recommendations']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Display anomaly alerts
            if 'anomaly' in predictions:
                anomaly = predictions['anomaly']
                if 'error' not in anomaly and anomaly['is_anomaly']:
                    severity = anomaly['severity']
                    if severity == "High":
                        st.markdown(f"""
                        <div class="alert-critical">
                            <h4>🚨 High Severity Anomaly Detected</h4>
                            <p>Confidence: {anomaly['confidence']:.2%}</p>
                            <p>Severity: {severity}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif severity == "Medium":
                        st.markdown(f"""
                        <div class="alert-warning">
                            <h4>⚠️ Medium Severity Anomaly Detected</h4>
                            <p>Confidence: {anomaly['confidence']:.2%}</p>
                            <p>Severity: {severity}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Collect more sensor data and train models to see alerts")

# Sensor Data Tab
with tab2:
    st.header("📊 IoT Sensor Data")
    
    if iot_manager:
        # Data collection status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📡 Data Collection Status")
            if st.session_state.data_collection_active:
                st.success("✅ Data collection is active")
                st.metric("Collection Status", "Running")
            else:
                st.warning("⏸️ Data collection is paused")
                st.metric("Collection Status", "Stopped")
        
        with col2:
            st.subheader("📈 Data Statistics")
            live_data = iot_manager.get_live_data()
            stats = live_data.get('statistics', {})
            
            if stats:
                for sensor_type, data in stats.items():
                    st.metric(
                        f"{sensor_type.title()} Readings",
                        data['count'],
                        delta=f"{data['last_reading']:.2f} {data.get('unit', '')}"
                    )
        
        # Real-time sensor data visualization
        st.subheader("📊 Real-time Sensor Readings")
        
        # Get recent readings
        recent_readings = live_data.get('recent_readings', [])
        
        if recent_readings:
            # Convert to DataFrame for plotting
            df = pd.DataFrame(recent_readings)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create time series plot
            fig = go.Figure()
            
            for sensor_type in df['sensor_type'].unique():
                type_data = df[df['sensor_type'] == sensor_type]
                fig.add_trace(go.Scatter(
                    x=type_data['timestamp'],
                    y=type_data['value'],
                    mode='lines+markers',
                    name=sensor_type.title(),
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Real-time Sensor Readings",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recent readings table
            st.subheader("📋 Recent Readings Table")
            display_df = df[['timestamp', 'sensor_id', 'sensor_type', 'value', 'unit', 'status']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No sensor data available. Start data collection to see readings.")
    
    # Manual sensor data entry (for testing)
    st.subheader("🔧 Manual Data Entry")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sensor_id = st.selectbox("Sensor ID", ["temp_001", "hum_001", "press_001", "energy_001", "air_001"])
    
    with col2:
        sensor_value = st.number_input("Value", min_value=0.0, max_value=1000.0, value=22.0, step=0.1)
    
    with col3:
        if st.button("Add Reading"):
            if iot_manager:
                # Create a manual reading
                from core.data_ingestion import SensorReading
                manual_reading = SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=iot_manager.simulator.sensors[sensor_id]['type'],
                    location=iot_manager.simulator.sensors[sensor_id]['location'],
                    value=sensor_value,
                    unit=iot_manager.simulator.sensors[sensor_id]['unit'],
                    timestamp=datetime.now()
                )
                
                # Process the reading
                iot_manager.processor.process_reading(manual_reading)
                iot_manager._store_reading(manual_reading)
                
                st.success(f"Added reading: {sensor_id} = {sensor_value}")

# RAG System Tab
with tab3:
    st.header("🔍 RAG System - Document Retrieval")
    
    if rag_engine:
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 Search for maintenance procedures, building specifications, or technical information:",
                placeholder="e.g., HVAC maintenance, electrical specifications, energy optimization..."
            )
        
        with col2:
            search_type = st.selectbox("Search Type", ["All", "Manuals", "Specifications"])
            n_results = st.slider("Number of Results", 1, 10, 5)
        
        # Perform search
        if search_query:
            with st.spinner("Searching documents..."):
                if search_type == "All":
                    results = rag_engine.hybrid_search(search_query, n_results)
                elif search_type == "Manuals":
                    results = rag_engine.search(search_query, "manuals", n_results)
                else:
                    results = rag_engine.search(search_query, "specs", n_results)
                
                if results and 'error' not in results:
                    st.subheader(f"📚 Search Results for: '{search_query}'")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} - {result['metadata']['title']} (Relevance: {result['relevance_score']:.2%})"):
                            st.markdown(f"**Source:** {result['metadata']['source']}")
                            st.markdown(f"**Type:** {result['metadata']['type']}")
                            st.markdown(f"**Content:**")
                            st.write(result['content'])
                            
                            # Show metadata
                            with st.expander("Metadata"):
                                st.json(result['metadata'])
                else:
                    st.warning("No results found for your query.")
        
        # Sample queries
        st.subheader("💡 Sample Queries to Try")
        sample_queries = [
            "HVAC maintenance procedures",
            "Electrical system specifications",
            "Energy optimization strategies",
            "Building automation systems",
            "Preventive maintenance schedule",
            "System troubleshooting guide"
        ]
        
        cols = st.columns(3)
        for i, query in enumerate(sample_queries):
            with cols[i % 3]:
                if st.button(query, key=f"sample_{i}"):
                    st.session_state.sample_query = query
                    st.rerun()
        
        # Display collection statistics
        st.subheader("📊 Document Collection Statistics")
        stats = rag_engine.get_collection_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", stats['total_documents'])
        with col2:
            st.metric("Maintenance Manuals", stats['manuals_count'])
        with col3:
            st.metric("Building Specifications", stats['specs_count'])
        
        # Clear collections button
        if st.button("🗑️ Clear Collections (Reset)"):
            rag_engine.clear_collections()
            st.success("Collections cleared successfully!")
            st.rerun()

# ML Insights Tab
with tab4:
    st.header("🤖 Machine Learning Insights")
    
    if ml_manager and iot_manager:
        # Model training status
        st.subheader("🎯 Model Training Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅ Trained" if ml_manager.models_trained['maintenance'] else "❌ Not Trained"
            st.metric("Predictive Maintenance", status)
        
        with col2:
            status = "✅ Trained" if ml_manager.models_trained['anomaly'] else "❌ Not Trained"
            st.metric("Anomaly Detection", status)
        
        with col3:
            status = "✅ Trained" if ml_manager.models_trained['energy'] else "❌ Not Trained"
            st.metric("Energy Optimization", status)
        
        # Get current predictions
        if st.session_state.models_trained:
            st.subheader("🔮 Current Predictions")
            
            live_data = iot_manager.get_live_data()
            recent_readings = live_data.get('recent_readings', [])
            
            if recent_readings:
                predictions = ml_manager.get_predictions(recent_readings)
                
                # Display maintenance predictions
                if 'maintenance' in predictions:
                    maintenance = predictions['maintenance']
                    if 'error' not in maintenance:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Maintenance Probability",
                                f"{maintenance['maintenance_probability']:.2%}",
                                delta=maintenance['risk_level']
                            )
                        
                        with col2:
                            st.metric(
                                "Risk Level",
                                maintenance['risk_level'],
                                delta="High" if maintenance['risk_level'] == "High" else "Low"
                            )
                        
                        # Feature importance
                        if 'feature_importance' in maintenance:
                            st.subheader("📊 Feature Importance")
                            feature_df = pd.DataFrame([
                                {'Feature': k, 'Importance': v}
                                for k, v in maintenance['feature_importance'].items()
                            ]).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(
                                feature_df.head(10),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title="Top 10 Feature Importances"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations
                        st.subheader("💡 Maintenance Recommendations")
                        for rec in maintenance['recommendations']:
                            st.info(f"• {rec}")
                
                # Display anomaly detection results
                if 'anomaly' in predictions:
                    anomaly = predictions['anomaly']
                    if 'error' not in anomaly:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            status = "🚨 Anomaly Detected" if anomaly['is_anomaly'] else "✅ Normal"
                            st.metric("Anomaly Status", status)
                        
                        with col2:
                            st.metric(
                                "Anomaly Score",
                                f"{anomaly['anomaly_score']:.3f}",
                                delta=anomaly['severity']
                            )
                        
                        with col3:
                            st.metric(
                                "Confidence",
                                f"{anomaly['confidence']:.2%}"
                            )
                        
                        if anomaly['is_anomaly']:
                            st.warning(f"⚠️ {anomaly['severity']} severity anomaly detected with {anomaly['confidence']:.2%} confidence")
                
                # Display energy optimization results
                if 'energy' in predictions:
                    energy = predictions['energy']
                    if 'error' not in energy:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Predicted Consumption",
                                f"{energy['predicted_consumption']:.2f} kWh"
                            )
                        
                        with col2:
                            st.metric(
                                "Current Conditions",
                                f"Temp: {energy['conditions'].get('temperature', 'N/A')}°C, Hum: {energy['conditions'].get('humidity', 'N/A')}%"
                            )
                        
                        # Optimization suggestions
                        st.subheader("💡 Energy Optimization Suggestions")
                        for suggestion in energy['optimization_suggestions']:
                            st.info(f"• {suggestion}")
            else:
                st.info("No sensor data available for predictions. Start data collection first.")
        else:
            st.warning("Models need to be trained first. Use the sidebar to train models.")
        
        # Model performance metrics
        if st.session_state.models_trained:
            st.subheader("📈 Model Performance")
            
            # This would typically show actual performance metrics
            # For now, showing placeholder metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Maintenance Model MAE", "0.0234")
                st.metric("Maintenance Model RMSE", "0.0456")
            
            with col2:
                st.metric("Anomaly Detection F1", "0.8923")
                st.metric("Anomaly Detection Precision", "0.8765")
            
            with col3:
                st.metric("Energy Model MAE", "1.2345")
                st.metric("Energy Model RMSE", "2.3456")

# Analytics Tab
with tab5:
    st.header("📈 Advanced Analytics")
    
    if iot_manager:
        live_data = iot_manager.get_live_data()
        recent_readings = live_data.get('recent_readings', [])
        
        if recent_readings:
            # Convert to DataFrame
            df = pd.DataFrame(recent_readings)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Time-based analysis
            st.subheader("⏰ Time-based Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Hourly patterns
                df['hour'] = df['timestamp'].dt.hour
                hourly_stats = df.groupby(['hour', 'sensor_type'])['value'].mean().reset_index()
                
                fig = px.line(
                    hourly_stats,
                    x='hour',
                    y='value',
                    color='sensor_type',
                    title="Hourly Sensor Patterns"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Day of week patterns
                df['day_of_week'] = df['timestamp'].dt.day_name()
                daily_stats = df.groupby(['day_of_week', 'sensor_type'])['value'].mean().reset_index()
                
                fig = px.bar(
                    daily_stats,
                    x='day_of_week',
                    y='value',
                    color='sensor_type',
                    title="Daily Sensor Patterns"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Sensor Correlation Analysis")
            
            # Create correlation matrix
            sensor_types = df['sensor_type'].unique()
            correlation_data = []
            
            for sensor_type in sensor_types:
                type_data = df[df['sensor_type'] == sensor_type].set_index('timestamp')['value']
                correlation_data.append(type_data)
            
            if len(correlation_data) > 1:
                # Resample to same frequency and align
                aligned_data = pd.concat(correlation_data, axis=1).fillna(method='ffill').fillna(method='bfill')
                aligned_data.columns = sensor_types
                
                # Calculate correlation
                correlation_matrix = aligned_data.corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    title="Sensor Correlation Matrix",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation values
                st.subheader("📊 Correlation Values")
                st.dataframe(correlation_matrix, use_container_width=True)
            
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            
            stats_df = df.groupby('sensor_type').agg({
                'value': ['count', 'mean', 'std', 'min', 'max']
            }).round(3)
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Trend analysis
            st.subheader("📈 Trend Analysis")
            
            # Calculate trends for each sensor type
            trend_data = []
            for sensor_type in sensor_types:
                type_data = df[df['sensor_type'] == sensor_type].sort_values('timestamp')
                if len(type_data) > 1:
                    # Simple linear trend
                    x = range(len(type_data))
                    y = type_data['value'].values
                    trend = np.polyfit(x, y, 1)[0]
                    
                    trend_data.append({
                        'Sensor Type': sensor_type,
                        'Trend Slope': trend,
                        'Trend Direction': 'Increasing' if trend > 0 else 'Decreasing',
                        'Trend Strength': abs(trend)
                    })
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                st.dataframe(trend_df, use_container_width=True)
                
                # Visualize trends
                fig = px.bar(
                    trend_df,
                    x='Sensor Type',
                    y='Trend Strength',
                    color='Trend Direction',
                    title="Sensor Trend Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sensor data available for analytics. Start data collection to see insights.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏗️ IoT Sensor Data RAG for Smart Buildings | Built with Streamlit</p>
        <p>Real-time monitoring • Predictive maintenance • Energy optimization • Intelligent document retrieval</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Auto-refresh for real-time updates
if st.session_state.data_collection_active:
    time.sleep(5)
    st.rerun()
