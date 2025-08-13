import yaml
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return get_default_config()
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values
    
    Returns:
        Default configuration dictionary
    """
    return {
        'system': {
            'name': 'IoT Sensor Data RAG for Smart Buildings',
            'version': '1.0.0',
            'environment': 'development',
            'debug': True,
            'log_level': 'INFO'
        },
        'rag': {
            'engine': {
                'model': 'all-MiniLM-L6-v2',
                'chunk_size': 512,
                'chunk_overlap': 50,
                'max_results': 10,
                'similarity_threshold': 0.7
            }
        },
        'iot': {
            'data_collection': {
                'simulation_interval': 5.0,
                'max_buffer_size': 10000
            }
        }
    }

def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml") -> bool:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def create_directory_structure(base_path: str = ".") -> None:
    """
    Create the necessary directory structure for the application
    
    Args:
        base_path: Base path for creating directories
    """
    directories = [
        "data/sensors",
        "data/manuals", 
        "data/specs",
        "data/chroma_db",
        "models",
        "logs",
        "config",
        "tests"
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")

def load_sample_documents() -> Dict[str, list]:
    """
    Load sample documents for demonstration
    
    Returns:
        Dictionary containing sample manuals and specifications
    """
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
            'id': 'electrical_maintenance_001',
            'title': 'Electrical System Maintenance Guide',
            'type': 'maintenance',
            'content': '''
            Electrical System Maintenance Procedures:
            1. Monthly inspection of electrical panels
            2. Quarterly testing of emergency systems
            3. Annual thermal imaging of electrical equipment
            4. Regular cleaning of electrical rooms
            5. Testing of ground fault protection systems
            
            Safety Procedures:
            - Always follow lockout/tagout procedures
            - Use appropriate PPE when working on electrical equipment
            - Test circuits before working on them
            - Keep electrical rooms clean and organized
            
            Common Electrical Issues:
            - Loose connections causing overheating
            - Corroded terminals affecting conductivity
            - Worn insulation creating safety hazards
            - Improper grounding causing equipment damage
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
    
    return {
        'manuals': sample_manuals,
        'specs': sample_specs
    }

def generate_sample_sensor_data(num_readings: int = 100) -> list:
    """
    Generate sample sensor data for testing
    
    Args:
        num_readings: Number of readings to generate
        
    Returns:
        List of sensor readings
    """
    import random
    from datetime import datetime, timedelta
    
    sensor_configs = {
        'temp_001': {'type': 'temperature', 'unit': 'celsius', 'base': 22.0, 'variance': 3.0},
        'hum_001': {'type': 'humidity', 'unit': 'percentage', 'base': 45.0, 'variance': 5.0},
        'press_001': {'type': 'pressure', 'unit': 'kPa', 'base': 101.3, 'variance': 0.5},
        'energy_001': {'type': 'energy', 'unit': 'kWh', 'base': 15.0, 'variance': 2.0},
        'air_001': {'type': 'air_quality', 'unit': 'ppm', 'base': 400.0, 'variance': 50.0}
    }
    
    readings = []
    base_time = datetime.now() - timedelta(hours=num_readings//10)
    
    for i in range(num_readings):
        timestamp = base_time + timedelta(minutes=i*6)  # 6-minute intervals
        
        for sensor_id, config in sensor_configs.items():
            # Add some realistic variation
            value = config['base'] + random.gauss(0, config['variance'] * 0.3)
            
            # Add time-based patterns
            hour = timestamp.hour
            if config['type'] == 'temperature':
                value += 2 * (hour - 12) / 12  # Daily temperature cycle
            elif config['type'] == 'energy':
                if 8 <= hour <= 18:  # Business hours
                    value *= 1.5
            
            reading = {
                'sensor_id': sensor_id,
                'sensor_type': config['type'],
                'location': 'HVAC_room' if sensor_id in ['temp_001', 'hum_001', 'press_001'] else 'main_panel',
                'value': round(value, 2),
                'unit': config['unit'],
                'timestamp': timestamp.isoformat(),
                'status': 'normal'
            }
            
            readings.append(reading)
    
    return readings

def export_data_to_csv(data: list, filename: str, output_dir: str = "data/exports") -> str:
    """
    Export data to CSV file
    
    Args:
        data: List of data dictionaries
        filename: Output filename
        output_dir: Output directory
        
    Returns:
        Path to exported file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(data)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Data exported to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return ""

def import_data_from_csv(filepath: str) -> list:
    """
    Import data from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of data dictionaries
    """
    try:
        df = pd.read_csv(filepath)
        data = df.to_dict('records')
        logger.info(f"Data imported from {filepath}: {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Error importing data: {e}")
        return []

def format_timestamp(timestamp: str) -> str:
    """
    Format timestamp for display
    
    Args:
        timestamp: ISO format timestamp string
        
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp

def calculate_statistics(data: list, sensor_type: str = None) -> Dict[str, Any]:
    """
    Calculate basic statistics for sensor data
    
    Args:
        data: List of sensor readings
        sensor_type: Optional filter for specific sensor type
        
    Returns:
        Dictionary of statistics
    """
    if not data:
        return {}
    
    df = pd.DataFrame(data)
    
    if sensor_type:
        df = df[df['sensor_type'] == sensor_type]
    
    if len(df) == 0:
        return {}
    
    numeric_data = pd.to_numeric(df['value'], errors='coerce').dropna()
    
    if len(numeric_data) == 0:
        return {}
    
    stats = {
        'count': len(numeric_data),
        'mean': float(numeric_data.mean()),
        'std': float(numeric_data.std()),
        'min': float(numeric_data.min()),
        'max': float(numeric_data.max()),
        'median': float(numeric_data.median()),
        'q25': float(numeric_data.quantile(0.25)),
        'q75': float(numeric_data.quantile(0.75))
    }
    
    return stats

def validate_sensor_reading(reading: Dict[str, Any]) -> bool:
    """
    Validate sensor reading data structure
    
    Args:
        reading: Sensor reading dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['sensor_id', 'sensor_type', 'value', 'unit', 'timestamp']
    
    # Check required fields
    for field in required_fields:
        if field not in reading:
            logger.warning(f"Missing required field: {field}")
            return False
    
    # Validate data types
    try:
        float(reading['value'])
        datetime.fromisoformat(reading['timestamp'].replace('Z', '+00:00'))
    except (ValueError, TypeError):
        logger.warning("Invalid data types in reading")
        return False
    
    return True

def get_system_info() -> Dict[str, Any]:
    """
    Get system information
    
    Returns:
        Dictionary of system information
    """
    import platform
    import psutil
    
    try:
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent
        }
        return system_info
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {}

def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Log file path
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}")



