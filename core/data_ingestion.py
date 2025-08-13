import json
import time
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Data class for sensor readings"""
    sensor_id: str
    sensor_type: str
    location: str
    value: float
    unit: str
    timestamp: datetime
    status: str = "normal"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class SensorSimulator:
    """Simulates IoT sensors for demonstration purposes"""
    
    def __init__(self):
        self.sensors = {
            'temp_001': {
                'type': 'temperature',
                'location': 'HVAC_room',
                'unit': 'celsius',
                'base_value': 22.0,
                'variance': 3.0,
                'trend': 0.01  # Gradual temperature increase
            },
            'hum_001': {
                'type': 'humidity',
                'location': 'HVAC_room',
                'unit': 'percentage',
                'base_value': 45.0,
                'variance': 5.0,
                'trend': 0.005
            },
            'press_001': {
                'type': 'pressure',
                'location': 'HVAC_room',
                'unit': 'kPa',
                'base_value': 101.3,
                'variance': 0.5,
                'trend': 0.0
            },
            'energy_001': {
                'type': 'energy',
                'location': 'main_panel',
                'unit': 'kWh',
                'base_value': 15.0,
                'variance': 2.0,
                'trend': 0.02
            },
            'air_001': {
                'type': 'air_quality',
                'location': 'office_area',
                'unit': 'ppm',
                'base_value': 400.0,
                'variance': 50.0,
                'trend': 0.1
            }
        }
        
        self.current_values = {}
        self.running = False
        self.callbacks = []
        
        # Initialize current values
        for sensor_id, config in self.sensors.items():
            self.current_values[sensor_id] = config['base_value']
    
    def add_callback(self, callback: Callable[[SensorReading], None]):
        """Add callback function for sensor readings"""
        self.callbacks.append(callback)
    
    def generate_reading(self, sensor_id: str) -> SensorReading:
        """Generate a realistic sensor reading"""
        config = self.sensors[sensor_id]
        
        # Add trend
        self.current_values[sensor_id] += config['trend']
        
        # Add random variance
        noise = random.gauss(0, config['variance'] * 0.1)
        value = self.current_values[sensor_id] + noise
        
        # Add time-based patterns (e.g., daily cycles)
        hour = datetime.now().hour
        if config['type'] == 'temperature':
            # Temperature varies throughout the day
            daily_variation = 2 * np.sin(2 * np.pi * hour / 24)
            value += daily_variation
        
        # Determine status based on value
        status = "normal"
        if config['type'] == 'temperature' and (value < 18 or value > 28):
            status = "warning"
        elif config['type'] == 'humidity' and (value < 30 or value > 70):
            status = "warning"
        elif config['type'] == 'air_quality' and value > 800:
            status = "critical"
        
        reading = SensorReading(
            sensor_id=sensor_id,
            sensor_type=config['type'],
            location=config['location'],
            value=round(value, 2),
            unit=config['unit'],
            timestamp=datetime.now(),
            status=status,
            metadata={
                'trend': config['trend'],
                'variance': config['variance']
            }
        )
        
        return reading
    
    def start_simulation(self, interval: float = 5.0):
        """Start the sensor simulation"""
        self.running = True
        logger.info(f"Starting sensor simulation with {interval}s interval")
        
        def simulation_loop():
            while self.running:
                for sensor_id in self.sensors.keys():
                    reading = self.generate_reading(sensor_id)
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        try:
                            callback(reading)
                        except Exception as e:
                            logger.error(f"Error in callback: {e}")
                
                time.sleep(interval)
        
        self.simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
        self.simulation_thread.start()
    
    def stop_simulation(self):
        """Stop the sensor simulation"""
        self.running = False
        logger.info("Sensor simulation stopped")

class DataProcessor:
    """Processes and analyzes sensor data"""
    
    def __init__(self):
        self.data_buffer = []
        self.max_buffer_size = 1000
        self.anomaly_thresholds = {
            'temperature': {'min': 18, 'max': 28},
            'humidity': {'min': 30, 'max': 70},
            'pressure': {'min': 100, 'max': 103},
            'energy': {'min': 5, 'max': 30},
            'air_quality': {'min': 300, 'max': 800}
        }
    
    def process_reading(self, reading: SensorReading) -> Dict[str, Any]:
        """Process a sensor reading and return analysis results"""
        # Add to buffer
        self.data_buffer.append(reading)
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)
        
        # Analyze the reading
        analysis = {
            'reading': reading.to_dict(),
            'anomaly_detected': False,
            'trend': 'stable',
            'recommendations': []
        }
        
        # Check for anomalies
        if reading.sensor_type in self.anomaly_thresholds:
            thresholds = self.anomaly_thresholds[reading.sensor_type]
            if reading.value < thresholds['min'] or reading.value > thresholds['max']:
                analysis['anomaly_detected'] = True
                analysis['recommendations'].append(
                    f"{reading.sensor_type.title()} value {reading.value} {reading.unit} is outside normal range"
                )
        
        # Analyze trends
        if len(self.data_buffer) >= 10:
            recent_readings = [r for r in self.data_buffer[-10:] if r.sensor_id == reading.sensor_id]
            if len(recent_readings) >= 5:
                values = [r.value for r in recent_readings]
                trend = np.polyfit(range(len(values)), values, 1)[0]
                
                if abs(trend) > 0.1:
                    analysis['trend'] = 'increasing' if trend > 0 else 'decreasing'
                    if abs(trend) > 0.5:
                        analysis['recommendations'].append(
                            f"Rapid {analysis['trend']} trend detected for {reading.sensor_type}"
                        )
        
        return analysis
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the collected data"""
        if not self.data_buffer:
            return {}
        
        df = pd.DataFrame([r.to_dict() for r in self.data_buffer])
        
        stats = {}
        for sensor_type in df['sensor_type'].unique():
            type_data = df[df['sensor_type'] == sensor_type]
            stats[sensor_type] = {
                'count': len(type_data),
                'mean': type_data['value'].mean(),
                'std': type_data['value'].std(),
                'min': type_data['value'].min(),
                'max': type_data['value'].max(),
                'last_reading': type_data.iloc[-1]['value'] if len(type_data) > 0 else None
            }
        
        return stats
    
    def get_recent_readings(self, sensor_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent sensor readings"""
        readings = self.data_buffer
        if sensor_id:
            readings = [r for r in readings if r.sensor_id == sensor_id]
        
        return [r.to_dict() for r in readings[-limit:]]

class IoTDataManager:
    """Main class for managing IoT data ingestion and processing"""
    
    def __init__(self):
        self.simulator = SensorSimulator()
        self.processor = DataProcessor()
        self.data_storage = []
        
        # Connect simulator to processor
        self.simulator.add_callback(self.processor.process_reading)
        self.simulator.add_callback(self._store_reading)
    
    def _store_reading(self, reading: SensorReading):
        """Store reading in memory (in production, this would go to a database)"""
        self.data_storage.append(reading.to_dict())
        
        # Keep only last 10000 readings
        if len(self.data_storage) > 10000:
            self.data_storage = self.data_storage[-10000:]
    
    def start_data_collection(self, interval: float = 5.0):
        """Start collecting sensor data"""
        self.simulator.start_simulation(interval)
        logger.info("IoT data collection started")
    
    def stop_data_collection(self):
        """Stop collecting sensor data"""
        self.simulator.stop_simulation()
        logger.info("IoT data collection stopped")
    
    def get_live_data(self) -> Dict[str, Any]:
        """Get current live data status"""
        stats = self.processor.get_statistics()
        recent_readings = self.processor.get_recent_readings(limit=20)
        
        return {
            'statistics': stats,
            'recent_readings': recent_readings,
            'total_readings': len(self.data_storage),
            'collection_active': self.simulator.running
        }
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """Get status of all sensors"""
        status = {}
        for sensor_id, config in self.simulator.sensors.items():
            current_value = self.simulator.current_values.get(sensor_id, config['base_value'])
            status[sensor_id] = {
                'type': config['type'],
                'location': config['location'],
                'current_value': round(current_value, 2),
                'unit': config['unit'],
                'status': 'active'
            }
        
        return status

