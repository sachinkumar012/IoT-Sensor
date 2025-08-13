import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Any, Tuple, Optional
import joblib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictiveMaintenanceModel:
    """Predictive maintenance model for equipment failure prediction"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def prepare_features(self, sensor_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features from sensor data for training/prediction"""
        if not sensor_data:
            return np.array([]), np.array([])
        
        df = pd.DataFrame(sensor_data)
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Create lag features for time series
        for sensor_type in ['temperature', 'humidity', 'pressure', 'energy']:
            if sensor_type in df['sensor_type'].values:
                type_data = df[df['sensor_type'] == sensor_type].sort_values('timestamp')
                if len(type_data) > 0:
                    # Create lag features
                    for lag in [1, 2, 3]:
                        lag_values = type_data['value'].shift(lag)
                        df[f'{sensor_type}_lag_{lag}'] = lag_values
                    
                    # Create rolling statistics
                    df[f'{sensor_type}_rolling_mean'] = type_data['value'].rolling(window=5, min_periods=1).mean()
                    df[f'{sensor_type}_rolling_std'] = type_data['value'].rolling(window=5, min_periods=1).std()
        
        # Create interaction features
        if 'temperature' in df['sensor_type'].values and 'humidity' in df['sensor_type'].values:
            temp_data = df[df['sensor_type'] == 'temperature']['value'].values
            hum_data = df[df['sensor_type'] == 'humidity']['value'].values
            
            if len(temp_data) > 0 and len(hum_data) > 0:
                min_len = min(len(temp_data), len(hum_data))
                df['temp_humidity_interaction'] = temp_data[:min_len] * hum_data[:min_len]
        
        # Drop non-numeric columns and handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns].fillna(0)
        
        # Store feature names
        self.feature_names = df_numeric.columns.tolist()
        
        # Create target variable (simulate maintenance needs based on sensor values)
        # In a real scenario, this would be based on actual maintenance records
        target = self._create_maintenance_target(df_numeric)
        
        return df_numeric.values, target
    
    def _create_maintenance_target(self, df_numeric: pd.DataFrame) -> np.ndarray:
        """Create synthetic maintenance target for demonstration"""
        # Simulate maintenance needs based on sensor values
        target = np.zeros(len(df_numeric))
        
        # High temperature + high humidity = higher maintenance probability
        if 'temperature_rolling_mean' in df_numeric.columns and 'humidity_rolling_mean' in df_numeric.columns:
            temp_mean = df_numeric['temperature_rolling_mean'].values
            hum_mean = df_numeric['humidity_rolling_mean'].values
            
            # Normalize values
            temp_norm = (temp_mean - np.mean(temp_mean)) / np.std(temp_mean)
            hum_norm = (hum_mean - np.mean(hum_mean)) / np.std(hum_mean)
            
            # Combined risk score
            risk_score = temp_norm + hum_norm
            
            # Convert to maintenance probability (0-1)
            target = 1 / (1 + np.exp(-risk_score))
        
        return target
    
    def train(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train the predictive maintenance model"""
        X, y = self.prepare_features(sensor_data)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No data available for training")
            return {}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Make predictions on training data
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        logger.info(f"Model trained successfully. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'training_samples': len(X)
        }
    
    def predict_maintenance(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict maintenance needs"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        X, _ = self.prepare_features(sensor_data)
        
        if len(X) == 0:
            return {'error': 'No data available for prediction'}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        maintenance_prob = self.model.predict(X_scaled)
        
        # Get feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return {
            'maintenance_probability': float(maintenance_prob[-1]),
            'risk_level': self._get_risk_level(maintenance_prob[-1]),
            'feature_importance': feature_importance,
            'recommendations': self._get_maintenance_recommendations(maintenance_prob[-1])
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        else:
            return "High"
    
    def _get_maintenance_recommendations(self, probability: float) -> List[str]:
        """Generate maintenance recommendations based on probability"""
        recommendations = []
        
        if probability > 0.7:
            recommendations.extend([
                "Schedule immediate maintenance inspection",
                "Check HVAC system performance",
                "Monitor sensor readings closely"
            ])
        elif probability > 0.5:
            recommendations.extend([
                "Plan maintenance within 1-2 weeks",
                "Review system performance metrics",
                "Check for equipment wear and tear"
            ])
        elif probability > 0.3:
            recommendations.extend([
                "Schedule routine maintenance",
                "Monitor trends for changes",
                "Update maintenance schedule if needed"
            ])
        else:
            recommendations.append("No immediate maintenance required")
        
        return recommendations

class AnomalyDetectionModel:
    """Anomaly detection using Isolation Forest"""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, sensor_data: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare features for anomaly detection"""
        if not sensor_data:
            return np.array([])
        
        df = pd.DataFrame(sensor_data)
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Create sensor-specific features
        features = []
        for sensor_type in ['temperature', 'humidity', 'pressure', 'energy', 'air_quality']:
            if sensor_type in df['sensor_type'].values:
                type_data = df[df['sensor_type'] == sensor_type]
                if len(type_data) > 0:
                    # Current value
                    features.append(type_data['value'].iloc[-1])
                    
                    # Rolling statistics
                    rolling_mean = type_data['value'].rolling(window=10, min_periods=1).mean().iloc[-1]
                    rolling_std = type_data['value'].rolling(window=10, min_periods=1).std().iloc[-1]
                    features.extend([rolling_mean, rolling_std])
                    
                    # Rate of change
                    if len(type_data) > 1:
                        rate_of_change = (type_data['value'].iloc[-1] - type_data['value'].iloc[-2])
                        features.append(rate_of_change)
                    else:
                        features.append(0)
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0])
        
        # Add time features
        features.extend([df['hour'].iloc[-1], df['day_of_week'].iloc[-1]])
        
        return np.array(features).reshape(1, -1)
    
    def train(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the anomaly detection model"""
        if len(sensor_data) < 50:
            logger.warning("Insufficient data for training anomaly detection model")
            return {'error': 'Insufficient data'}
        
        # Prepare features for multiple data points
        features_list = []
        for i in range(len(sensor_data)):
            features = self.prepare_features(sensor_data[:i+1])
            if len(features) > 0:
                features_list.append(features.flatten())
        
        if len(features_list) < 10:
            return {'error': 'Insufficient features for training'}
        
        X = np.array(features_list)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        logger.info(f"Anomaly detection model trained with {len(X)} samples")
        
        return {
            'training_samples': len(X),
            'contamination': self.model.contamination
        }
    
    def detect_anomalies(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalies in sensor data"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        features = self.prepare_features(sensor_data)
        
        if len(features) == 0:
            return {'error': 'No data available for anomaly detection'}
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict (1 for normal, -1 for anomaly)
        prediction = self.model.predict(features_scaled)[0]
        
        # Get anomaly score (lower = more anomalous)
        anomaly_score = self.model.score_samples(features_scaled)[0]
        
        is_anomaly = prediction == -1
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'severity': self._get_anomaly_severity(anomaly_score),
            'confidence': self._get_confidence_score(anomaly_score)
        }
    
    def _get_anomaly_severity(self, score: float) -> str:
        """Convert anomaly score to severity level"""
        if score > -0.5:
            return "Low"
        elif score > -1.0:
            return "Medium"
        else:
            return "High"
    
    def _get_confidence_score(self, score: float) -> float:
        """Convert anomaly score to confidence (0-1)"""
        # Normalize score to 0-1 range
        normalized_score = (score + 2) / 4  # Assuming score range is roughly -2 to 2
        return max(0, min(1, normalized_score))

class EnergyOptimizationModel:
    """Energy consumption optimization model"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, sensor_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for energy optimization"""
        if not sensor_data:
            return np.array([]), np.array([])
        
        df = pd.DataFrame(sensor_data)
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Focus on energy-related data
        energy_data = df[df['sensor_type'] == 'energy']
        temp_data = df[df['sensor_type'] == 'temperature']
        hum_data = df[df['sensor_type'] == 'humidity']
        
        if len(energy_data) == 0:
            return np.array([]), np.array([])
        
        # Create time-based features
        energy_data['hour'] = energy_data['timestamp'].dt.hour
        energy_data['day_of_week'] = energy_data['timestamp'].dt.dayofweek
        energy_data['month'] = energy_data['timestamp'].dt.month
        
        # Create features
        features = []
        targets = []
        
        for idx, row in energy_data.iterrows():
            feature_vector = [
                row['hour'],
                row['day_of_week'],
                row['month']
            ]
            
            # Add environmental factors if available
            if len(temp_data) > 0:
                # Find closest temperature reading
                temp_idx = (temp_data['timestamp'] - row['timestamp']).abs().idxmin()
                feature_vector.append(temp_data.loc[temp_idx, 'value'])
            else:
                feature_vector.append(22.0)  # Default temperature
            
            if len(hum_data) > 0:
                # Find closest humidity reading
                hum_idx = (hum_data['timestamp'] - row['timestamp']).abs().idxmin()
                feature_vector.append(hum_data.loc[hum_idx, 'value'])
            else:
                feature_vector.append(45.0)  # Default humidity
            
            features.append(feature_vector)
            targets.append(row['value'])
        
        return np.array(features), np.array(targets)
    
    def train(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train the energy optimization model"""
        X, y = self.prepare_features(sensor_data)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No energy data available for training")
            return {}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Make predictions on training data
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        logger.info(f"Energy optimization model trained. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'training_samples': len(X)
        }
    
    def predict_energy_consumption(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Predict energy consumption under given conditions"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        # Prepare feature vector
        features = [
            conditions.get('hour', 12),
            conditions.get('day_of_week', 0),
            conditions.get('month', 1),
            conditions.get('temperature', 22.0),
            conditions.get('humidity', 45.0)
        ]
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        predicted_consumption = self.model.predict(X_scaled)[0]
        
        return {
            'predicted_consumption': float(predicted_consumption),
            'conditions': conditions,
            'optimization_suggestions': self._get_optimization_suggestions(conditions, predicted_consumption)
        }
    
    def _get_optimization_suggestions(self, conditions: Dict[str, Any], consumption: float) -> List[str]:
        """Generate energy optimization suggestions"""
        suggestions = []
        
        # Temperature-based suggestions
        temp = conditions.get('temperature', 22.0)
        if temp < 20 or temp > 26:
            suggestions.append("Maintain temperature between 20-26°C for optimal energy efficiency")
        
        # Time-based suggestions
        hour = conditions.get('hour', 12)
        if hour < 6 or hour > 22:
            suggestions.append("Consider reducing energy consumption during off-hours")
        
        # General suggestions
        if consumption > 20:
            suggestions.append("High energy consumption detected - review HVAC settings")
        
        if not suggestions:
            suggestions.append("Current conditions are optimal for energy efficiency")
        
        return suggestions

class MLModelManager:
    """Main class for managing all ML models"""
    
    def __init__(self):
        self.maintenance_model = PredictiveMaintenanceModel()
        self.anomaly_model = AnomalyDetectionModel()
        self.energy_model = EnergyOptimizationModel()
        
        self.models_trained = {
            'maintenance': False,
            'anomaly': False,
            'energy': False
        }
    
    def train_all_models(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train all ML models"""
        results = {}
        
        # Train maintenance model
        try:
            maintenance_result = self.maintenance_model.train(sensor_data)
            if 'error' not in maintenance_result:
                self.models_trained['maintenance'] = True
            results['maintenance'] = maintenance_result
        except Exception as e:
            results['maintenance'] = {'error': str(e)}
        
        # Train anomaly detection model
        try:
            anomaly_result = self.anomaly_model.train(sensor_data)
            if 'error' not in anomaly_result:
                self.models_trained['anomaly'] = True
            results['anomaly'] = anomaly_result
        except Exception as e:
            results['anomaly'] = {'error': str(e)}
        
        # Train energy optimization model
        try:
            energy_result = self.energy_model.train(sensor_data)
            if 'error' not in energy_result:
                self.models_trained['energy'] = True
            results['energy'] = energy_result
        except Exception as e:
            results['energy'] = {'error': str(e)}
        
        logger.info(f"Model training completed. Results: {results}")
        return results
    
    def get_predictions(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get predictions from all trained models"""
        predictions = {}
        
        if self.models_trained['maintenance']:
            predictions['maintenance'] = self.maintenance_model.predict_maintenance(sensor_data)
        
        if self.models_trained['anomaly']:
            predictions['anomaly'] = self.anomaly_model.detect_anomalies(sensor_data)
        
        if self.models_trained['energy']:
            # Get current conditions for energy prediction
            current_conditions = self._extract_current_conditions(sensor_data)
            predictions['energy'] = self.energy_model.predict_energy_consumption(current_conditions)
        
        return predictions
    
    def _extract_current_conditions(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract current conditions from sensor data"""
        if not sensor_data:
            return {}
        
        # Get latest timestamp
        latest_time = max(sensor_data, key=lambda x: x['timestamp'])
        dt = pd.to_datetime(latest_time['timestamp'])
        
        conditions = {
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'month': dt.month
        }
        
        # Get latest sensor readings
        for sensor_type in ['temperature', 'humidity']:
            type_data = [s for s in sensor_data if s['sensor_type'] == sensor_type]
            if type_data:
                latest = max(type_data, key=lambda x: x['timestamp'])
                conditions[sensor_type] = latest['value']
        
        return conditions
    
    def save_models(self, directory: str = "./models"):
        """Save trained models to disk"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        if self.models_trained['maintenance']:
            joblib.dump(self.maintenance_model, f"{directory}/maintenance_model.pkl")
        
        if self.models_trained['anomaly']:
            joblib.dump(self.anomaly_model, f"{directory}/anomaly_model.pkl")
        
        if self.models_trained['energy']:
            joblib.dump(self.energy_model, f"{directory}/energy_model.pkl")
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str = "./models"):
        """Load trained models from disk"""
        try:
            if os.path.exists(f"{directory}/maintenance_model.pkl"):
                self.maintenance_model = joblib.load(f"{directory}/maintenance_model.pkl")
                self.models_trained['maintenance'] = True
            
            if os.path.exists(f"{directory}/anomaly_model.pkl"):
                self.anomaly_model = joblib.load(f"{directory}/anomaly_model.pkl")
                self.models_trained['anomaly'] = True
            
            if os.path.exists(f"{directory}/energy_model.pkl"):
                self.energy_model = joblib.load(f"{directory}/energy_model.pkl")
                self.models_trained['energy'] = True
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

