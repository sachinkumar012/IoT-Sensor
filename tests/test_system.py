import unittest
import sys
import os
import tempfile
import shutil
from datetime import datetime

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_engine import RAGEngine
from core.data_ingestion import IoTDataManager, SensorReading
from core.ml_models import MLModelManager
from core.utils import load_config, create_directory_structure, generate_sample_sensor_data

class TestRAGEngine(unittest.TestCase):
    """Test cases for RAG Engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.rag_engine = RAGEngine(persist_directory=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_rag_engine_initialization(self):
        """Test RAG engine initialization"""
        self.assertIsNotNone(self.rag_engine.client)
        self.assertIsNotNone(self.rag_engine.embedding_model)
    
    def test_text_chunking(self):
        """Test text chunking functionality"""
        test_text = "This is a test document with multiple sentences. " * 20
        chunks = self.rag_engine.chunk_text(test_text, chunk_size=100, overlap=20)
        
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 100)
    
    def test_document_addition(self):
        """Test adding documents to RAG engine"""
        test_docs = [
            {
                'id': 'test_001',
                'title': 'Test Document',
                'type': 'test',
                'content': 'This is a test document for testing purposes.',
                'source': 'test'
            }
        ]
        
        self.rag_engine.add_documents(test_docs, "manuals")
        stats = self.rag_engine.get_collection_stats()
        self.assertGreater(stats['manuals_count'], 0)
    
    def test_document_search(self):
        """Test document search functionality"""
        # Add test documents
        test_docs = [
            {
                'id': 'search_test_001',
                'title': 'Search Test Document',
                'type': 'test',
                'content': 'This document contains information about HVAC maintenance procedures.',
                'source': 'test'
            }
        ]
        
        self.rag_engine.add_documents(test_docs, "manuals")
        
        # Search for documents
        results = self.rag_engine.search("HVAC maintenance", "manuals")
        self.assertGreater(len(results), 0)
        self.assertIn('HVAC maintenance', results[0]['content'])

class TestIoTDataManager(unittest.TestCase):
    """Test cases for IoT Data Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.iot_manager = IoTDataManager()
    
    def test_sensor_simulator_initialization(self):
        """Test sensor simulator initialization"""
        self.assertIsNotNone(self.iot_manager.simulator)
        self.assertGreater(len(self.iot_manager.simulator.sensors), 0)
    
    def test_sensor_reading_generation(self):
        """Test sensor reading generation"""
        sensor_id = 'temp_001'
        reading = self.iot_manager.simulator.generate_reading(sensor_id)
        
        self.assertIsInstance(reading, SensorReading)
        self.assertEqual(reading.sensor_id, sensor_id)
        self.assertEqual(reading.sensor_type, 'temperature')
    
    def test_data_processing(self):
        """Test data processing functionality"""
        # Generate a test reading
        test_reading = SensorReading(
            sensor_id='test_001',
            sensor_type='temperature',
            location='test_room',
            value=25.0,
            unit='celsius',
            timestamp=datetime.now()
        )
        
        # Process the reading
        analysis = self.iot_manager.processor.process_reading(test_reading)
        
        self.assertIn('reading', analysis)
        self.assertIn('anomaly_detected', analysis)
        self.assertIn('recommendations', analysis)
    
    def test_data_collection_control(self):
        """Test data collection start/stop functionality"""
        # Start data collection
        self.iot_manager.start_data_collection(interval=0.1)
        self.assertTrue(self.iot_manager.simulator.running)
        
        # Stop data collection
        self.iot_manager.stop_data_collection()
        self.assertFalse(self.iot_manager.simulator.running)
    
    def test_live_data_retrieval(self):
        """Test live data retrieval"""
        # Start data collection briefly
        self.iot_manager.start_data_collection(interval=0.1)
        
        # Wait a moment for data to be collected
        import time
        time.sleep(0.5)
        
        # Stop collection
        self.iot_manager.stop_data_collection()
        
        # Get live data
        live_data = self.iot_manager.get_live_data()
        
        self.assertIn('statistics', live_data)
        self.assertIn('recent_readings', live_data)
        self.assertIn('total_readings', live_data)

class TestMLModels(unittest.TestCase):
    """Test cases for Machine Learning Models"""
    
    def setUp(self):
        """Set up test environment"""
        self.ml_manager = MLModelManager()
        self.test_data = generate_sample_sensor_data(100)
    
    def test_ml_manager_initialization(self):
        """Test ML manager initialization"""
        self.assertIsNotNone(self.ml_manager.maintenance_model)
        self.assertIsNotNone(self.ml_manager.anomaly_model)
        self.assertIsNotNone(self.ml_manager.energy_model)
    
    def test_model_training(self):
        """Test model training functionality"""
        if len(self.test_data) >= 50:
            training_results = self.ml_manager.train_all_models(self.test_data)
            
            # Check that training results are returned
            self.assertIn('maintenance', training_results)
            self.assertIn('anomaly', training_results)
            self.assertIn('energy', training_results)
    
    def test_predictions_after_training(self):
        """Test predictions after model training"""
        if len(self.test_data) >= 50:
            # Train models first
            self.ml_manager.train_all_models(self.test_data)
            
            # Get predictions
            predictions = self.ml_manager.get_predictions(self.test_data)
            
            # Check that predictions are available
            if self.ml_manager.models_trained['maintenance']:
                self.assertIn('maintenance', predictions)
            
            if self.ml_manager.models_trained['anomaly']:
                self.assertIn('anomaly', predictions)
            
            if self.ml_manager.models_trained['energy']:
                self.assertIn('energy', predictions)

class TestUtils(unittest.TestCase):
    """Test cases for Utility Functions"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = load_config()
        self.assertIsInstance(config, dict)
        self.assertIn('system', config)
    
    def test_directory_structure_creation(self):
        """Test directory structure creation"""
        test_dir = tempfile.mkdtemp()
        try:
            create_directory_structure(test_dir)
            
            # Check that directories were created
            expected_dirs = ['data/sensors', 'data/manuals', 'data/specs', 'models']
            for dir_path in expected_dirs:
                full_path = os.path.join(test_dir, dir_path)
                self.assertTrue(os.path.exists(full_path))
        finally:
            shutil.rmtree(test_dir)
    
    def test_sample_data_generation(self):
        """Test sample sensor data generation"""
        sample_data = generate_sample_sensor_data(50)
        
        self.assertEqual(len(sample_data), 250)  # 5 sensors * 50 readings
        
        # Check data structure
        for reading in sample_data:
            self.assertIn('sensor_id', reading)
            self.assertIn('sensor_type', reading)
            self.assertIn('value', reading)
            self.assertIn('timestamp', reading)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.rag_engine = RAGEngine(persist_directory=self.test_dir)
        self.iot_manager = IoTDataManager()
        self.ml_manager = MLModelManager()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Load sample documents into RAG
        sample_docs = [
            {
                'id': 'integration_test_001',
                'title': 'Integration Test Document',
                'type': 'maintenance',
                'content': 'This document contains HVAC maintenance procedures and troubleshooting guides.',
                'source': 'integration_test'
            }
        ]
        
        self.rag_engine.add_documents(sample_docs, "manuals")
        
        # 2. Generate and collect sensor data
        self.iot_manager.start_data_collection(interval=0.1)
        import time
        time.sleep(0.5)
        self.iot_manager.stop_data_collection()
        
        # 3. Train ML models
        live_data = self.iot_manager.get_live_data()
        sensor_data = live_data.get('recent_readings', [])
        
        if len(sensor_data) >= 50:
            training_results = self.ml_manager.train_all_models(sensor_data)
            
            # 4. Get predictions
            predictions = self.ml_manager.get_predictions(sensor_data)
            
            # 5. Search documents using RAG
            search_results = self.rag_engine.search("HVAC maintenance", "manuals")
            
            # Verify all components are working
            self.assertGreater(len(search_results), 0)
            self.assertIn('maintenance', predictions)
    
    def test_system_performance(self):
        """Test system performance with larger datasets"""
        # Generate larger dataset
        large_dataset = generate_sample_sensor_data(500)
        
        # Test RAG performance
        start_time = datetime.now()
        self.rag_engine.add_documents([
            {
                'id': f'perf_test_{i}',
                'title': f'Performance Test Document {i}',
                'type': 'test',
                'content': f'This is performance test document number {i} with some content.',
                'source': 'performance_test'
            } for i in range(100)
        ], "manuals")
        rag_time = (datetime.now() - start_time).total_seconds()
        
        # Test ML training performance
        if len(large_dataset) >= 50:
            start_time = datetime.now()
            self.ml_manager.train_all_models(large_dataset)
            ml_time = (datetime.now() - start_time).total_seconds()
            
            # Performance should be reasonable (adjust thresholds as needed)
            self.assertLess(rag_time, 30)  # RAG should complete in under 30 seconds
            self.assertLess(ml_time, 60)   # ML training should complete in under 60 seconds

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRAGEngine,
        TestIoTDataManager,
        TestMLModels,
        TestUtils,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Running IoT Sensor Data RAG System Tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed! The system is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)





