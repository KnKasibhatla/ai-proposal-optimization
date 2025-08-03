"""
Test Suite for AI Proposal Optimization Platform
Comprehensive testing for all components
"""

import unittest
import sys
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

import sys
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import application modules with error handling
try:
    from backend.app import app, db
    from models.proposal_predictor import ProposalPredictor
    from models.competitive_analyzer import CompetitiveAnalyzer
    from models.reinforcement_agent import ReinforcementAgent
    from utils.data_processor import DataProcessor
    from utils.feature_engineer import FeatureEngineer
    from config.config import TestingConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False
    # Create mock classes for testing
    app = Mock()
    db = Mock()
    ProposalPredictor = Mock
    CompetitiveAnalyzer = Mock
    ReinforcementAgent = Mock
    DataProcessor = Mock
    FeatureEngineer = Mock
    TestingConfig = Mock

class TestFlaskApp(unittest.TestCase):
    """Test Flask application endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        app.config.from_object(TestingConfig)
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        
        with app.app_context():
            db.create_all()
    
    def tearDown(self):
        """Clean up test environment"""
        with app.app_context():
            db.session.remove()
            db.drop_all()
        self.app_context.pop()
    
    def test_home_page(self):
        """Test home page loads"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI Proposal Optimization', response.data)
    
    def test_predict_page_get(self):
        """Test prediction page loads"""
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction', response.data)
    
    def test_analyze_page(self):
        """Test analysis page loads"""
        response = self.app.get('/analyze')
        self.assertEqual(response.status_code, 200)
    
    def test_dashboard_data_api(self):
        """Test dashboard data API"""
        response = self.app.get('/api/dashboard-data')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('recent_predictions', data)
        self.assertIn('performance_metrics', data)
    
    @patch('backend.app.proposal_predictor')
    def test_prediction_api(self, mock_predictor):
        """Test prediction API endpoint"""
        mock_predictor.predict.return_value = {
            'price': 50000.0,
            'win_probability': 0.75,
            'confidence_interval': {'lower': 45000, 'upper': 55000},
            'feature_importance': {}
        }
        
        test_data = {
            'quality_score': 8,
            'delivery_time': 30,
            'complexity': 6,
            'project_category': 'software_development',
            'num_bids': 5
        }
        
        response = self.app.post('/predict',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('predicted_price', data)
        self.assertIn('win_probability', data)
    
    def test_prediction_api_missing_data(self):
        """Test prediction API with missing required data"""
        test_data = {
            'quality_score': 8
            # Missing required fields
        }
        
        response = self.app.post('/predict',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 500)
    
    def test_competitive_analysis_api(self):
        """Test competitive analysis API"""
        test_data = {
            'features': {
                'quality_score': 8,
                'delivery_time': 30,
                'num_bids': 5
            },
            'competitors': [
                {'provider_id': 'comp1', 'quality_score': 7, 'win_rate': 0.6},
                {'provider_id': 'comp2', 'quality_score': 6, 'win_rate': 0.4}
            ]
        }
        
        response = self.app.post('/api/competitive-analysis',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)

class TestProposalPredictor(unittest.TestCase):
    """Test ProposalPredictor class"""
    
    def setUp(self):
        """Set up test environment"""
        self.predictor = ProposalPredictor()
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        
        data = []
        for i in range(100):
            record = {
                'bid_id': f'bid_{i:03d}',
                'price': np.random.normal(50000, 15000),
                'provider_id': f'provider_{i % 10}',
                'win_loss': np.random.choice(['win', 'loss']),
                'quality_score': np.random.randint(1, 11),
                'delivery_time': np.random.randint(10, 90),
                'complexity': np.random.randint(1, 11),
                'project_category': np.random.choice(['software', 'consulting', 'manufacturing']),
                'winning_price': np.random.normal(45000, 12000),
                'num_bids': np.random.randint(2, 10),
                'date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test predictor initialization"""
        self.assertIsInstance(self.predictor.models, dict)
        self.assertIn('random_forest', self.predictor.models)
        self.assertFalse(self.predictor.is_trained)
    
    def test_preprocess_features(self):
        """Test feature preprocessing"""
        processed_data = self.predictor.preprocess_features(self.sample_data)
        
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertGreater(len(processed_data.columns), len(self.sample_data.columns))
    
    def test_training(self):
        """Test model training"""
        # Add winning_price for training
        self.sample_data['winning_price'] = self.sample_data['price'] * np.random.uniform(0.8, 1.2, len(self.sample_data))
        
        results = self.predictor.train(self.sample_data)
        
        self.assertIsInstance(results, dict)
        self.assertTrue(self.predictor.is_trained)
        self.assertIn('random_forest', results)
    
    @patch.object(ProposalPredictor, 'is_trained', True)
    @patch.object(ProposalPredictor, 'feature_columns', ['quality_score', 'delivery_time'])
    def test_prediction(self):
        """Test making predictions"""
        # Mock the trained state
        self.predictor.feature_columns = ['quality_score', 'delivery_time']
        self.predictor.ensemble_weights = {'random_forest': 1.0}
        
        # Mock model predictions
        mock_model = Mock()
        mock_model.predict.return_value = [50000]
        self.predictor.models['random_forest'] = mock_model
        
        test_input = {
            'quality_score': 8,
            'delivery_time': 30,
            'complexity': 6,
            'project_category': 'software_development'
        }
        
        result = self.predictor.predict(test_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn('price', result)
        self.assertIn('win_probability', result)
        self.assertIn('confidence_interval', result)
    
    def test_get_model_info(self):
        """Test getting model information"""
        info = self.predictor.get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('is_trained', info)
        self.assertIn('models', info)

class TestCompetitiveAnalyzer(unittest.TestCase):
    """Test CompetitiveAnalyzer class"""
    
    def setUp(self):
        """Set up test environment"""
        self.analyzer = CompetitiveAnalyzer()
        self.sample_features = {
            'quality_score': 8,
            'delivery_time': 30,
            'num_bids': 5,
            'market_share': 0.15
        }
        self.sample_competitors = [
            {'provider_id': 'comp1', 'quality_score': 7, 'win_rate': 0.6, 'market_share': 0.2},
            {'provider_id': 'comp2', 'quality_score': 6, 'win_rate': 0.4, 'market_share': 0.1}
        ]
    
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer.provider_profiles, dict)
        self.assertIsInstance(self.analyzer.market_segments, dict)
    
    def test_analyze_competition(self):
        """Test competitive analysis"""
        result = self.analyzer.analyze_competition(self.sample_features, self.sample_competitors)
        
        self.assertIsInstance(result, dict)
        self.assertIn('competitor_analysis', result)
        self.assertIn('market_position', result)
        self.assertIn('competitive_intensity', result)
        self.assertIn('strategic_recommendations', result)
    
    def test_nash_equilibrium_two_players(self):
        """Test Nash equilibrium calculation for two players"""
        game_data = {
            'players': ['player1', 'player2'],
            'payoff_matrix': [[10, 5], [6, 8]]
        }
        
        result = self.analyzer.find_nash_equilibrium(game_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('equilibrium_type', result)
    
    def test_full_analysis(self):
        """Test comprehensive competitive analysis"""
        data = {
            'features': self.sample_features,
            'competitors': self.sample_competitors
        }
        
        result = self.analyzer.full_analysis(data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('basic_analysis', result)
        self.assertIn('game_theory', result)
        self.assertIn('market_structure', result)

class TestReinforcementAgent(unittest.TestCase):
    """Test ReinforcementAgent class"""
    
    def setUp(self):
        """Set up test environment"""
        self.agent = ReinforcementAgent()
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample RL training data"""
        data = []
        for i in range(50):
            record = {
                'quality_score': np.random.randint(1, 11),
                'delivery_time': np.random.randint(10, 90),
                'num_bids': np.random.randint(2, 10),
                'complexity': np.random.randint(1, 11),
                'win_loss': np.random.choice(['win', 'loss']),
                'price': np.random.normal(50000, 15000)
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test RL agent initialization"""
        self.assertIsNone(self.agent.environment)
        self.assertIsNone(self.agent.agent)
        self.assertFalse(self.agent.is_trained)
    
    def test_initialize_with_data(self):
        """Test initialization with historical data"""
        self.agent.initialize_with_data(self.sample_data)
        
        self.assertIsNotNone(self.agent.environment)
    
    def test_get_recommendation_untrained(self):
        """Test getting recommendation from untrained agent"""
        features = {
            'quality_score': 8,
            'delivery_time': 30,
            'complexity': 6
        }
        
        recommendation = self.agent.get_recommendation(features)
        
        self.assertIsInstance(recommendation, dict)
        self.assertIn('action', recommendation)
        self.assertIn('price_multiplier', recommendation)
    
    def test_strategy_optimization(self):
        """Test strategy optimization"""
        data = {
            'features': {
                'quality_score': 8,
                'delivery_time': 30,
                'complexity': 6
            }
        }
        
        result = self.agent.optimize_strategy(data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('rl_recommendation', result)

class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor class"""
    
    def setUp(self):
        """Set up test environment"""
        self.processor = DataProcessor()
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample data for processing"""
        data = {
            'bid_id': ['bid_001', 'bid_002', 'bid_003'],
            'price': [50000, 45000, 55000],
            'provider_id': ['prov_1', 'prov_2', 'prov_1'],
            'win_loss': ['win', 'loss', 'win'],
            'quality_score': [8, 7, 9],
            'delivery_time': [30, 45, 25],
            'complexity': [6, 5, 7],
            'project_category': ['software', 'consulting', 'software'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'winning_price': [50000, 42000, 55000],
            'num_bids': [5, 3, 6]
        }
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test data processor initialization"""
        self.assertIsInstance(self.processor.data_schema, dict)
        self.assertIn('bid_id', self.processor.data_schema)
    
    def test_process_raw_data(self):
        """Test raw data processing"""
        processed_data = self.processor.process_raw_data(self.sample_data)
        
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertGreater(len(processed_data.columns), len(self.sample_data.columns))
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        sample_data = self.processor.generate_sample_data(n_samples=100)
        
        self.assertIsInstance(sample_data, pd.DataFrame)
        self.assertEqual(len(sample_data), 100)
        self.assertIn('bid_id', sample_data.columns)
        self.assertIn('price', sample_data.columns)
    
    def test_validate_prediction_input(self):
        """Test prediction input validation"""
        valid_input = {
            'quality_score': 8,
            'delivery_time': 30,
            'complexity': 6,
            'project_category': 'software'
        }
        
        is_valid, errors = self.processor.validate_prediction_input(valid_input)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid input
        invalid_input = {
            'quality_score': 15,  # Out of range
            'delivery_time': -5   # Negative
        }
        
        is_valid, errors = self.processor.validate_prediction_input(invalid_input)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_get_data_summary(self):
        """Test data summary generation"""
        summary = self.processor.get_data_summary(self.sample_data)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('basic_stats', summary)
        self.assertIn('win_loss_distribution', summary)
        self.assertIn('price_stats', summary)
    
    def test_detect_data_drift(self):
        """Test data drift detection"""
        # Create reference and new data
        reference_data = self.sample_data.copy()
        new_data = self.sample_data.copy()
        new_data['price'] = new_data['price'] * 1.5  # Simulate price drift
        
        drift_results = self.processor.detect_data_drift(reference_data, new_data)
        
        self.assertIsInstance(drift_results, dict)
        self.assertIn('overall_drift_detected', drift_results)
        self.assertIn('drift_score', drift_results)

class TestFeatureEngineer(unittest.TestCase):
    """Test FeatureEngineer class"""
    
    def setUp(self):
        """Set up test environment"""
        self.engineer = FeatureEngineer()
        self.sample_input = {
            'quality_score': 8,
            'delivery_time': 30,
            'complexity': 6,
            'project_category': 'software_development',
            'num_bids': 5,
            'provider_experience': 15,
            'market_share': 0.15
        }
    
    def test_initialization(self):
        """Test feature engineer initialization"""
        self.assertIsInstance(self.engineer.scalers, dict)
        self.assertIsInstance(self.engineer.encoders, dict)
    
    def test_create_features(self):
        """Test feature creation"""
        features = self.engineer.create_features(self.sample_input)
        
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), len(self.sample_input))
        
        # Check for specific engineered features
        self.assertIn('quality_normalized', features)
        self.assertIn('competitive_intensity', features)
        self.assertIn('delivery_urgency', features)
    
    def test_validate_features(self):
        """Test feature validation"""
        features = self.engineer.create_features(self.sample_input)
        is_valid, errors = self.engineer.validate_features(features)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid features
        invalid_features = features.copy()
        invalid_features['invalid_feature'] = float('inf')
        
        is_valid, errors = self.engineer.validate_features(invalid_features)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.predictor = ProposalPredictor()
        self.analyzer = CompetitiveAnalyzer()
        self.rl_agent = ReinforcementAgent()
    
    def test_end_to_end_prediction_workflow(self):
        """Test complete prediction workflow"""
        # Generate sample data
        raw_data = self.data_processor.generate_sample_data(n_samples=200)
        
        # Process data
        processed_data = self.data_processor.process_raw_data(raw_data)
        
        # Train models
        training_data = processed_data.copy()
        training_data['winning_price'] = training_data['price'] * np.random.uniform(0.8, 1.2, len(training_data))
        
        # Train predictor
        self.predictor.train(training_data)
        
        # Initialize RL agent
        self.rl_agent.initialize_with_data(training_data)
        
        # Make prediction
        test_input = {
            'quality_score': 8,
            'delivery_time': 30,
            'complexity': 6,
            'project_category': 'software_development',
            'num_bids': 5
        }
        
        # Engineer features
        features = self.feature_engineer.create_features(test_input)
        
        # Get prediction
        prediction = self.predictor.predict(test_input)
        
        # Get competitive analysis
        competitors = [
            {'provider_id': 'comp1', 'quality_score': 7, 'win_rate': 0.6},
            {'provider_id': 'comp2', 'quality_score': 6, 'win_rate': 0.4}
        ]
        
        competitive_analysis = self.analyzer.analyze_competition(features, competitors)
        
        # Get RL recommendation
        rl_recommendation = self.rl_agent.get_recommendation(features)
        
        # Verify results
        self.assertIsInstance(prediction, dict)
        self.assertIn('price', prediction)
        self.assertIn('win_probability', prediction)
        
        self.assertIsInstance(competitive_analysis, dict)
        self.assertIn('competitor_analysis', competitive_analysis)
        
        self.assertIsInstance(rl_recommendation, dict)
        self.assertIn('price_multiplier', rl_recommendation)
    
    def test_data_flow_consistency(self):
        """Test data consistency through the pipeline"""
        # Generate and process data
        raw_data = self.data_processor.generate_sample_data(n_samples=100)
        processed_data = self.data_processor.process_raw_data(raw_data)
        
        # Check data integrity
        self.assertEqual(len(raw_data), len(processed_data))
        self.assertIn('bid_id', processed_data.columns)
        self.assertIn('price', processed_data.columns)
        
        # Check for engineered features
        self.assertIn('provider_win_rate', processed_data.columns)
        self.assertIn('competitive_intensity', processed_data.columns)
    
    def test_model_compatibility(self):
        """Test compatibility between different models"""
        # Generate training data
        training_data = self.data_processor.generate_sample_data(n_samples=150)
        training_data['winning_price'] = training_data['price'] * np.random.uniform(0.9, 1.1, len(training_data))
        
        # Train models
        self.predictor.train(training_data)
        self.rl_agent.initialize_with_data(training_data)
        
        # Test feature compatibility
        test_input = {
            'quality_score': 8,
            'delivery_time': 30,
            'complexity': 6,
            'project_category': 'software_development'
        }
        
        features = self.feature_engineer.create_features(test_input)
        
        # All models should accept the same feature format
        prediction = self.predictor.predict(test_input)
        rl_rec = self.rl_agent.get_recommendation(features)
        
        self.assertIsInstance(prediction, dict)
        self.assertIsInstance(rl_rec, dict)

class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.data_processor = DataProcessor()
        self.predictor = ProposalPredictor()
    
    def test_prediction_speed(self):
        """Test prediction response time"""
        import time
        
        # Generate training data
        training_data = self.data_processor.generate_sample_data(n_samples=1000)
        training_data['winning_price'] = training_data['price'] * np.random.uniform(0.9, 1.1, len(training_data))
        
        # Train model
        self.predictor.train(training_data)
        
        # Test prediction speed
        test_input = {
            'quality_score': 8,
            'delivery_time': 30,
            'complexity': 6,
            'project_category': 'software_development'
        }
        
        start_time = time.time()
        prediction = self.predictor.predict(test_input)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        
        # Prediction should complete within 5 seconds
        self.assertLess(prediction_time, 5.0)
        self.assertIsInstance(prediction, dict)
    
    def test_memory_usage(self):
        """Test memory usage during training"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large dataset
        large_data = self.data_processor.generate_sample_data(n_samples=5000)
        large_data['winning_price'] = large_data['price'] * np.random.uniform(0.9, 1.1, len(large_data))
        
        # Train model
        self.predictor.train(large_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB)
        self.assertLess(memory_increase, 500)

def create_test_suite():
    """Create comprehensive test suite"""
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestFlaskApp))
    suite.addTest(unittest.makeSuite(TestProposalPredictor))
    suite.addTest(unittest.makeSuite(TestCompetitiveAnalyzer))
    suite.addTest(unittest.makeSuite(TestReinforcementAgent))
    suite.addTest(unittest.makeSuite(TestDataProcessor))
    suite.addTest(unittest.makeSuite(TestFeatureEngineer))
    suite.addTest(unittest.makeSuite(TestIntegration))
    suite.addTest(unittest.makeSuite(TestPerformance))
    
    return suite

def run_tests():
    """Run all tests with detailed output"""
    
    # Create test suite
    suite = create_test_suite()
    
    # Create test runner
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # Run tests
    print("=" * 70)
    print("AI PROPOSAL OPTIMIZATION PLATFORM - TEST SUITE")
    print("=" * 70)
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)