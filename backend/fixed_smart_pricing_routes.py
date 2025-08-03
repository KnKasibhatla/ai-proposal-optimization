#!/usr/bin/env python3
"""
Fixed Smart Pricing Routes - Ensures Dynamic Predictions
"""

from flask import jsonify, request
import pandas as pd
from fixed_smart_pricing import fixed_smart_pricing_engine

def add_fixed_smart_pricing_routes(app):
    """Add fixed smart pricing routes to Flask app"""
    
    @app.route('/api/predict-pricing-fixed', methods=['POST'])
    def api_predict_pricing_fixed():
        """Fixed pricing prediction with guaranteed dynamic results"""
        try:
            # Get current data from the global variable
            from app import current_data
            
            if current_data is None or len(current_data) == 0:
                return jsonify({'error': 'No historical data available. Please upload data first.'}), 400
            
            # Train model if not already trained
            if not fixed_smart_pricing_engine.trained:
                print("🔄 Training fixed smart pricing model...")
                training_result = fixed_smart_pricing_engine.train_model(current_data)
                if training_result['status'] != 'success':
                    return jsonify({'error': f'Model training failed: {training_result.get("message", "Unknown error")}'}), 500
                print(f"✅ Model trained with {training_result['training_samples']} samples")
            
            # Get request data
            data = request.get_json()
            client_id = data.get('client_id')
            base_amount = float(data.get('base_amount', 0))
            industry = data.get('industry', '')
            project_type = data.get('project_type', '')
            
            if not client_id or base_amount <= 0:
                return jsonify({'error': 'Client ID and base amount are required'}), 400
            
            print(f"🎯 Processing prediction request: Client={client_id}, Amount=${base_amount:,.2f}")
            
            # Get prediction
            result = fixed_smart_pricing_engine.predict_optimal_price(
                client_id=client_id,
                base_amount=base_amount,
                industry=industry,
                project_type=project_type,
                data=current_data
            )
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            print(f"✅ Prediction successful: ${result['optimal_price']['price']:,.2f}")
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ API error: {str(e)}")
            return jsonify({'error': f'Prediction API error: {str(e)}'}), 500
    
    @app.route('/api/retrain-fixed-model', methods=['POST'])
    def api_retrain_fixed_model():
        """Retrain the fixed smart pricing model"""
        try:
            from app import current_data
            
            if current_data is None or len(current_data) == 0:
                return jsonify({'error': 'No data available for retraining'}), 400
            
            print("🔄 Retraining fixed smart pricing model...")
            training_result = fixed_smart_pricing_engine.train_model(current_data)
            
            if training_result['status'] == 'success':
                print(f"✅ Model retrained successfully with {training_result['training_samples']} samples")
                return jsonify({
                    'status': 'success',
                    'message': 'Model retrained successfully',
                    'training_result': training_result
                })
            else:
                return jsonify({'error': f'Retraining failed: {training_result.get("message", "Unknown error")}'}), 500
                
        except Exception as e:
            print(f"❌ Retraining error: {str(e)}")
            return jsonify({'error': f'Retraining error: {str(e)}'}), 500
    
    @app.route('/api/smart-pricing-status', methods=['GET'])
    def api_smart_pricing_status():
        """Get status of the fixed smart pricing system"""
        try:
            from app import current_data
            
            status = {
                'model_trained': fixed_smart_pricing_engine.trained,
                'data_available': current_data is not None and len(current_data) > 0,
                'training_samples': len(fixed_smart_pricing_engine.training_data) if fixed_smart_pricing_engine.training_data is not None else 0,
                'clients_analyzed': len(fixed_smart_pricing_engine.client_stats),
                'system_ready': fixed_smart_pricing_engine.trained and current_data is not None
            }
            
            if current_data is not None:
                status['data_info'] = {
                    'total_records': len(current_data),
                    'unique_clients': len(current_data['client_id'].unique()) if 'client_id' in current_data.columns else 0,
                    'win_rate': (current_data['win_loss'].str.lower() == 'win').mean() if 'win_loss' in current_data.columns else 0
                }
            
            return jsonify({
                'status': 'success',
                'smart_pricing_status': status
            })
            
        except Exception as e:
            return jsonify({'error': f'Status check error: {str(e)}'}), 500
    
    @app.route('/api/client-analysis', methods=['POST'])
    def api_client_analysis():
        """Get detailed analysis for a specific client"""
        try:
            data = request.get_json()
            client_id = data.get('client_id')
            
            if not client_id:
                return jsonify({'error': 'Client ID is required'}), 400
            
            if not fixed_smart_pricing_engine.trained:
                return jsonify({'error': 'Model not trained'}), 400
            
            # Get client statistics
            client_stats = fixed_smart_pricing_engine.client_stats.get(client_id, {})
            
            if not client_stats:
                return jsonify({
                    'client_id': client_id,
                    'status': 'new_client',
                    'message': 'No historical data available for this client',
                    'recommendations': ['Use conservative pricing for new client relationship']
                })
            
            # Calculate additional insights
            analysis = {
                'client_id': client_id,
                'status': 'existing_client',
                'statistics': {
                    'total_bids': client_stats['total_bids'],
                    'wins': client_stats['wins'],
                    'win_rate': round(client_stats['win_rate'], 3),
                    'average_bid': round(client_stats['avg_bid'], 2),
                    'bid_range': {
                        'min': round(client_stats['min_bid'], 2),
                        'max': round(client_stats['max_bid'], 2)
                    }
                },
                'client_classification': fixed_smart_pricing_engine._classify_client(client_stats),
                'insights': []
            }
            
            # Generate insights
            if client_stats['win_rate'] > 0.7:
                analysis['insights'].append('High-performing client with strong win rate')
            elif client_stats['win_rate'] < 0.3:
                analysis['insights'].append('Challenging client - consider aggressive pricing')
            
            if client_stats['total_bids'] > 20:
                analysis['insights'].append('Well-established client relationship with extensive history')
            elif client_stats['total_bids'] < 5:
                analysis['insights'].append('Limited historical data - predictions may be less accurate')
            
            return jsonify(analysis)
            
        except Exception as e:
            return jsonify({'error': f'Client analysis error: {str(e)}'}), 500

def integrate_fixed_smart_pricing(app):
    """Integrate fixed smart pricing with the main Flask app"""
    try:
        add_fixed_smart_pricing_routes(app)
        print("✅ Fixed smart pricing routes integrated successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to integrate fixed smart pricing: {str(e)}")
        return False
