/**
 * Prediction Fix Override - Ensures dynamic predictions and adds retraining
 * This script overrides the existing prediction function to use working endpoints
 */

console.log('🔧 Loading Prediction Fix Override...');

// Override the existing predictPricing function when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Applying prediction fixes...');
    
    // Override the existing prediction function
    window.predictPricing = async function() {
        console.log('🎯 Using fixed prediction system...');
        
        try {
            // Get form values
            const clientSelect = document.getElementById('clientSelect');
            const baseAmountInput = document.getElementById('baseAmount');
            const industrySelect = document.getElementById('industrySelect');
            const projectTypeSelect = document.getElementById('projectTypeSelect');
            
            if (!clientSelect || !baseAmountInput) {
                console.error('❌ Required form elements not found');
                showPredictionError('Form elements not found. Please refresh the page.');
                return;
            }
            
            const clientId = clientSelect.value;
            const baseAmount = parseFloat(baseAmountInput.value || 0);
            const industry = industrySelect ? industrySelect.value : '';
            const projectType = projectTypeSelect ? projectTypeSelect.value : '';
            
            // Validate inputs
            if (!clientId) {
                showPredictionError('Please select a client');
                return;
            }
            
            if (!baseAmount || baseAmount <= 0) {
                showPredictionError('Please enter a valid base amount');
                return;
            }
            
            console.log(`📊 Fixed prediction request: ${clientId}, $${baseAmount.toLocaleString()}`);
            
            // Show loading state
            showPredictionLoading();
            
            // Try the working endpoint first
            let response;
            let data;
            
            try {
                response = await fetch('/api/predict-pricing-working', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        client_id: clientId,
                        base_amount: baseAmount,
                        industry: industry,
                        project_type: projectType
                    })
                });
                
                data = await response.json();
                
                if (!response.ok || data.error) {
                    throw new Error(data.error || 'Working endpoint failed');
                }
                
                console.log('✅ Working endpoint success:', data);
                
            } catch (workingError) {
                console.log('⚠️ Working endpoint failed, trying fixed endpoint:', workingError.message);
                
                // Fallback to fixed endpoint
                try {
                    response = await fetch('/api/predict-pricing-fixed', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            client_id: clientId,
                            base_amount: baseAmount,
                            industry: industry,
                            project_type: projectType
                        })
                    });
                    
                    data = await response.json();
                    
                    if (!response.ok || data.error) {
                        throw new Error(data.error || 'Fixed endpoint failed');
                    }
                    
                    console.log('✅ Fixed endpoint success:', data);
                    
                } catch (fixedError) {
                    console.log('⚠️ Fixed endpoint failed, trying comprehensive endpoint:', fixedError.message);
                    
                    // Last fallback to comprehensive endpoint
                    response = await fetch('/api/predict-pricing-working', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            client_id: clientId,
                            base_amount: baseAmount,
                            industry: industry,
                            project_type: projectType
                        })
                    });
                    
                    data = await response.json();
                    
                    if (!response.ok || data.error) {
                        throw new Error(data.error || 'All endpoints failed');
                    }
                }
            }
            
            // Display the results
            displayFixedPredictionResults(data, baseAmount);
            
        } catch (error) {
            console.error('❌ Prediction error:', error);
            showPredictionError('Prediction failed: ' + error.message);
        }
    };
    
    // Add retraining functionality
    window.startEnhancedRetraining = async function() {
        try {
            console.log('🚀 Starting enhanced retraining...');
            showRetrainingProgress();
            
            const response = await fetch('/api/retrain-comprehensive', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (!response.ok || data.error) {
                throw new Error(data.error || 'Retraining failed');
            }
            
            console.log('✅ Retraining completed:', data);
            showRetrainingResults(data);
            
        } catch (error) {
            console.error('❌ Retraining error:', error);
            showPredictionError('Retraining failed: ' + error.message);
        }
    };
    
    // Add test variation functionality
    window.testPredictionVariation = async function() {
        try {
            const clientSelect = document.getElementById('clientSelect');
            const baseAmountInput = document.getElementById('baseAmount');
            
            if (!clientSelect || !baseAmountInput) {
                showPredictionError('Form elements not found');
                return;
            }
            
            const clientId = clientSelect.value || 'TEST-CLIENT';
            const baseAmount = parseFloat(baseAmountInput.value || 250000);
            
            console.log('🔍 Testing prediction variation...');
            
            const response = await fetch('/api/test-prediction-variation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    client_id: clientId,
                    base_amount: baseAmount
                })
            });
            
            const data = await response.json();
            
            if (!response.ok || data.error) {
                throw new Error(data.error || 'Variation test failed');
            }
            
            console.log('✅ Variation test completed:', data);
            showVariationResults(data);
            
        } catch (error) {
            console.error('❌ Variation test error:', error);
            showPredictionError('Variation test failed: ' + error.message);
        }
    };
    
    console.log('✅ Prediction fixes applied successfully');
});

function displayFixedPredictionResults(data, baseAmount) {
    console.log('📊 Displaying fixed prediction results...');
    
    const resultsContainer = document.getElementById('predictionResults');
    if (!resultsContainer) {
        console.error('❌ Results container not found');
        return;
    }
    
    const optimal = data.optimal_price;
    const modelInfo = data.model_information;
    const clientAnalysis = data.client_analysis;
    
    // Format values
    const priceFormatted = `$${optimal.price.toLocaleString()}`;
    const winProbFormatted = `${(optimal.win_probability * 100).toFixed(1)}%`;
    const expectedValueFormatted = `$${optimal.expected_value.toLocaleString()}`;
    const marginFormatted = `${optimal.margin_vs_base > 0 ? '+' : ''}${optimal.margin_vs_base.toFixed(1)}%`;
    
    // Create comprehensive results display
    resultsContainer.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <div class="prediction-card optimal-price">
                    <div class="prediction-value">${priceFormatted}</div>
                    <div class="prediction-label">Optimal Price</div>
                    <div class="prediction-desc">AI-recommended bid amount</div>
                    <div class="prediction-detail">
                        🎯 Primary: ${modelInfo.primary_algorithm} (${modelInfo.algorithm_contribution} contribution)
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="prediction-card win-probability">
                    <div class="prediction-value">${winProbFormatted}</div>
                    <div class="prediction-label">Win Probability</div>
                    <div class="prediction-desc">Chance of winning with this bid</div>
                    <div class="prediction-detail">
                        🎯 Confidence: ${optimal.confidence_level} (${clientAnalysis.historical_bids} historical bids)
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="prediction-card expected-value">
                    <div class="prediction-value">${expectedValueFormatted}</div>
                    <div class="prediction-label">Expected Value</div>
                    <div class="prediction-desc">Price × Win Probability</div>
                    <div class="prediction-detail">
                        🎯 Calculated: ${priceFormatted} × ${winProbFormatted}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-md-6">
                <div class="analysis-section">
                    <h6>📊 Dynamic Client Analysis</h6>
                    <div class="analysis-items">
                        <div class="analysis-item">
                            <span class="label">Client Type:</span>
                            <span class="value">${clientAnalysis.client_type}</span>
                        </div>
                        <div class="analysis-item">
                            <span class="label">Historical Win Rate:</span>
                            <span class="value">${(clientAnalysis.win_rate * 100).toFixed(1)}%</span>
                        </div>
                        <div class="analysis-item">
                            <span class="label">Total Bids:</span>
                            <span class="value">${clientAnalysis.historical_bids}</span>
                        </div>
                        <div class="analysis-item">
                            <span class="label">Average Bid:</span>
                            <span class="value">$${clientAnalysis.average_bid.toLocaleString()}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="analysis-section">
                    <h6>🤖 Model Ensemble Analysis</h6>
                    <div class="model-predictions">
                        ${Object.entries(modelInfo.model_predictions || {}).map(([model, pred]) => `
                            <div class="model-item">
                                <span class="model-name">${model.replace(/_/g, ' ').toUpperCase()}:</span>
                                <span class="model-value">${(pred * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                    <div class="training-info">
                        <small>Training Samples: ${modelInfo.training_samples}</small>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-12">
                <div class="recommendations-section">
                    <h6>💡 Dynamic Recommendations</h6>
                    <div class="recommendations">
                        ${(data.recommendations || []).map(rec => `
                            <div class="recommendation">
                                <div class="rec-type">${rec.type}</div>
                                <div class="rec-action">${rec.action}</div>
                                <div class="rec-rationale">${rec.rationale}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-12">
                <div class="action-buttons text-center">
                    <button class="btn btn-primary btn-sm" onclick="testPredictionVariation()">
                        🔍 Test Prediction Variation
                    </button>
                    <button class="btn btn-success btn-sm" onclick="startEnhancedRetraining()">
                        🚀 Start Enhanced Retraining
                    </button>
                    <button class="btn btn-info btn-sm" onclick="showAccuracyHistory()">
                        📈 Show Accuracy History
                    </button>
                </div>
            </div>
        </div>
        
        <div class="row mt-2">
            <div class="col-12">
                <div class="system-info">
                    <small class="text-muted">
                        ✅ Dynamic prediction generated at ${new Date().toLocaleTimeString()} 
                        | Base: $${baseAmount.toLocaleString()} 
                        | Margin: ${marginFormatted}
                        | System: ${data.system_info ? data.system_info.model_version : 'Fixed'}
                    </small>
                </div>
            </div>
        </div>
    `;
    
    console.log('✅ Fixed results displayed successfully');
}

function showPredictionLoading() {
    const resultsContainer = document.getElementById('predictionResults');
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <div class="text-center p-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Generating dynamic prediction...</p>
            </div>
        `;
    }
}

function showPredictionError(message) {
    const resultsContainer = document.getElementById('predictionResults');
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <h6>❌ Prediction Error</h6>
                <p>${message}</p>
                <div class="mt-2">
                    <button class="btn btn-sm btn-outline-danger" onclick="startEnhancedRetraining()">
                        🔄 Try Retraining
                    </button>
                    <button class="btn btn-sm btn-outline-primary" onclick="testPredictionVariation()">
                        🔍 Test Variation
                    </button>
                </div>
            </div>
        `;
    }
}

function showRetrainingProgress() {
    const resultsContainer = document.getElementById('predictionResults');
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <div class="text-center p-4">
                <div class="spinner-border text-success" role="status">
                    <span class="visually-hidden">Retraining...</span>
                </div>
                <p class="mt-2">🚀 Enhanced retraining in progress...</p>
                <small class="text-muted">This may take a few moments</small>
            </div>
        `;
    }
}

function showRetrainingResults(data) {
    const resultsContainer = document.getElementById('predictionResults');
    if (resultsContainer) {
        const beforeAfter = data.before_after_comparison;
        const improvement = beforeAfter.improvement;
        
        resultsContainer.innerHTML = `
            <div class="alert alert-success">
                <h6>🎉 Enhanced Retraining Completed!</h6>
                <div class="row">
                    <div class="col-md-6">
                        <div class="retraining-metric">
                            <span class="label">Before Accuracy:</span>
                            <span class="value">${(beforeAfter.before.overall_accuracy * 100).toFixed(1)}%</span>
                        </div>
                        <div class="retraining-metric">
                            <span class="label">After Accuracy:</span>
                            <span class="value text-success">${(beforeAfter.after.overall_accuracy * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="retraining-metric">
                            <span class="label">Improvement:</span>
                            <span class="value ${improvement > 0 ? 'text-success' : 'text-warning'}">
                                ${improvement > 0 ? '+' : ''}${(improvement * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div class="retraining-metric">
                            <span class="label">Training Samples:</span>
                            <span class="value">${data.retraining_result.training_samples}</span>
                        </div>
                    </div>
                </div>
                <div class="mt-2">
                    <button class="btn btn-sm btn-primary" onclick="predictPricing()">
                        🎯 Test Improved Predictions
                    </button>
                </div>
            </div>
        `;
    }
}

function showVariationResults(data) {
    const modal = `
        <div class="modal fade" id="variationModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">🔍 Prediction Variation Test Results</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-${data.analysis.dynamic_behavior === 'Confirmed' ? 'success' : 'warning'}">
                            <strong>Dynamic Behavior: ${data.analysis.dynamic_behavior}</strong>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Scenario</th>
                                        <th>Client</th>
                                        <th>Amount</th>
                                        <th>Optimal Price</th>
                                        <th>Win Probability</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.variation_test.map(result => `
                                        <tr>
                                            <td>${result.scenario}</td>
                                            <td>${result.input?.client_id || 'N/A'}</td>
                                            <td>$${result.input?.base_amount?.toLocaleString() || 'N/A'}</td>
                                            <td>$${result.output?.optimal_price?.toLocaleString() || 'Error'}</td>
                                            <td>${result.output ? (result.output.win_probability * 100).toFixed(1) + '%' : 'Error'}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to page and show it
    document.body.insertAdjacentHTML('beforeend', modal);
    const modalElement = document.getElementById('variationModal');
    const bootstrapModal = new bootstrap.Modal(modalElement);
    bootstrapModal.show();
    
    // Clean up when modal is hidden
    modalElement.addEventListener('hidden.bs.modal', () => {
        modalElement.remove();
    });
}

// Add required CSS
const style = document.createElement('style');
style.textContent = `
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .prediction-value {
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .prediction-label {
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .prediction-desc {
        font-size: 0.9em;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    
    .prediction-detail {
        font-size: 0.8em;
        opacity: 0.8;
    }
    
    .analysis-section {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    .analysis-items, .model-predictions {
        background: white;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    .analysis-item, .model-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        padding: 3px 0;
        border-bottom: 1px solid #e9ecef;
    }
    
    .recommendations-section {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
    }
    
    .recommendation {
        background: white;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 10px;
        border-left: 4px solid #007bff;
    }
    
    .rec-type {
        font-weight: bold;
        color: #007bff;
        margin-bottom: 5px;
    }
    
    .rec-action {
        font-weight: 600;
        margin-bottom: 3px;
    }
    
    .rec-rationale {
        font-size: 0.9em;
        color: #6c757d;
    }
    
    .action-buttons .btn {
        margin: 0 5px;
    }
    
    .retraining-metric {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
    }
    
    .system-info {
        text-align: center;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 6px;
    }
`;
document.head.appendChild(style);

console.log('✅ Prediction Fix Override loaded successfully');
