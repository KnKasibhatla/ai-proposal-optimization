/**
 * Fixed Smart Pricing JavaScript - Ensures Dynamic Predictions
 */

// Override the existing prediction function to use the fixed endpoint
function fixSmartPricingPredictions() {
    console.log('🔧 Applying smart pricing fixes...');
    
    // Find and replace the existing prediction function
    const originalPredictPricing = window.predictPricing;
    
    window.predictPricing = async function() {
        console.log('🎯 Using fixed smart pricing prediction...');
        
        try {
            // Get form values
            const clientId = document.getElementById('clientSelect')?.value;
            const baseAmount = parseFloat(document.getElementById('baseAmount')?.value || 0);
            const industry = document.getElementById('industrySelect')?.value || '';
            const projectType = document.getElementById('projectTypeSelect')?.value || '';
            
            // Validate inputs
            if (!clientId) {
                showError('Please select a client');
                return;
            }
            
            if (!baseAmount || baseAmount <= 0) {
                showError('Please enter a valid base amount');
                return;
            }
            
            console.log(`📊 Requesting prediction for: ${clientId}, $${baseAmount.toLocaleString()}`);
            
            // Show loading state
            showLoadingState();
            
            // Make API call to fixed endpoint
            const response = await fetch('/api/predict-pricing-fixed', {
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
            
            const data = await response.json();
            
            if (!response.ok || data.error) {
                throw new Error(data.error || 'Prediction failed');
            }
            
            console.log('✅ Fixed prediction received:', data);
            
            // Display the dynamic results
            displayDynamicPredictionResults(data, baseAmount);
            
        } catch (error) {
            console.error('❌ Fixed prediction error:', error);
            showError('Prediction failed: ' + error.message);
        } finally {
            hideLoadingState();
        }
    };
    
    // Add retrain model function
    window.retrainFixedModel = async function() {
        try {
            console.log('🔄 Retraining fixed smart pricing model...');
            showLoadingState('Retraining model...');
            
            const response = await fetch('/api/retrain-fixed-model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (!response.ok || data.error) {
                throw new Error(data.error || 'Retraining failed');
            }
            
            console.log('✅ Model retrained successfully:', data);
            showSuccess('Model retrained successfully with ' + data.training_result.training_samples + ' samples');
            
        } catch (error) {
            console.error('❌ Retraining error:', error);
            showError('Retraining failed: ' + error.message);
        } finally {
            hideLoadingState();
        }
    };
    
    // Add system status check
    window.checkSmartPricingStatus = async function() {
        try {
            const response = await fetch('/api/smart-pricing-status');
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('📊 Smart pricing status:', data.smart_pricing_status);
                displaySystemStatus(data.smart_pricing_status);
            }
            
        } catch (error) {
            console.error('❌ Status check error:', error);
        }
    };
}

function displayDynamicPredictionResults(data, baseAmount) {
    console.log('📊 Displaying dynamic prediction results...');
    
    const resultsContainer = document.getElementById('predictionResults');
    if (!resultsContainer) {
        console.error('❌ Results container not found');
        return;
    }
    
    const optimal = data.optimal_price;
    const modelInfo = data.model_information;
    const clientAnalysis = data.client_analysis;
    
    // Calculate dynamic values
    const priceFormatted = `$${optimal.price.toLocaleString()}`;
    const winProbFormatted = `${(optimal.win_probability * 100).toFixed(1)}%`;
    const expectedValueFormatted = `$${optimal.expected_value.toLocaleString()}`;
    const marginFormatted = `${optimal.margin_vs_base > 0 ? '+' : ''}${optimal.margin_vs_base.toFixed(1)}%`;
    
    // Update the results display with dynamic content
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
            <div class="col-12">
                <div class="prediction-summary">
                    <h6>📊 Dynamic Analysis Summary</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="summary-item">
                                <span class="summary-label">Price vs Base:</span>
                                <span class="summary-value ${optimal.margin_vs_base >= 0 ? 'text-success' : 'text-primary'}">${marginFormatted}</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-label">Client Type:</span>
                                <span class="summary-value">${clientAnalysis.client_type}</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-label">Historical Win Rate:</span>
                                <span class="summary-value">${(clientAnalysis.win_rate * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="summary-item">
                                <span class="summary-label">Competitiveness:</span>
                                <span class="summary-value">${(optimal.competitiveness_score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-label">Training Samples:</span>
                                <span class="summary-value">${modelInfo.training_samples}</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-label">Base Amount:</span>
                                <span class="summary-value">$${baseAmount.toLocaleString()}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        ${data.recommendations && data.recommendations.length > 0 ? `
        <div class="row mt-3">
            <div class="col-12">
                <div class="recommendations">
                    <h6>💡 Dynamic Recommendations</h6>
                    ${data.recommendations.map(rec => `
                        <div class="recommendation-item">
                            <div class="rec-type">${rec.type}</div>
                            <div class="rec-action">${rec.action}</div>
                            <div class="rec-rationale">${rec.rationale}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
        ` : ''}
        
        <div class="row mt-2">
            <div class="col-12">
                <div class="system-note">
                    <small class="text-muted">
                        ${data.system_note || 'Dynamic prediction generated successfully'}
                    </small>
                </div>
            </div>
        </div>
    `;
    
    // Show price sensitivity chart if available
    if (data.price_sensitivity && data.price_sensitivity.length > 0) {
        displayPriceSensitivityChart(data.price_sensitivity);
    }
    
    console.log('✅ Dynamic results displayed successfully');
}

function displayPriceSensitivityChart(priceSensitivity) {
    // Create or update price sensitivity chart
    const chartContainer = document.getElementById('priceSensitivityChart');
    if (!chartContainer) return;
    
    const prices = priceSensitivity.map(p => p.price);
    const winProbs = priceSensitivity.map(p => p.win_probability);
    const expectedValues = priceSensitivity.map(p => p.expected_value);
    
    // Simple chart implementation (you can enhance this with Chart.js)
    chartContainer.innerHTML = `
        <h6>📈 Price Sensitivity Analysis</h6>
        <div class="chart-placeholder">
            <p>Price range: $${Math.min(...prices).toLocaleString()} - $${Math.max(...prices).toLocaleString()}</p>
            <p>Win probability range: ${(Math.min(...winProbs) * 100).toFixed(1)}% - ${(Math.max(...winProbs) * 100).toFixed(1)}%</p>
            <p>Expected value range: $${Math.min(...expectedValues).toLocaleString()} - $${Math.max(...expectedValues).toLocaleString()}</p>
        </div>
    `;
}

function displaySystemStatus(status) {
    console.log('📊 System Status:', status);
    
    // You can add a status indicator to the UI
    const statusIndicator = document.getElementById('systemStatus');
    if (statusIndicator) {
        const statusText = status.system_ready ? 
            `✅ Ready (${status.training_samples} samples, ${status.clients_analyzed} clients)` :
            '⚠️ Not Ready - Please upload data';
        
        statusIndicator.innerHTML = statusText;
    }
}

function showLoadingState(message = 'Generating prediction...') {
    const resultsContainer = document.getElementById('predictionResults');
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <div class="text-center p-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="sr-only">Loading...</span>
                </div>
                <p class="mt-2">${message}</p>
            </div>
        `;
    }
}

function hideLoadingState() {
    // Loading state will be replaced by results or error message
}

function showError(message) {
    const resultsContainer = document.getElementById('predictionResults');
    if (resultsContainer) {
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <h6>❌ Prediction Error</h6>
                <p>${message}</p>
                <button class="btn btn-sm btn-outline-danger" onclick="retrainFixedModel()">
                    🔄 Retrain Model
                </button>
            </div>
        `;
    }
    console.error('❌ Error:', message);
}

function showSuccess(message) {
    // You can implement a toast notification or alert
    console.log('✅ Success:', message);
    
    // Simple alert for now
    if (typeof alert !== 'undefined') {
        alert('✅ ' + message);
    }
}

// Auto-apply fixes when the script loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing fixed smart pricing system...');
    
    // Apply the fixes
    fixSmartPricingPredictions();
    
    // Check system status
    setTimeout(() => {
        if (typeof checkSmartPricingStatus === 'function') {
            checkSmartPricingStatus();
        }
    }, 1000);
    
    console.log('✅ Fixed smart pricing system initialized');
});

// Make functions globally available
window.fixSmartPricingPredictions = fixSmartPricingPredictions;
window.displayDynamicPredictionResults = displayDynamicPredictionResults;
