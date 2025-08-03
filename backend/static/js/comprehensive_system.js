/**
 * Comprehensive System JavaScript - Working Predictions + Enhanced Retraining
 */

class ComprehensiveSystem {
    constructor() {
        this.accuracyChart = null;
        this.modelComparisonChart = null;
        this.isRetraining = false;
        
        this.initializeSystem();
    }
    
    initializeSystem() {
        console.log('🚀 Initializing Comprehensive System...');
        
        // Override existing prediction function
        this.overridePredictionFunction();
        
        // Add retraining functionality
        this.addRetrainingInterface();
        
        // Check system status
        this.checkSystemStatus();
        
        console.log('✅ Comprehensive System initialized');
    }
    
    overridePredictionFunction() {
        // Override the existing predictPricing function
        window.predictPricing = async () => {
            console.log('🎯 Using comprehensive prediction system...');
            
            try {
                // Get form values
                const clientId = document.getElementById('clientSelect')?.value;
                const baseAmount = parseFloat(document.getElementById('baseAmount')?.value || 0);
                const industry = document.getElementById('industrySelect')?.value || '';
                const projectType = document.getElementById('projectTypeSelect')?.value || '';
                
                // Validate inputs
                if (!clientId) {
                    this.showError('Please select a client');
                    return;
                }
                
                if (!baseAmount || baseAmount <= 0) {
                    this.showError('Please enter a valid base amount');
                    return;
                }
                
                console.log(`📊 Requesting comprehensive prediction for: ${clientId}, $${baseAmount.toLocaleString()}`);
                
                // Show loading state
                this.showLoadingState();
                
                // Make API call to working endpoint
                const response = await fetch('/api/predict-pricing-working', {
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
                
                console.log('✅ Comprehensive prediction received:', data);
                
                // Display the dynamic results
                this.displayComprehensivePredictionResults(data, baseAmount);
                
            } catch (error) {
                console.error('❌ Comprehensive prediction error:', error);
                this.showError('Prediction failed: ' + error.message);
            } finally {
                this.hideLoadingState();
            }
        };
    }
    
    displayComprehensivePredictionResults(data, baseAmount) {
        console.log('📊 Displaying comprehensive prediction results...');
        
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
        
        // Update the results display with comprehensive dynamic content
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
                        <h6>📊 Client Analysis</h6>
                        <div class="analysis-grid">
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
                        <h6>🤖 Model Analysis</h6>
                        <div class="model-predictions">
                            ${Object.entries(modelInfo.model_predictions).map(([model, pred]) => `
                                <div class="model-pred-item">
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
                    <div class="prediction-summary">
                        <div class="summary-header">
                            <h6>💡 Dynamic Recommendations</h6>
                        </div>
                        <div class="recommendations-list">
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
            </div>
            
            <div class="row mt-3">
                <div class="col-12">
                    <div class="system-info">
                        <div class="info-grid">
                            <div class="info-item">
                                <span class="info-label">Price vs Base:</span>
                                <span class="info-value ${optimal.margin_vs_base >= 0 ? 'text-success' : 'text-primary'}">${marginFormatted}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Competitiveness:</span>
                                <span class="info-value">${(optimal.competitiveness_score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Base Amount:</span>
                                <span class="info-value">$${baseAmount.toLocaleString()}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Prediction Time:</span>
                                <span class="info-value">${new Date(data.system_info.prediction_timestamp).toLocaleTimeString()}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-2">
                <div class="col-12">
                    <div class="action-buttons">
                        <button class="btn btn-primary btn-sm" onclick="comprehensiveSystem.testPredictionVariation('${clientAnalysis.client_id}', ${baseAmount})">
                            🔍 Test Prediction Variation
                        </button>
                        <button class="btn btn-success btn-sm" onclick="comprehensiveSystem.startComprehensiveRetraining()">
                            🚀 Start Enhanced Retraining
                        </button>
                        <button class="btn btn-info btn-sm" onclick="comprehensiveSystem.showAccuracyChart()">
                            📈 Show Accuracy History
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        console.log('✅ Comprehensive results displayed successfully');
    }
    
    addRetrainingInterface() {
        // Add retraining interface to the page
        const existingInterface = document.getElementById('retrainingInterface');
        if (existingInterface) return; // Already exists
        
        const retrainingHTML = `
            <div id="retrainingInterface" class="mt-4" style="display: none;">
                <div class="card">
                    <div class="card-header">
                        <h5>🚀 Enhanced Model Retraining</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div id="retrainingProgress" style="display: none;">
                                    <h6>⚡ Retraining Progress</h6>
                                    <div class="progress mb-3">
                                        <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                             id="retrainingProgressBar" style="width: 0%">0%</div>
                                    </div>
                                    <div id="retrainingLogs" class="retraining-logs"></div>
                                </div>
                                <div id="retrainingResults" style="display: none;">
                                    <h6>📊 Retraining Results</h6>
                                    <div id="retrainingResultsContent"></div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div id="accuracyChartContainer">
                                    <h6>📈 Accuracy Over Time</h6>
                                    <canvas id="accuracyChart" width="400" height="200"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add to the page
        const mainContainer = document.querySelector('.container-fluid') || document.body;
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = retrainingHTML;
        mainContainer.appendChild(tempDiv.firstElementChild);
    }
    
    async startComprehensiveRetraining() {
        if (this.isRetraining) return;
        
        console.log('🚀 Starting comprehensive retraining...');
        this.isRetraining = true;
        
        // Show retraining interface
        const retrainingInterface = document.getElementById('retrainingInterface');
        const retrainingProgress = document.getElementById('retrainingProgress');
        
        if (retrainingInterface) retrainingInterface.style.display = 'block';
        if (retrainingProgress) retrainingProgress.style.display = 'block';
        
        try {
            // Simulate retraining progress
            await this.simulateRetrainingProgress();
            
            // Call retraining API
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
            
            console.log('✅ Comprehensive retraining completed:', data);
            
            // Display results
            this.displayRetrainingResults(data);
            
            // Update accuracy chart
            await this.updateAccuracyChart();
            
            this.showSuccess('Enhanced retraining completed successfully!');
            
        } catch (error) {
            console.error('❌ Comprehensive retraining error:', error);
            this.showError('Retraining failed: ' + error.message);
        } finally {
            this.isRetraining = false;
        }
    }
    
    async simulateRetrainingProgress() {
        const phases = [
            { name: 'Capturing baseline performance', progress: 25 },
            { name: 'Analyzing model performance', progress: 50 },
            { name: 'Retraining ensemble models', progress: 75 },
            { name: 'Validating improvements', progress: 100 }
        ];
        
        for (const phase of phases) {
            this.addRetrainingLog(`🔄 ${phase.name}...`);
            await this.animateProgress(phase.progress);
            this.addRetrainingLog(`✅ ${phase.name} complete`);
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }
    
    animateProgress(targetProgress) {
        return new Promise((resolve) => {
            const progressBar = document.getElementById('retrainingProgressBar');
            if (!progressBar) {
                resolve();
                return;
            }
            
            let currentProgress = parseInt(progressBar.style.width) || 0;
            const increment = (targetProgress - currentProgress) / 20;
            
            const animate = () => {
                currentProgress += increment;
                if (currentProgress >= targetProgress) {
                    currentProgress = targetProgress;
                    progressBar.style.width = `${currentProgress}%`;
                    progressBar.textContent = `${Math.round(currentProgress)}%`;
                    resolve();
                } else {
                    progressBar.style.width = `${currentProgress}%`;
                    progressBar.textContent = `${Math.round(currentProgress)}%`;
                    requestAnimationFrame(animate);
                }
            };
            
            animate();
        });
    }
    
    addRetrainingLog(message) {
        const logsContainer = document.getElementById('retrainingLogs');
        if (logsContainer) {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logsContainer.appendChild(logEntry);
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
    }
    
    displayRetrainingResults(data) {
        const resultsContainer = document.getElementById('retrainingResults');
        const resultsContent = document.getElementById('retrainingResultsContent');
        
        if (!resultsContainer || !resultsContent) return;
        
        const beforeAfter = data.before_after_comparison;
        const improvement = beforeAfter.improvement;
        
        resultsContent.innerHTML = `
            <div class="results-grid">
                <div class="result-item">
                    <div class="result-label">Before Accuracy:</div>
                    <div class="result-value">${(beforeAfter.before.overall_accuracy * 100).toFixed(1)}%</div>
                </div>
                <div class="result-item">
                    <div class="result-label">After Accuracy:</div>
                    <div class="result-value text-success">${(beforeAfter.after.overall_accuracy * 100).toFixed(1)}%</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Improvement:</div>
                    <div class="result-value ${improvement > 0 ? 'text-success' : 'text-warning'}">
                        ${improvement > 0 ? '+' : ''}${(improvement * 100).toFixed(1)}%
                    </div>
                </div>
                <div class="result-item">
                    <div class="result-label">Training Samples:</div>
                    <div class="result-value">${data.retraining_result.training_samples}</div>
                </div>
            </div>
        `;
        
        resultsContainer.style.display = 'block';
    }
    
    async updateAccuracyChart() {
        try {
            const response = await fetch('/api/accuracy-history');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.createAccuracyChart(data.accuracy_history, data.retraining_points);
            }
        } catch (error) {
            console.error('❌ Error updating accuracy chart:', error);
        }
    }
    
    createAccuracyChart(accuracyHistory, retrainingPoints) {
        const canvas = document.getElementById('accuracyChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart
        if (this.accuracyChart) {
            this.accuracyChart.destroy();
        }
        
        // Prepare data
        const chartData = accuracyHistory.map(point => ({
            x: new Date(point.date),
            y: point.accuracy * 100
        }));
        
        this.accuracyChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Model Accuracy (%)',
                    data: chartData,
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        }
                    },
                    y: {
                        beginAtZero: false,
                        min: 60,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Accuracy Over Time'
                    }
                }
            }
        });
    }
    
    async testPredictionVariation(clientId, baseAmount) {
        try {
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
            
            if (data.status === 'success') {
                this.displayVariationResults(data.variation_test, data.analysis);
            } else {
                throw new Error(data.error || 'Variation test failed');
            }
            
        } catch (error) {
            console.error('❌ Variation test error:', error);
            this.showError('Variation test failed: ' + error.message);
        }
    }
    
    displayVariationResults(results, analysis) {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">🔍 Prediction Variation Test Results</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-${analysis.dynamic_behavior === 'Confirmed' ? 'success' : 'warning'}">
                            <strong>Dynamic Behavior: ${analysis.dynamic_behavior}</strong>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Scenario</th>
                                        <th>Client ID</th>
                                        <th>Base Amount</th>
                                        <th>Optimal Price</th>
                                        <th>Win Probability</th>
                                        <th>Expected Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${results.map(result => `
                                        <tr>
                                            <td>${result.scenario}</td>
                                            <td>${result.input?.client_id || 'N/A'}</td>
                                            <td>$${result.input?.base_amount?.toLocaleString() || 'N/A'}</td>
                                            <td>$${result.output?.optimal_price?.toLocaleString() || 'Error'}</td>
                                            <td>${result.output ? (result.output.win_probability * 100).toFixed(1) + '%' : 'Error'}</td>
                                            <td>$${result.output?.expected_value?.toLocaleString() || 'Error'}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();
        
        // Clean up when modal is hidden
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
    }
    
    async showAccuracyChart() {
        const retrainingInterface = document.getElementById('retrainingInterface');
        if (retrainingInterface) {
            retrainingInterface.style.display = 'block';
            await this.updateAccuracyChart();
        }
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/system-status-comprehensive');
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('📊 System Status:', data.system_status);
                this.updateSystemStatusDisplay(data.system_status);
            }
        } catch (error) {
            console.error('❌ Status check error:', error);
        }
    }
    
    updateSystemStatusDisplay(status) {
        // Update any status indicators on the page
        const statusElements = document.querySelectorAll('.system-status');
        statusElements.forEach(element => {
            element.textContent = status.system_ready ? '✅ Ready' : '⚠️ Not Ready';
        });
    }
    
    showLoadingState() {
        const resultsContainer = document.getElementById('predictionResults');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="text-center p-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Generating comprehensive prediction...</p>
                </div>
            `;
        }
    }
    
    hideLoadingState() {
        // Loading state will be replaced by results
    }
    
    showError(message) {
        const resultsContainer = document.getElementById('predictionResults');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="alert alert-danger">
                    <h6>❌ Error</h6>
                    <p>${message}</p>
                    <button class="btn btn-sm btn-outline-danger" onclick="comprehensiveSystem.startComprehensiveRetraining()">
                        🔄 Try Retraining
                    </button>
                </div>
            `;
        }
    }
    
    showSuccess(message) {
        // Create a toast notification
        const toast = document.createElement('div');
        toast.className = 'toast align-items-center text-white bg-success border-0';
        toast.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999;';
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">✅ ${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        const bootstrapToast = new bootstrap.Toast(toast);
        bootstrapToast.show();
        
        // Clean up after toast is hidden
        toast.addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toast);
        });
    }
}

// Initialize the comprehensive system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
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
        
        .analysis-grid, .info-grid, .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .analysis-item, .info-item, .result-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #e9ecef;
        }
        
        .model-predictions {
            background: white;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        
        .model-pred-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        
        .recommendations-list {
            background: white;
            padding: 15px;
            border-radius: 8px;
        }
        
        .recommendation-item {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 4px solid #007bff;
            background: #f8f9fa;
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
        
        .action-buttons {
            text-align: center;
            margin-top: 15px;
        }
        
        .action-buttons .btn {
            margin: 0 5px;
        }
        
        .retraining-logs {
            background: #1a1a1a;
            color: #00ff00;
            font-family: monospace;
            padding: 10px;
            border-radius: 4px;
            height: 150px;
            overflow-y: auto;
            font-size: 0.8em;
        }
        
        .log-entry {
            margin-bottom: 2px;
        }
    `;
    document.head.appendChild(style);
    
    // Initialize the comprehensive system
    window.comprehensiveSystem = new ComprehensiveSystem();
});
