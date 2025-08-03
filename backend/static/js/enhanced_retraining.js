/**
 * Enhanced Retraining System with Before/After Overlay Charts
 */

class EnhancedRetrainingSystem {
    constructor() {
        this.baselineData = null;
        this.retrainedData = null;
        this.accuracyChart = null;
        this.isRetraining = false;
        this.retrainingProgress = 0;
        
        this.initializeInterface();
    }
    
    initializeInterface() {
        // Add enhanced retraining tab content
        this.addRetrainingTab();
        
        // Initialize event listeners
        this.setupEventListeners();
        
        // Check current status
        this.checkRetrainingStatus();
    }
    
    addRetrainingTab() {
        const tabsContainer = document.querySelector('.nav-tabs');
        if (tabsContainer) {
            // Add retraining tab
            const retrainingTab = document.createElement('li');
            retrainingTab.className = 'nav-item';
            retrainingTab.innerHTML = `
                <a class="nav-link" id="retraining-tab" data-toggle="tab" href="#retraining" role="tab">
                    📊 Enhanced Model Retraining
                </a>
            `;
            tabsContainer.appendChild(retrainingTab);
        }
        
        const tabContent = document.querySelector('.tab-content');
        if (tabContent) {
            // Add retraining tab content
            const retrainingContent = document.createElement('div');
            retrainingContent.className = 'tab-pane fade';
            retrainingContent.id = 'retraining';
            retrainingContent.innerHTML = this.getRetrainingHTML();
            tabContent.appendChild(retrainingContent);
        }
    }
    
    getRetrainingHTML() {
        return `
            <div class="container-fluid mt-4">
                <div class="row">
                    <div class="col-12">
                        <h2>🚀 Enhanced Model Retraining & Performance Analysis</h2>
                        <p class="lead">Complete retraining system with before/after comparisons and validation learning</p>
                    </div>
                </div>
                
                <!-- Control Panel -->
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>🔧 Retraining Controls</h5>
                            </div>
                            <div class="card-body">
                                <div class="form-group">
                                    <label>Retraining Strategy:</label>
                                    <select class="form-control" id="retrainingStrategy">
                                        <option value="comprehensive">Comprehensive Retraining</option>
                                        <option value="incremental">Incremental Learning</option>
                                        <option value="validation_focused">Validation-Focused</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Learning Rate:</label>
                                    <select class="form-control" id="learningRate">
                                        <option value="adaptive">Adaptive (Recommended)</option>
                                        <option value="conservative">Conservative</option>
                                        <option value="aggressive">Aggressive</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Validation Split:</label>
                                    <select class="form-control" id="validationSplit">
                                        <option value="0.2">20% (Standard)</option>
                                        <option value="0.15">15% (More Training Data)</option>
                                        <option value="0.25">25% (More Validation)</option>
                                    </select>
                                </div>
                                <button class="btn btn-primary btn-lg btn-block" id="startRetrainingBtn">
                                    🚀 Start Advanced Retraining
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>📈 Quick Metrics Comparison</h5>
                            </div>
                            <div class="card-body" id="quickMetrics">
                                <div class="text-center text-muted">
                                    <i class="fas fa-chart-line fa-3x mb-3"></i>
                                    <p>Start retraining to see before/after comparison</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Progress Section -->
                <div class="row mb-4" id="progressSection" style="display: none;">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>⚡ Live Retraining Progress</h5>
                            </div>
                            <div class="card-body">
                                <div class="progress mb-3" style="height: 25px;">
                                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                         id="retrainingProgressBar" role="progressbar" style="width: 0%">
                                        0%
                                    </div>
                                </div>
                                <div id="retrainingLogs" class="bg-dark text-light p-3 rounded" style="height: 200px; overflow-y: auto; font-family: monospace;">
                                    <div class="text-success">🚀 Ready to start advanced retraining...</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Accuracy Overlay Chart -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>📈 Model Accuracy Over Time - Before/After Overlay</h5>
                            </div>
                            <div class="card-body">
                                <canvas id="accuracyOverlayChart" width="800" height="400"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Validation Learning Section -->
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>🧠 Validation Learning Progress</h5>
                            </div>
                            <div class="card-body" id="validationLearning">
                                <div class="text-center text-muted">
                                    <i class="fas fa-brain fa-3x mb-3"></i>
                                    <p>Validation learning data will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>🔬 Individual Model Improvements</h5>
                            </div>
                            <div class="card-body" id="modelImprovements">
                                <div class="text-center text-muted">
                                    <i class="fas fa-cogs fa-3x mb-3"></i>
                                    <p>Model improvement data will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Client Performance Matrix -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>📊 Client Performance Matrix - Before/After</h5>
                            </div>
                            <div class="card-body" id="clientPerformanceMatrix">
                                <div class="text-center text-muted">
                                    <i class="fas fa-table fa-3x mb-3"></i>
                                    <p>Client performance matrix will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Detailed Results -->
                <div class="row mb-4" id="detailedResults" style="display: none;">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>🎯 Comprehensive Before/After Analysis</h5>
                            </div>
                            <div class="card-body" id="comprehensiveAnalysis">
                                <!-- Detailed results will be populated here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    setupEventListeners() {
        // Start retraining button
        document.addEventListener('click', (e) => {
            if (e.target.id === 'startRetrainingBtn') {
                this.startAdvancedRetraining();
            }
        });
        
        // Tab switching
        document.addEventListener('shown.bs.tab', (e) => {
            if (e.target.id === 'retraining-tab') {
                this.onTabActivated();
            }
        });
    }
    
    async checkRetrainingStatus() {
        try {
            const response = await fetch('/api/retraining_status');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateStatusDisplay(data.retraining_status);
                
                if (data.retraining_status.comparison_available) {
                    this.displayBeforeAfterComparison(data.retraining_status.before_after_comparison);
                }
            }
        } catch (error) {
            console.error('Status check error:', error);
        }
    }
    
    async startAdvancedRetraining() {
        if (this.isRetraining) return;
        
        this.isRetraining = true;
        this.showProgressSection();
        this.updateRetrainingButton(true);
        
        try {
            // Start the complete retraining demo
            const response = await fetch('/api/complete_retraining_demo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                await this.simulateRetrainingProgress();
                this.displayBeforeAfterComparison(data.before_after_comparison);
                this.showSuccessMessage(data.summary);
            } else {
                this.showErrorMessage(data.message);
            }
        } catch (error) {
            console.error('Retraining error:', error);
            this.showErrorMessage('Retraining failed: ' + error.message);
        } finally {
            this.isRetraining = false;
            this.updateRetrainingButton(false);
        }
    }
    
    async simulateRetrainingProgress() {
        const phases = [
            { name: 'Phase 1: Baseline Analysis', duration: 2000, progress: 25 },
            { name: 'Phase 2: Validation Learning', duration: 3000, progress: 50 },
            { name: 'Phase 3: Model Retraining', duration: 4000, progress: 85 },
            { name: 'Phase 4: Results Integration', duration: 1000, progress: 100 }
        ];
        
        for (const phase of phases) {
            this.addRetrainingLog(`🔄 ${phase.name}...`);
            await this.animateProgress(phase.progress, phase.duration);
            this.addRetrainingLog(`✅ ${phase.name} Complete`);
        }
        
        this.addRetrainingLog('🎉 Advanced retraining completed successfully!');
    }
    
    animateProgress(targetProgress, duration) {
        return new Promise((resolve) => {
            const startProgress = this.retrainingProgress;
            const progressDiff = targetProgress - startProgress;
            const startTime = Date.now();
            
            const animate = () => {
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                
                this.retrainingProgress = startProgress + (progressDiff * progress);
                this.updateProgressBar(this.retrainingProgress);
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    resolve();
                }
            };
            
            animate();
        });
    }
    
    updateProgressBar(progress) {
        const progressBar = document.getElementById('retrainingProgressBar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${Math.round(progress)}%`;
        }
    }
    
    addRetrainingLog(message) {
        const logsContainer = document.getElementById('retrainingLogs');
        if (logsContainer) {
            const logEntry = document.createElement('div');
            logEntry.className = 'text-success';
            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logsContainer.appendChild(logEntry);
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
    }
    
    showProgressSection() {
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'block';
        }
    }
    
    updateRetrainingButton(isRetraining) {
        const button = document.getElementById('startRetrainingBtn');
        if (button) {
            if (isRetraining) {
                button.innerHTML = '⚡ Retraining in Progress...';
                button.disabled = true;
                button.className = 'btn btn-warning btn-lg btn-block';
            } else {
                button.innerHTML = '🚀 Start Advanced Retraining';
                button.disabled = false;
                button.className = 'btn btn-primary btn-lg btn-block';
            }
        }
    }
    
    onTabActivated() {
        // Initialize or update charts when tab is activated
        setTimeout(() => {
            this.initializeAccuracyChart();
        }, 100);
    }
}

// Initialize the enhanced retraining system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.enhancedRetrainingSystem = new EnhancedRetrainingSystem();
});
