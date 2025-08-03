"""
Enhanced Win Probability Analytics and Competitive Intelligence
Extends the existing system to analyze past winners and optimize against specific competitors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WinProbabilityAnalyzer:
    """
    Advanced analytics system to improve win probability by analyzing
    past winners and competitive patterns for specific clients
    """
    
    def __init__(self):
        self.client_winner_profiles = {}
        self.competitor_strategies = {}
        self.winning_patterns = {}
        self.client_preferences = {}
        self.competitive_network = None
        
    def analyze_client_winners(self, data: pd.DataFrame, client_id: str) -> Dict[str, Any]:
        """
        Analyze past winners for a specific client to identify winning patterns
        
        Args:
            data: Historical bidding data
            client_id: Target client identifier
            
        Returns:
            Comprehensive analysis of winning patterns for the client
        """
        try:
            # Filter data for the specific client
            client_data = data[data['client_id'] == client_id] if 'client_id' in data.columns else data
            
            # Get winning bids only
            winners = client_data[client_data['win_loss'].str.lower() == 'win'].copy()
            
            if len(winners) == 0:
                return {'error': 'No winning bids found for this client'}
            
            # Analyze winning patterns
            winning_analysis = {
                'client_id': client_id,
                'total_bids': len(client_data),
                'total_wins': len(winners),
                'client_win_rate': len(winners) / len(client_data) if len(client_data) > 0 else 0,
                'winner_profiles': self._analyze_winner_profiles(winners),
                'winning_price_patterns': self._analyze_price_patterns(winners),
                'winning_quality_patterns': self._analyze_quality_patterns(winners),
                'temporal_patterns': self._analyze_temporal_patterns(winners),
                'competitive_landscape': self._analyze_competitive_landscape(client_data),
                'client_preferences': self._extract_client_preferences(winners),
                'recommendations': self._generate_winning_recommendations(winners, client_data)
            }
            
            # Store client profile for future use
            self.client_winner_profiles[client_id] = winning_analysis
            
            return winning_analysis
            
        except Exception as e:
            logger.error(f"Client winner analysis error: {str(e)}")
            raise
    
    def _analyze_winner_profiles(self, winners: pd.DataFrame) -> Dict[str, Any]:
        """Analyze profiles of past winners"""
        
        profiles = {}
        
        # Group by provider to analyze winner characteristics
        if 'provider_id' in winners.columns:
            winner_providers = winners['provider_id'].value_counts()
            
            profiles['dominant_winners'] = []
            for provider, wins in winner_providers.head(5).items():
                provider_data = winners[winners['provider_id'] == provider]
                
                profile = {
                    'provider_id': provider,
                    'wins': int(wins),
                    'win_share': wins / len(winners),
                    'avg_price': float(provider_data['price'].mean()),
                    'avg_quality': float(provider_data['quality_score'].mean()),
                    'avg_delivery': float(provider_data['delivery_time'].mean()) if 'delivery_time' in provider_data.columns else None,
                    'price_strategy': self._identify_pricing_strategy(provider_data),
                    'competitive_advantage': self._identify_competitive_advantage(provider_data)
                }
                profiles['dominant_winners'].append(profile)
        
        # Overall winner characteristics
        profiles['winner_characteristics'] = {
            'avg_winning_price': float(winners['price'].mean()),
            'price_std': float(winners['price'].std()),
            'avg_quality_score': float(winners['quality_score'].mean()),
            'quality_std': float(winners['quality_score'].std()),
            'price_quality_correlation': float(winners['price'].corr(winners['quality_score'])),
        }
        
        if 'delivery_time' in winners.columns:
            profiles['winner_characteristics'].update({
                'avg_delivery_time': float(winners['delivery_time'].mean()),
                'delivery_std': float(winners['delivery_time'].std())
            })
        
        return profiles
    
    def _analyze_price_patterns(self, winners: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pricing patterns of winners"""
        
        prices = winners['price'].values
        
        patterns = {
            'price_distribution': {
                'min': float(np.min(prices)),
                'max': float(np.max(prices)),
                'mean': float(np.mean(prices)),
                'median': float(np.median(prices)),
                'std': float(np.std(prices)),
                'percentiles': {
                    '25th': float(np.percentile(prices, 25)),
                    '75th': float(np.percentile(prices, 75)),
                    '90th': float(np.percentile(prices, 90))
                }
            },
            'price_clustering': self._cluster_winning_prices(prices),
            'optimal_price_range': self._calculate_optimal_price_range(winners),
            'price_vs_competition': self._analyze_price_vs_competition(winners)
        }
        
        return patterns
    
    def _analyze_quality_patterns(self, winners: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality patterns of winners"""
        
        quality_scores = winners['quality_score'].values
        
        patterns = {
            'quality_distribution': {
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores)),
                'mean': float(np.mean(quality_scores)),
                'median': float(np.median(quality_scores)),
                'std': float(np.std(quality_scores))
            },
            'quality_thresholds': self._identify_quality_thresholds(winners),
            'quality_price_relationship': self._analyze_quality_price_relationship(winners),
            'quality_competitive_advantage': self._analyze_quality_advantage(winners)
        }
        
        return patterns
    
    def _analyze_temporal_patterns(self, winners: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in winning bids"""
        
        if 'date' not in winners.columns:
            return {'error': 'No date information available'}
        
        winners['date'] = pd.to_datetime(winners['date'])
        winners['month'] = winners['date'].dt.month
        winners['quarter'] = winners['date'].dt.quarter
        winners['year'] = winners['date'].dt.year
        
        patterns = {
            'seasonal_patterns': {
                'monthly_distribution': winners['month'].value_counts().sort_index().to_dict(),
                'quarterly_distribution': winners['quarter'].value_counts().sort_index().to_dict(),
                'peak_months': winners['month'].mode().tolist()
            },
            'trend_analysis': self._analyze_winning_trends(winners),
            'timing_insights': self._extract_timing_insights(winners)
        }
        
        return patterns
    
    def _analyze_competitive_landscape(self, client_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the competitive landscape for the client"""
        
        landscape = {
            'total_unique_providers': client_data['provider_id'].nunique() if 'provider_id' in client_data.columns else 0,
            'avg_competition_per_bid': client_data['num_bids'].mean() if 'num_bids' in client_data.columns else 0,
            'market_concentration': self._calculate_market_concentration(client_data),
            'competitive_intensity_trends': self._analyze_competition_trends(client_data),
            'key_competitors': self._identify_key_competitors(client_data)
        }
        
        return landscape
    
    def _extract_client_preferences(self, winners: pd.DataFrame) -> Dict[str, Any]:
        """Extract client decision-making preferences"""
        
        preferences = {}
        
        # Price sensitivity analysis
        if len(winners) > 5:
            price_range = winners['price'].max() - winners['price'].min()
            price_variance = winners['price'].var()
            
            if price_variance < (winners['price'].mean() * 0.1) ** 2:
                preferences['price_sensitivity'] = 'high'
            elif price_variance > (winners['price'].mean() * 0.3) ** 2:
                preferences['price_sensitivity'] = 'low'
            else:
                preferences['price_sensitivity'] = 'medium'
        
        # Quality preference analysis
        quality_scores = winners['quality_score'].values
        if len(quality_scores) > 3:
            if np.mean(quality_scores) > 7.5:
                preferences['quality_preference'] = 'premium'
            elif np.mean(quality_scores) < 5.5:
                preferences['quality_preference'] = 'cost_focused'
            else:
                preferences['quality_preference'] = 'balanced'
        
        # Delivery time preferences
        if 'delivery_time' in winners.columns:
            delivery_times = winners['delivery_time'].values
            if np.mean(delivery_times) < 20:
                preferences['delivery_preference'] = 'urgent'
            elif np.mean(delivery_times) > 45:
                preferences['delivery_preference'] = 'flexible'
            else:
                preferences['delivery_preference'] = 'standard'
        
        # Provider loyalty analysis
        if 'provider_id' in winners.columns:
            provider_distribution = winners['provider_id'].value_counts()
            max_share = provider_distribution.iloc[0] / len(winners) if len(provider_distribution) > 0 else 0
            
            if max_share > 0.6:
                preferences['provider_loyalty'] = 'high'
            elif max_share < 0.3:
                preferences['provider_loyalty'] = 'low'
            else:
                preferences['provider_loyalty'] = 'medium'
        
        return preferences
    
    def generate_competitive_strategy(self, features: Dict[str, Any], 
                                    client_id: str, 
                                    target_competitors: List[str]) -> Dict[str, Any]:
        """
        Generate strategy to beat specific competitors for a client
        
        Args:
            features: Current bid features
            client_id: Target client
            target_competitors: List of competitor provider IDs to beat
            
        Returns:
            Competitive strategy recommendations
        """
        try:
            # Get client analysis if available
            client_analysis = self.client_winner_profiles.get(client_id, {})
            
            if not client_analysis:
                return {'error': 'Client analysis not available. Run analyze_client_winners first.'}
            
            # Analyze target competitors
            competitor_analysis = self._analyze_target_competitors(target_competitors, client_analysis)
            
            # Generate competitive positioning
            competitive_positioning = self._generate_competitive_positioning(features, competitor_analysis)
            
            # Calculate win probability against each competitor
            win_probabilities = self._calculate_competitor_win_probabilities(features, competitor_analysis)
            
            # Generate specific tactics
            tactics = self._generate_competitive_tactics(features, competitor_analysis, client_analysis)
            
            # Risk assessment
            risk_assessment = self._assess_competitive_risks(competitive_positioning, competitor_analysis)
            
            strategy = {
                'client_id': client_id,
                'target_competitors': target_competitors,
                'competitor_analysis': competitor_analysis,
                'competitive_positioning': competitive_positioning,
                'win_probabilities': win_probabilities,
                'recommended_tactics': tactics,
                'risk_assessment': risk_assessment,
                'success_probability': self._calculate_overall_success_probability(win_probabilities),
                'implementation_roadmap': self._create_implementation_roadmap(tactics)
            }
            
            return strategy
            
        except Exception as e:
            logger.error(f"Competitive strategy generation error: {str(e)}")
            raise
    
    def _analyze_target_competitors(self, competitors: List[str], client_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific competitors for the client"""
        
        analysis = {}
        
        # Get competitor profiles from client's past bids
        dominant_winners = client_analysis.get('winner_profiles', {}).get('dominant_winners', [])
        
        for competitor in competitors:
            competitor_profile = None
            
            # Find competitor in dominant winners
            for winner in dominant_winners:
                if winner['provider_id'] == competitor:
                    competitor_profile = winner
                    break
            
            if competitor_profile:
                analysis[competitor] = {
                    'historical_performance': competitor_profile,
                    'threat_level': self._calculate_threat_level(competitor_profile),
                    'strengths': self._identify_competitor_strengths(competitor_profile),
                    'weaknesses': self._identify_competitor_weaknesses(competitor_profile),
                    'typical_strategy': competitor_profile.get('price_strategy', 'unknown'),
                    'competitive_advantage': competitor_profile.get('competitive_advantage', 'unknown')
                }
            else:
                # Competitor not in past winners - analyze differently
                analysis[competitor] = {
                    'historical_performance': None,
                    'threat_level': 'unknown',
                    'note': 'Competitor has not won bids from this client recently'
                }
        
        return analysis
    
    def _generate_competitive_positioning(self, features: Dict[str, Any], 
                                        competitor_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate competitive positioning strategy"""
        
        positioning = {
            'price_positioning': {},
            'quality_positioning': {},
            'delivery_positioning': {},
            'overall_strategy': 'balanced'
        }
        
        our_price = features.get('estimated_price', 0)
        our_quality = features.get('quality_score', 5)
        our_delivery = features.get('delivery_time', 30)
        
        # Analyze against each competitor
        for competitor_id, analysis in competitor_analysis.items():
            if analysis.get('historical_performance'):
                comp_profile = analysis['historical_performance']
                
                # Price positioning
                comp_avg_price = comp_profile.get('avg_price', our_price)
                if our_price < comp_avg_price * 0.95:
                    positioning['price_positioning'][competitor_id] = 'aggressive_undercut'
                elif our_price > comp_avg_price * 1.05:
                    positioning['price_positioning'][competitor_id] = 'premium_positioning'
                else:
                    positioning['price_positioning'][competitor_id] = 'price_match'
                
                # Quality positioning
                comp_avg_quality = comp_profile.get('avg_quality', our_quality)
                if our_quality > comp_avg_quality + 1:
                    positioning['quality_positioning'][competitor_id] = 'quality_superiority'
                elif our_quality < comp_avg_quality - 1:
                    positioning['quality_positioning'][competitor_id] = 'quality_disadvantage'
                else:
                    positioning['quality_positioning'][competitor_id] = 'quality_parity'
        
        # Determine overall strategy
        price_strategies = list(positioning['price_positioning'].values())
        quality_strategies = list(positioning['quality_positioning'].values())
        
        if 'aggressive_undercut' in price_strategies:
            positioning['overall_strategy'] = 'cost_leadership'
        elif 'quality_superiority' in quality_strategies:
            positioning['overall_strategy'] = 'differentiation'
        else:
            positioning['overall_strategy'] = 'competitive_parity'
        
        return positioning
    
    def _calculate_competitor_win_probabilities(self, features: Dict[str, Any], 
                                              competitor_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate win probability against each competitor"""
        
        win_probabilities = {}
        
        our_quality = features.get('quality_score', 5)
        our_delivery = features.get('delivery_time', 30)
        our_price_comp = features.get('price_competitiveness', 0.5)
        
        for competitor_id, analysis in competitor_analysis.items():
            if analysis.get('historical_performance'):
                comp_profile = analysis['historical_performance']
                
                # Quality factor
                comp_quality = comp_profile.get('avg_quality', 5)
                quality_advantage = (our_quality - comp_quality) / 10.0
                
                # Delivery factor
                comp_delivery = comp_profile.get('avg_delivery', 30)
                delivery_advantage = (comp_delivery - our_delivery) / 60.0  # Faster is better
                
                # Historical performance factor
                comp_win_rate = comp_profile.get('win_share', 0.2)
                performance_factor = 1 - comp_win_rate  # Lower competitor win rate = higher our chance
                
                # Price competitiveness factor
                price_factor = our_price_comp
                
                # Calculate win probability
                win_prob = 0.5 + (  # Base 50% chance
                    quality_advantage * 0.25 +
                    delivery_advantage * 0.15 +
                    performance_factor * 0.20 +
                    price_factor * 0.25 +
                    np.random.normal(0, 0.05)  # Random factor
                )
                
                win_probabilities[competitor_id] = max(0.05, min(0.95, win_prob))
            else:
                win_probabilities[competitor_id] = 0.5  # Default for unknown competitors
        
        return win_probabilities
    
    def _generate_competitive_tactics(self, features: Dict[str, Any],
                                    competitor_analysis: Dict[str, Any],
                                    client_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific competitive tactics"""
        
        tactics = []
        
        # Get client preferences
        client_prefs = client_analysis.get('client_preferences', {})
        
        # Price-based tactics
        if client_prefs.get('price_sensitivity') == 'high':
            tactics.append({
                'type': 'pricing',
                'priority': 'high',
                'tactic': 'aggressive_pricing',
                'description': 'Client is highly price-sensitive. Use competitive pricing strategy.',
                'implementation': 'Price 5-10% below strongest competitor average',
                'risk': 'medium'
            })
        
        # Quality-based tactics
        if client_prefs.get('quality_preference') == 'premium':
            tactics.append({
                'type': 'quality',
                'priority': 'high',
                'tactic': 'quality_differentiation',
                'description': 'Client values premium quality. Emphasize superior quality metrics.',
                'implementation': 'Highlight quality scores above 8, showcase past quality achievements',
                'risk': 'low'
            })
        
        # Delivery-based tactics
        if client_prefs.get('delivery_preference') == 'urgent':
            tactics.append({
                'type': 'delivery',
                'priority': 'medium',
                'tactic': 'fast_delivery_promise',
                'description': 'Client prefers fast delivery. Commit to aggressive timelines.',
                'implementation': 'Offer delivery 20% faster than competitor average',
                'risk': 'high'
            })
        
        # Relationship-based tactics
        if client_prefs.get('provider_loyalty') == 'low':
            tactics.append({
                'type': 'relationship',
                'priority': 'medium',
                'tactic': 'new_provider_advantage',
                'description': 'Client open to new providers. Emphasize fresh perspective.',
                'implementation': 'Position as innovative alternative with proven capabilities',
                'risk': 'low'
            })
        
        # Competitor-specific tactics
        for competitor_id, analysis in competitor_analysis.items():
            if analysis.get('weaknesses'):
                for weakness in analysis['weaknesses']:
                    tactics.append({
                        'type': 'competitive',
                        'priority': 'medium',
                        'tactic': f'exploit_{weakness}',
                        'description': f'Target {competitor_id} weakness in {weakness}',
                        'implementation': f'Emphasize our strength in {weakness} area',
                        'target_competitor': competitor_id,
                        'risk': 'low'
                    })
        
        return tactics
    
    def optimize_bid_parameters(self, features: Dict[str, Any], 
                              client_id: str,
                              target_win_probability: float = 0.7) -> Dict[str, Any]:
        """
        Optimize bid parameters to achieve target win probability against past winners
        
        Args:
            features: Current bid features
            client_id: Target client
            target_win_probability: Desired win probability
            
        Returns:
            Optimized bid parameters
        """
        try:
            client_analysis = self.client_winner_profiles.get(client_id, {})
            
            if not client_analysis:
                return {'error': 'Client analysis not available'}
            
            # Get winning patterns
            winner_characteristics = client_analysis.get('winner_profiles', {}).get('winner_characteristics', {})
            price_patterns = client_analysis.get('winning_price_patterns', {})
            quality_patterns = client_analysis.get('winning_quality_patterns', {})
            
            # Optimize price
            optimal_price = self._optimize_price_for_client(features, price_patterns, target_win_probability)
            
            # Optimize quality score
            optimal_quality = self._optimize_quality_for_client(features, quality_patterns, target_win_probability)
            
            # Optimize delivery time
            optimal_delivery = self._optimize_delivery_for_client(features, client_analysis, target_win_probability)
            
            # Calculate expected win probability with optimizations
            optimized_features = features.copy()
            optimized_features.update({
                'price': optimal_price,
                'quality_score': optimal_quality,
                'delivery_time': optimal_delivery
            })
            
            expected_win_prob = self._calculate_win_probability_vs_winners(optimized_features, client_analysis)
            
            optimization_result = {
                'client_id': client_id,
                'target_win_probability': target_win_probability,
                'expected_win_probability': expected_win_prob,
                'original_features': features,
                'optimized_features': {
                    'price': optimal_price,
                    'quality_score': optimal_quality,
                    'delivery_time': optimal_delivery
                },
                'optimization_impact': {
                    'price_change': optimal_price - features.get('price', optimal_price),
                    'quality_change': optimal_quality - features.get('quality_score', optimal_quality),
                    'delivery_change': optimal_delivery - features.get('delivery_time', optimal_delivery)
                },
                'confidence_level': self._calculate_optimization_confidence(client_analysis),
                'implementation_notes': self._generate_implementation_notes(optimized_features, client_analysis)
            }
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Bid optimization error: {str(e)}")
            raise
    
    def _optimize_price_for_client(self, features: Dict[str, Any], 
                                 price_patterns: Dict[str, Any], 
                                 target_win_prob: float) -> float:
        """Optimize price based on client's winning price patterns"""
        
        current_price = features.get('price', 50000)
        
        if not price_patterns or 'price_distribution' not in price_patterns:
            return current_price
        
        price_dist = price_patterns['price_distribution']
        
        # Target price based on percentiles and target win probability
        if target_win_prob > 0.8:
            # Aggressive pricing - target 25th percentile
            target_price = price_dist.get('percentiles', {}).get('25th', price_dist.get('mean', current_price))
        elif target_win_prob > 0.6:
            # Competitive pricing - target median
            target_price = price_dist.get('median', price_dist.get('mean', current_price))
        else:
            # Conservative pricing - target 75th percentile
            target_price = price_dist.get('percentiles', {}).get('75th', price_dist.get('mean', current_price))
        
        # Ensure reasonable bounds
        min_price = price_dist.get('min', current_price * 0.8)
        max_price = price_dist.get('max', current_price * 1.2)
        
        return max(min_price, min(max_price, target_price))
    
    def _optimize_quality_for_client(self, features: Dict[str, Any],
                                   quality_patterns: Dict[str, Any],
                                   target_win_prob: float) -> float:
        """Optimize quality score based on client's winning quality patterns"""
        
        current_quality = features.get('quality_score', 5)
        
        if not quality_patterns or 'quality_distribution' not in quality_patterns:
            return current_quality
        
        quality_dist = quality_patterns['quality_distribution']
        
        # Target quality based on target win probability
        if target_win_prob > 0.8:
            # High quality target - mean + std
            target_quality = quality_dist.get('mean', 5) + quality_dist.get('std', 1)
        elif target_win_prob > 0.6:
            # Above average quality
            target_quality = quality_dist.get('mean', 5) + 0.5 * quality_dist.get('std', 1)
        else:
            # Average quality
            target_quality = quality_dist.get('mean', 5)
        
        return max(1, min(10, target_quality))
    
    def _optimize_delivery_for_client(self, features: Dict[str, Any],
                                    client_analysis: Dict[str, Any],
                                    target_win_prob: float) -> float:
        """Optimize delivery time based on client preferences"""
        
        current_delivery = features.get('delivery_time', 30)
        
        # Get client delivery preference
        delivery_pref = client_analysis.get('client_preferences', {}).get('delivery_preference', 'standard')
        
        if delivery_pref == 'urgent':
            # Client prefers fast delivery
            if target_win_prob > 0.7:
                target_delivery = current_delivery * 0.8  # 20% faster
            else:
                target_delivery = current_delivery * 0.9  # 10% faster
        elif delivery_pref == 'flexible':
            # Client is flexible on delivery
            target_delivery = current_delivery * 1.1  # Can be 10% slower for better price
        else:
            # Standard delivery expectations
            target_delivery = current_delivery
        
        return max(5, min(90, target_delivery))
    
    def _calculate_win_probability_vs_winners(self, features: Dict[str, Any],
                                            client_analysis: Dict[str, Any]) -> float:
        """Calculate win probability against past winners"""
        
        winner_chars = client_analysis.get('winner_profiles', {}).get('winner_characteristics', {})
        
        if not winner_chars:
            return 0.5
        
        our_price = features.get('price', 50000)
        our_quality = features.get('quality_score', 5)
        our_delivery = features.get('delivery_time', 30)
        
        # Compare against winner averages
        winner_avg_price = winner_chars.get('avg_winning_price', our_price)
        winner_avg_quality = winner_chars.get('avg_quality_score', 5)
        winner_avg_delivery = winner_chars.get('avg_delivery_time', 30)
        
        # Price competitiveness (lower is better)
        price_factor = max(0, 1 - (our_price - winner_avg_price) / winner_avg_price) if winner_avg_price > 0 else 0.5
        
        # Quality competitiveness (higher is better)
        quality_factor = our_quality / 10.0
        
        # Delivery competitiveness (faster is better)
        delivery_factor = max(0, 1 - (our_delivery - winner_avg_delivery) / 60.0) if winner_avg_delivery > 0 else 0.5
        
        # Combine factors
        win_prob = (price_factor * 0.4 + quality_factor * 0.35 + delivery_factor * 0.25)
        
        return max(0.05, min(0.95, win_prob))
    
    # Helper methods for various analyses
    def _identify_pricing_strategy(self, provider_data: pd.DataFrame) -> str:
        """Identify pricing strategy of a provider"""
        if len(provider_data) < 2:
            return 'insufficient_data'
        
        prices = provider_data['price'].values
        price_cv = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        if price_cv < 0.1:
            return 'consistent_pricing'
        elif np.mean(prices) < provider_data['price'].quantile(0.3):
            return 'aggressive_pricing'
        elif np.mean(prices) > provider_data['price'].quantile(0.7):
            return 'premium_pricing'
        else:
            return 'market_pricing'
    
    def _identify_competitive_advantage(self, provider_data: Dict[str, Any]) -> str:
        """Identify competitive advantage of a provider"""
        avg_quality = provider_data.get('avg_quality', 5)
        avg_price = provider_data.get('avg_price', 50000)
        
        if avg_quality > 8:
            return 'quality_leader'
        elif avg_price < 40000:  # Assuming this is low for the market
            return 'cost_leader'
        else:
            return 'balanced_competitor'
    
    def _cluster_winning_prices(self, prices: np.ndarray) -> Dict[str, Any]:
        """Cluster winning prices to identify patterns"""
        if len(prices) < 3:
            return {'clusters': 1, 'pattern': 'insufficient_data'}
        
        # Reshape for clustering
        prices_reshaped = prices.reshape(-1, 1)
        
        # Try different numbers of clusters
        best_clusters = 1
        best_inertia = float('inf')
        
        for n_clusters in range(1, min(5, len(prices))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(prices_reshaped)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_clusters = n_clusters
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(prices_reshaped)
        
        cluster_centers = kmeans.cluster_centers_.flatten()
        
        return {
            'num_clusters': best_clusters,
            'cluster_centers': cluster_centers.tolist(),
            'cluster_assignments': clusters.tolist()
        }
    
    def _calculate_optimal_price_range(self, winners: pd.DataFrame) -> Dict[str, float]:
        """Calculate optimal price range for winning"""
        prices = winners['price'].values
        
        # Use percentiles to define optimal range
        return {
            'min_competitive': float(np.percentile(prices, 10)),
            'optimal_low': float(np.percentile(prices, 25)),
            'optimal_high': float(np.percentile(prices, 75)),
            'max_competitive': float(np.percentile(prices, 90))
        }
    
    def _analyze_price_vs_competition(self, winners: pd.DataFrame) -> Dict[str, Any]:
        """Analyze winning prices vs competition levels"""
        if 'num_bids' not in winners.columns:
            return {'error': 'Competition data not available'}
        
        # Group by competition level
        competition_groups = winners.groupby(pd.cut(winners['num_bids'], bins=3, labels=['Low', 'Medium', 'High']))
        
        analysis = {}
        for group_name, group_data in competition_groups:
            if len(group_data) > 0:
                analysis[str(group_name)] = {
                    'avg_winning_price': float(group_data['price'].mean()),
                    'price_std': float(group_data['price'].std()),
                    'sample_size': len(group_data)
                }
        
        return analysis
    
    def _calculate_threat_level(self, competitor_profile: Dict[str, Any]) -> str:
        """Calculate threat level of a competitor"""
        win_share = competitor_profile.get('win_share', 0)
        avg_quality = competitor_profile.get('avg_quality', 5)
        
        threat_score = win_share * 0.6 + (avg_quality / 10.0) * 0.4
        
        if threat_score > 0.7:
            return 'high'
        elif threat_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _identify_competitor_strengths(self, competitor_profile: Dict[str, Any]) -> List[str]:
        """Identify competitor strengths"""
        strengths = []
        
        if competitor_profile.get('win_share', 0) > 0.4:
            strengths.append('high_win_rate')
        
        if competitor_profile.get('avg_quality', 5) > 7:
            strengths.append('quality_excellence')
        
        strategy = competitor_profile.get('price_strategy', '')
        if 'aggressive' in strategy:
            strengths.append('competitive_pricing')
        elif 'premium' in strategy:
            strengths.append('premium_positioning')
        
        return strengths
    
    def _identify_competitor_weaknesses(self, competitor_profile: Dict[str, Any]) -> List[str]:
        """Identify competitor weaknesses"""
        weaknesses = []
        
        if competitor_profile.get('avg_quality', 5) < 6:
            weaknesses.append('quality_concerns')
        
        if competitor_profile.get('avg_delivery', 30) > 45:
            weaknesses.append('slow_delivery')
        
        # Add more weakness identification logic
        
        return weaknesses
    
    def get_real_time_recommendations(self, features: Dict[str, Any], 
                                    client_id: str) -> Dict[str, Any]:
        """Get real-time recommendations for improving win probability"""
        
        try:
            # Get client analysis
            client_analysis = self.client_winner_profiles.get(client_id, {})
            
            if not client_analysis:
                return {'error': 'No client analysis available. Please run client analysis first.'}
            
            # Current win probability
            current_win_prob = self._calculate_win_probability_vs_winners(features, client_analysis)
            
            # Generate improvement recommendations
            improvements = []
            
            # Price improvements
            price_rec = self._get_price_improvement_recommendation(features, client_analysis)
            if price_rec:
                improvements.append(price_rec)
            
            # Quality improvements
            quality_rec = self._get_quality_improvement_recommendation(features, client_analysis)
            if quality_rec:
                improvements.append(quality_rec)
            
            # Delivery improvements
            delivery_rec = self._get_delivery_improvement_recommendation(features, client_analysis)
            if delivery_rec:
                improvements.append(delivery_rec)
            
            # Sort by impact
            improvements.sort(key=lambda x: x.get('impact', 0), reverse=True)
            
            return {
                'client_id': client_id,
                'current_win_probability': current_win_prob,
                'improvement_recommendations': improvements[:5],  # Top 5 recommendations
                'quick_wins': [rec for rec in improvements if rec.get('effort', 'medium') == 'low'][:3],
                'high_impact_actions': [rec for rec in improvements if rec.get('impact', 0) > 0.1][:3]
            }
            
        except Exception as e:
            logger.error(f"Real-time recommendations error: {str(e)}")
            return {'error': str(e)}
    
    def _get_price_improvement_recommendation(self, features: Dict[str, Any], 
                                           client_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get price-specific improvement recommendation"""
        
        current_price = features.get('price', 0)
        if current_price == 0:
            return None
        
        price_patterns = client_analysis.get('winning_price_patterns', {})
        if not price_patterns:
            return None
        
        optimal_range = price_patterns.get('optimal_price_range', {})
        if not optimal_range:
            return None
        
        # Check if current price is in optimal range
        optimal_low = optimal_range.get('optimal_low', current_price)
        optimal_high = optimal_range.get('optimal_high', current_price)
        
        if current_price < optimal_low:
            return {
                'type': 'price_increase',
                'current_value': current_price,
                'recommended_value': optimal_low,
                'impact': 0.15,
                'effort': 'low',
                'description': f'Increase price to ${optimal_low:,.0f} to align with winning price range',
                'rationale': 'Current price is below the typical winning range for this client'
            }
        elif current_price > optimal_high:
            return {
                'type': 'price_reduction',
                'current_value': current_price,
                'recommended_value': optimal_high,
                'impact': 0.20,
                'effort': 'medium',
                'description': f'Reduce price to ${optimal_high:,.0f} to improve competitiveness',
                'rationale': 'Current price is above the typical winning range for this client'
            }
        
        return None
    
    def _get_quality_improvement_recommendation(self, features: Dict[str, Any],
                                              client_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get quality-specific improvement recommendation"""
        
        current_quality = features.get('quality_score', 5)
        
        quality_patterns = client_analysis.get('winning_quality_patterns', {})
        if not quality_patterns:
            return None
        
        quality_dist = quality_patterns.get('quality_distribution', {})
        winner_avg_quality = quality_dist.get('mean', 5)
        
        if current_quality < winner_avg_quality - 0.5:
            target_quality = min(10, winner_avg_quality + 0.5)
            return {
                'type': 'quality_improvement',
                'current_value': current_quality,
                'recommended_value': target_quality,
                'impact': 0.18,
                'effort': 'high',
                'description': f'Improve quality score to {target_quality:.1f}',
                'rationale': f'Current quality ({current_quality:.1f}) is below winner average ({winner_avg_quality:.1f})'
            }
        
        return None
    
    def _get_delivery_improvement_recommendation(self, features: Dict[str, Any],
                                               client_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get delivery-specific improvement recommendation"""
        
        current_delivery = features.get('delivery_time', 30)
        
        client_prefs = client_analysis.get('client_preferences', {})
        delivery_pref = client_prefs.get('delivery_preference', 'standard')
        
        if delivery_pref == 'urgent' and current_delivery > 25:
            target_delivery = 20
            return {
                'type': 'delivery_acceleration',
                'current_value': current_delivery,
                'recommended_value': target_delivery,
                'impact': 0.12,
                'effort': 'high',
                'description': f'Reduce delivery time to {target_delivery} days',
                'rationale': 'Client prefers urgent delivery and current timeline is too long'
            }
        elif delivery_pref == 'standard' and current_delivery > 40:
            target_delivery = 30
            return {
                'type': 'delivery_optimization',
                'current_value': current_delivery,
                'recommended_value': target_delivery,
                'impact': 0.08,
                'effort': 'medium',
                'description': f'Optimize delivery time to {target_delivery} days',
                'rationale': 'Current delivery time exceeds client expectations'
            }
        
        return None
