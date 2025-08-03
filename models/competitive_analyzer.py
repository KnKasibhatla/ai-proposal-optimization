"""
Competitive Analyzer
Implements game-theoretic analysis and competitive dynamics modeling
Based on the research paper's competitive dynamics framework
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nash
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
from itertools import combinations
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CompetitiveAnalyzer:
    """
    Competitive dynamics analyzer implementing game-theoretic frameworks
    and strategic behavior modeling for B2B proposal optimization
    """
    
    def __init__(self):
        """Initialize the competitive analyzer"""
        self.provider_profiles = {}
        self.market_segments = {}
        self.competition_network = None
        self.nash_equilibria = {}
        self.strategic_patterns = {}
        
        logger.info("Competitive analyzer initialized")
    
    def analyze_competition(self, features: Dict[str, Any], competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze competitive landscape for a specific bid
        
        Args:
            features: Current bid features
            competitors: List of competitor information
            
        Returns:
            Competitive analysis results
        """
        try:
            # Analyze competitor profiles
            competitor_analysis = self._analyze_competitor_profiles(competitors)
            
            # Calculate market positioning
            market_position = self._calculate_market_position(features, competitors)
            
            # Assess competitive intensity
            competitive_intensity = self._assess_competitive_intensity(competitors)
            
            # Strategic recommendations
            strategic_recommendations = self._generate_strategic_recommendations(
                features, competitor_analysis, market_position
            )
            
            # Risk assessment
            risk_assessment = self._assess_competitive_risks(competitors)
            
            result = {
                'competitor_analysis': competitor_analysis,
                'market_position': market_position,
                'competitive_intensity': competitive_intensity,
                'strategic_recommendations': strategic_recommendations,
                'risk_assessment': risk_assessment,
                'win_probability_factors': self._calculate_win_factors(features, competitors)
            }
            
            logger.info("Competitive analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Competitive analysis error: {str(e)}")
            raise
    
    def _analyze_competitor_profiles(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze individual competitor profiles"""
        
        if not competitors:
            return {'total_competitors': 0, 'profiles': []}
        
        profiles = []
        
        for comp in competitors:
            profile = {
                'provider_id': comp.get('provider_id', 'unknown'),
                'historical_win_rate': comp.get('win_rate', 0.5),
                'avg_quality_score': comp.get('quality_score', 5),
                'avg_delivery_time': comp.get('delivery_time', 30),
                'pricing_strategy': self._identify_pricing_strategy(comp),
                'market_share': comp.get('market_share', 0.1),
                'specialization': comp.get('specialization', 'general'),
                'threat_level': self._calculate_threat_level(comp)
            }
            profiles.append(profile)
        
        # Aggregate analysis
        analysis = {
            'total_competitors': len(competitors),
            'profiles': profiles,
            'avg_win_rate': np.mean([p['historical_win_rate'] for p in profiles]),
            'quality_distribution': self._analyze_quality_distribution(profiles),
            'pricing_strategies': self._analyze_pricing_strategies(profiles),
            'market_concentration': self._calculate_market_concentration(profiles)
        }
        
        return analysis
    
    def _identify_pricing_strategy(self, competitor: Dict[str, Any]) -> str:
        """Identify competitor's pricing strategy"""
        
        # Simple heuristic-based strategy identification
        win_rate = competitor.get('win_rate', 0.5)
        avg_price_ratio = competitor.get('avg_price_ratio', 1.0)
        
        if avg_price_ratio < 0.9 and win_rate > 0.6:
            return 'aggressive_low_pricing'
        elif avg_price_ratio > 1.1 and win_rate > 0.4:
            return 'premium_positioning'
        elif 0.95 <= avg_price_ratio <= 1.05:
            return 'market_following'
        elif win_rate < 0.3:
            return 'struggling'
        else:
            return 'balanced'
    
    def _calculate_threat_level(self, competitor: Dict[str, Any]) -> float:
        """Calculate threat level of a competitor"""
        
        win_rate = competitor.get('win_rate', 0.5)
        market_share = competitor.get('market_share', 0.1)
        quality_score = competitor.get('quality_score', 5) / 10.0
        
        # Weighted threat score
        threat_score = (win_rate * 0.4 + 
                       market_share * 0.3 + 
                       quality_score * 0.3)
        
        return min(1.0, threat_score)
    
    def _calculate_market_position(self, features: Dict[str, Any], competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate market positioning relative to competitors"""
        
        if not competitors:
            return {'position': 'unknown', 'relative_strength': 0.5}
        
        # Our metrics
        our_quality = features.get('quality_score', 5) / 10.0
        our_delivery = 1 - (features.get('delivery_time', 30) / 90.0)
        our_price_competitiveness = features.get('price_competitiveness', 0.5)
        
        # Competitor averages
        comp_quality = np.mean([c.get('quality_score', 5) for c in competitors]) / 10.0
        comp_delivery = 1 - (np.mean([c.get('delivery_time', 30) for c in competitors]) / 90.0)
        comp_price = np.mean([c.get('price_competitiveness', 0.5) for c in competitors])
        
        # Relative positioning
        quality_advantage = our_quality - comp_quality
        delivery_advantage = our_delivery - comp_delivery
        price_advantage = our_price_competitiveness - comp_price
        
        # Overall relative strength
        relative_strength = (quality_advantage + delivery_advantage + price_advantage) / 3.0
        
        # Determine position
        if relative_strength > 0.2:
            position = 'market_leader'
        elif relative_strength > 0:
            position = 'strong_competitor'
        elif relative_strength > -0.2:
            position = 'average_competitor'
        else:
            position = 'weak_competitor'
        
        return {
            'position': position,
            'relative_strength': relative_strength,
            'quality_advantage': quality_advantage,
            'delivery_advantage': delivery_advantage,
            'price_advantage': price_advantage,
            'competitive_score': (relative_strength + 1) / 2  # Normalize to 0-1
        }
    
    def _assess_competitive_intensity(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the intensity of competition"""
        
        num_competitors = len(competitors)
        
        if num_competitors == 0:
            return {'level': 'none', 'score': 0, 'factors': []}
        
        # Base intensity from number of competitors
        base_intensity = min(1.0, num_competitors / 10.0)
        
        # Quality of competition
        avg_threat_level = np.mean([self._calculate_threat_level(c) for c in competitors])
        
        # Market concentration
        market_shares = [c.get('market_share', 1/num_competitors) for c in competitors]
        hhi = sum(share**2 for share in market_shares)  # Herfindahl-Hirschman Index
        concentration_factor = 1 - hhi  # Higher when less concentrated
        
        # Overall intensity score
        intensity_score = (base_intensity * 0.4 + 
                          avg_threat_level * 0.4 + 
                          concentration_factor * 0.2)
        
        # Determine intensity level
        if intensity_score > 0.7:
            level = 'very_high'
        elif intensity_score > 0.5:
            level = 'high'
        elif intensity_score > 0.3:
            level = 'medium'
        else:
            level = 'low'
        
        factors = []
        if num_competitors > 5:
            factors.append('High number of competitors')
        if avg_threat_level > 0.6:
            factors.append('Strong competitor profiles')
        if concentration_factor > 0.7:
            factors.append('Fragmented market')
        
        return {
            'level': level,
            'score': intensity_score,
            'num_competitors': num_competitors,
            'avg_threat_level': avg_threat_level,
            'market_concentration': hhi,
            'factors': factors
        }
    
    def _generate_strategic_recommendations(self, features: Dict[str, Any], 
                                          competitor_analysis: Dict[str, Any],
                                          market_position: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate strategic recommendations based on competitive analysis"""
        
        recommendations = []
        
        # Price strategy recommendations
        if market_position['price_advantage'] < -0.1:
            recommendations.append({
                'type': 'pricing',
                'priority': 'high',
                'recommendation': 'Consider more aggressive pricing to improve competitiveness',
                'rationale': 'Current pricing appears disadvantageous relative to competitors'
            })
        
        # Quality positioning recommendations
        if market_position['quality_advantage'] > 0.1:
            recommendations.append({
                'type': 'positioning',
                'priority': 'medium',
                'recommendation': 'Leverage quality advantage in proposal messaging',
                'rationale': 'Quality scores exceed competitor average'
            })
        
        # Delivery time recommendations
        if market_position['delivery_advantage'] < -0.1:
            recommendations.append({
                'type': 'operations',
                'priority': 'medium',
                'recommendation': 'Improve delivery timeline to match competitor standards',
                'rationale': 'Delivery times are slower than competitive average'
            })
        
        # Market position strategy
        position = market_position['position']
        if position == 'weak_competitor':
            recommendations.append({
                'type': 'strategy',
                'priority': 'high',
                'recommendation': 'Focus on niche specialization or strategic partnerships',
                'rationale': 'Overall competitive position is weak across multiple dimensions'
            })
        elif position == 'market_leader':
            recommendations.append({
                'type': 'strategy',
                'priority': 'low',
                'recommendation': 'Maintain current positioning while monitoring competitor moves',
                'rationale': 'Strong competitive position across key metrics'
            })
        
        # Competitive intensity response
        intensity = competitor_analysis.get('competitive_intensity', {}).get('level', 'medium')
        if intensity == 'very_high':
            recommendations.append({
                'type': 'competitive_response',
                'priority': 'high',
                'recommendation': 'Consider differentiation strategy to avoid price wars',
                'rationale': 'High competitive intensity may lead to margin erosion'
            })
        
        return recommendations
    
    def _assess_competitive_risks(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess competitive risks"""
        
        risks = []
        risk_score = 0
        
        if not competitors:
            return {'risks': [], 'overall_risk': 'low', 'risk_score': 0}
        
        # High threat competitors
        high_threat_competitors = [c for c in competitors if self._calculate_threat_level(c) > 0.7]
        if high_threat_competitors:
            risks.append({
                'type': 'high_threat_competitors',
                'severity': 'high',
                'description': f'{len(high_threat_competitors)} competitors pose significant threat',
                'mitigation': 'Enhance differentiation and value proposition'
            })
            risk_score += 0.3
        
        # Price pressure risk
        aggressive_pricers = [c for c in competitors if c.get('pricing_strategy') == 'aggressive_low_pricing']
        if len(aggressive_pricers) > len(competitors) * 0.3:
            risks.append({
                'type': 'price_pressure',
                'severity': 'medium',
                'description': 'Multiple competitors using aggressive pricing strategies',
                'mitigation': 'Focus on value-based selling and cost optimization'
            })
            risk_score += 0.2
        
        # Market saturation risk
        if len(competitors) > 8:
            risks.append({
                'type': 'market_saturation',
                'severity': 'medium',
                'description': 'High number of competitors indicates market saturation',
                'mitigation': 'Consider market segmentation or new market entry'
            })
            risk_score += 0.15
        
        # Determine overall risk level
        if risk_score > 0.5:
            overall_risk = 'high'
        elif risk_score > 0.25:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'risks': risks,
            'overall_risk': overall_risk,
            'risk_score': min(1.0, risk_score),
            'mitigation_priority': 'high' if risk_score > 0.4 else 'medium' if risk_score > 0.2 else 'low'
        }
    
    def _calculate_win_factors(self, features: Dict[str, Any], competitors: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate factors contributing to win probability"""
        
        factors = {}
        
        # Quality factor
        our_quality = features.get('quality_score', 5)
        comp_avg_quality = np.mean([c.get('quality_score', 5) for c in competitors]) if competitors else 5
        factors['quality_factor'] = max(0, (our_quality - comp_avg_quality + 5) / 10)
        
        # Price factor
        our_price_comp = features.get('price_competitiveness', 0.5)
        factors['price_factor'] = our_price_comp
        
        # Delivery factor
        our_delivery = features.get('delivery_time', 30)
        comp_avg_delivery = np.mean([c.get('delivery_time', 30) for c in competitors]) if competitors else 30
        factors['delivery_factor'] = max(0, 1 - (our_delivery - comp_avg_delivery + 30) / 60)
        
        # Experience factor
        our_experience = features.get('provider_experience', 10)
        factors['experience_factor'] = min(1.0, our_experience / 50)
        
        # Market position factor
        market_pos = self._calculate_market_position(features, competitors)
        factors['market_position_factor'] = market_pos['competitive_score']
        
        return factors
    
    def find_nash_equilibrium(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find Nash equilibrium for the competitive bidding game
        
        Args:
            game_data: Game parameters including players and payoff structures
            
        Returns:
            Nash equilibrium analysis
        """
        try:
            players = game_data.get('players', [])
            if len(players) < 2:
                return {'equilibrium': None, 'message': 'Insufficient players for game analysis'}
            
            # Simplified 2-player game for demonstration
            if len(players) == 2:
                return self._find_two_player_nash_equilibrium(game_data)
            else:
                return self._find_multi_player_nash_equilibrium(game_data)
                
        except Exception as e:
            logger.error(f"Nash equilibrium calculation error: {str(e)}")
            return {'equilibrium': None, 'error': str(e)}
    
    def _find_two_player_nash_equilibrium(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find Nash equilibrium for 2-player game"""
        
        # Simplified bidding game
        # Each player chooses a price from a discrete set
        price_range = np.linspace(0.8, 1.2, 11)  # Price multipliers
        
        # Payoff calculation
        def calculate_payoff(p1_price, p2_price, base_value=100):
            # Winner takes all, payoff = (value - cost) if win, 0 if lose
            cost1 = base_value * p1_price
            cost2 = base_value * p2_price
            
            if p1_price < p2_price:
                return base_value - cost1, 0
            elif p2_price < p1_price:
                return 0, base_value - cost2
            else:
                # Tie - split the value
                return (base_value - cost1) / 2, (base_value - cost2) / 2
        
        # Build payoff matrix
        payoff_matrix_p1 = np.zeros((len(price_range), len(price_range)))
        payoff_matrix_p2 = np.zeros((len(price_range), len(price_range)))
        
        for i, p1 in enumerate(price_range):
            for j, p2 in enumerate(price_range):
                payoff1, payoff2 = calculate_payoff(p1, p2)
                payoff_matrix_p1[i, j] = payoff1
                payoff_matrix_p2[i, j] = payoff2
        
        # Find Nash equilibrium (pure strategy)
        nash_equilibria = []
        
        for i in range(len(price_range)):
            for j in range(len(price_range)):
                # Check if (i,j) is a Nash equilibrium
                # Player 1's best response to j
                p1_best_response = np.argmax(payoff_matrix_p1[:, j])
                # Player 2's best response to i
                p2_best_response = np.argmax(payoff_matrix_p2[i, :])
                
                if p1_best_response == i and p2_best_response == j:
                    nash_equilibria.append({
                        'player1_price': price_range[i],
                        'player2_price': price_range[j],
                        'player1_payoff': payoff_matrix_p1[i, j],
                        'player2_payoff': payoff_matrix_p2[i, j]
                    })
        
        result = {
            'equilibrium_type': 'pure_strategy' if nash_equilibria else 'mixed_strategy',
            'equilibria': nash_equilibria,
            'optimal_price': nash_equilibria[0]['player1_price'] if nash_equilibria else None,
            'expected_payoff': nash_equilibria[0]['player1_payoff'] if nash_equilibria else None
        }
        
        return result
    
    def _find_multi_player_nash_equilibrium(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find Nash equilibrium for multi-player game (simplified)"""
        
        players = game_data.get('players', [])
        n_players = len(players)
        
        # Simplified approach: assume symmetric players
        # Each player chooses price multiplier between 0.8 and 1.2
        
        def social_welfare_function(prices):
            """Calculate social welfare for given price vector"""
            base_value = 100
            costs = [base_value * p for p in prices]
            
            # Winner takes all - lowest price wins
            winner_idx = np.argmin(prices)
            winner_payoff = base_value - costs[winner_idx]
            
            # Social welfare = sum of all payoffs
            total_welfare = winner_payoff  # Others get 0
            return -total_welfare  # Negative for minimization
        
        # Find approximate equilibrium using optimization
        initial_guess = [1.0] * n_players
        bounds = [(0.8, 1.2) for _ in range(n_players)]
        
        result_opt = minimize(
            social_welfare_function,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        equilibrium_prices = result_opt.x
        
        return {
            'equilibrium_type': 'approximate',
            'optimal_prices': equilibrium_prices.tolist(),
            'optimal_price': float(np.mean(equilibrium_prices)),
            'social_welfare': -result_opt.fun,
            'convergence': result_opt.success
        }
    
    def full_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive competitive analysis"""
        try:
            # Extract competitors and features
            competitors = data.get('competitors', [])
            features = data.get('features', {})
            
            # Basic competitive analysis
            basic_analysis = self.analyze_competition(features, competitors)
            
            # Game-theoretic analysis
            game_theory_analysis = self.find_nash_equilibrium(data)
            
            # Market structure analysis
            market_structure = self._analyze_market_structure(competitors)
            
            # Strategic clustering
            strategic_clusters = self._perform_strategic_clustering(competitors)
            
            # Competitive scenarios
            scenarios = self._generate_competitive_scenarios(features, competitors)
            
            result = {
                'basic_analysis': basic_analysis,
                'game_theory': game_theory_analysis,
                'market_structure': market_structure,
                'strategic_clusters': strategic_clusters,
                'scenarios': scenarios,
                'summary': self._generate_analysis_summary(basic_analysis, game_theory_analysis)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Full competitive analysis error: {str(e)}")
            raise
    
    def _analyze_market_structure(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall market structure"""
        
        if not competitors:
            return {'structure_type': 'monopoly', 'concentration': 1.0}
        
        n_competitors = len(competitors)
        
        # Market shares
        market_shares = [c.get('market_share', 1/n_competitors) for c in competitors]
        
        # Herfindahl-Hirschman Index
        hhi = sum(share**2 for share in market_shares)
        
        # Determine market structure
        if hhi > 0.25:
            structure_type = 'highly_concentrated'
        elif hhi > 0.15:
            structure_type = 'moderately_concentrated'
        else:
            structure_type = 'competitive'
        
        # Calculate other metrics
        cr4 = sum(sorted(market_shares, reverse=True)[:4])  # 4-firm concentration ratio
        
        return {
            'structure_type': structure_type,
            'hhi': hhi,
            'cr4': cr4,
            'n_competitors': n_competitors,
            'market_shares': market_shares,
            'dominant_players': [i for i, share in enumerate(market_shares) if share > 0.2]
        }
    
    def _perform_strategic_clustering(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster competitors based on strategic characteristics"""
        
        if len(competitors) < 3:
            return {'clusters': [], 'n_clusters': 0}
        
        # Feature matrix for clustering
        features = []
        for comp in competitors:
            features.append([
                comp.get('win_rate', 0.5),
                comp.get('quality_score', 5) / 10,
                comp.get('avg_price_ratio', 1.0),
                comp.get('market_share', 0.1),
                comp.get('delivery_time', 30) / 90
            ])
        
        features_array = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # Determine optimal number of clusters
        n_clusters = min(4, max(2, len(competitors) // 3))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        clusters = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_competitors = [competitors[j] for j in cluster_indices]
            
            # Characterize cluster
            avg_win_rate = np.mean([comp.get('win_rate', 0.5) for comp in cluster_competitors])
            avg_quality = np.mean([comp.get('quality_score', 5) for comp in cluster_competitors])
            avg_price_ratio = np.mean([comp.get('avg_price_ratio', 1.0) for comp in cluster_competitors])
            
            # Determine cluster strategy
            if avg_price_ratio < 0.95 and avg_win_rate > 0.6:
                strategy = 'cost_leader'
            elif avg_quality > 7 and avg_price_ratio > 1.05:
                strategy = 'differentiator'
            elif 0.95 <= avg_price_ratio <= 1.05:
                strategy = 'market_follower'
            else:
                strategy = 'niche_player'
            
            clusters.append({
                'cluster_id': i,
                'strategy': strategy,
                'size': len(cluster_competitors),
                'avg_win_rate': avg_win_rate,
                'avg_quality': avg_quality,
                'avg_price_ratio': avg_price_ratio,
                'competitor_indices': cluster_indices.tolist()
            })
        
        return {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist()
        }
    
    def _generate_competitive_scenarios(self, features: Dict[str, Any], 
                                      competitors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate competitive scenarios for strategy testing"""
        
        scenarios = []
        
        # Scenario 1: Price war
        scenarios.append({
            'name': 'price_war',
            'description': 'Competitors engage in aggressive price cutting',
            'assumptions': 'All competitors reduce prices by 10-20%',
            'impact_on_win_probability': -0.15,
            'recommended_response': 'Focus on value differentiation, avoid price matching',
            'risk_level': 'high'
        })
        
        # Scenario 2: New entrant
        scenarios.append({
            'name': 'new_market_entrant',
            'description': 'Strong new competitor enters market',
            'assumptions': 'New player with superior technology/pricing',
            'impact_on_win_probability': -0.20,
            'recommended_response': 'Strengthen customer relationships, highlight experience',
            'risk_level': 'medium'
        })
        
        # Scenario 3: Market consolidation
        scenarios.append({
            'name': 'market_consolidation',
            'description': 'Major competitors merge or form alliances',
            'assumptions': '2-3 large players dominate market',
            'impact_on_win_probability': -0.10,
            'recommended_response': 'Find niche markets, consider partnerships',
            'risk_level': 'medium'
        })
        
        # Scenario 4: Technology disruption
        scenarios.append({
            'name': 'technology_disruption',
            'description': 'New technology changes competitive landscape',
            'assumptions': 'AI/automation reduces costs for tech-savvy competitors',
            'impact_on_win_probability': -0.25,
            'recommended_response': 'Invest in technology, upskill capabilities',
            'risk_level': 'high'
        })
        
        # Scenario 5: Economic downturn
        scenarios.append({
            'name': 'economic_downturn',
            'description': 'Economic conditions favor cost-conscious decisions',
            'assumptions': 'Clients prioritize price over quality',
            'impact_on_win_probability': self._calculate_downturn_impact(features),
            'recommended_response': 'Optimize costs, emphasize ROI in proposals',
            'risk_level': 'medium'
        })
        
        return scenarios
    
    def _calculate_downturn_impact(self, features: Dict[str, Any]) -> float:
        """Calculate impact of economic downturn based on current positioning"""
        
        price_competitiveness = features.get('price_competitiveness', 0.5)
        quality_score = features.get('quality_score', 5) / 10
        
        # If we're already price competitive, less negative impact
        if price_competitiveness > 0.7:
            return -0.05
        elif price_competitiveness > 0.5:
            return -0.10
        else:
            return -0.20
    
    def _generate_analysis_summary(self, basic_analysis: Dict[str, Any], 
                                 game_theory: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of competitive analysis"""
        
        # Extract key insights
        market_position = basic_analysis.get('market_position', {})
        competitive_intensity = basic_analysis.get('competitive_intensity', {})
        
        # Generate key findings
        findings = []
        
        position = market_position.get('position', 'unknown')
        if position == 'market_leader':
            findings.append("Strong competitive position across key metrics")
        elif position == 'weak_competitor':
            findings.append("Competitive position needs improvement")
        
        intensity_level = competitive_intensity.get('level', 'medium')
        if intensity_level in ['high', 'very_high']:
            findings.append("High competitive intensity requires strategic focus")
        
        # Nash equilibrium insights
        if game_theory.get('optimal_price'):
            findings.append(f"Game theory suggests optimal pricing around {game_theory['optimal_price']:.2f}")
        
        # Overall recommendation
        relative_strength = market_position.get('relative_strength', 0)
        if relative_strength > 0.2:
            overall_recommendation = "Maintain competitive advantages while monitoring market changes"
        elif relative_strength > -0.2:
            overall_recommendation = "Focus on selective improvements to strengthen position"
        else:
            overall_recommendation = "Significant strategic changes needed to improve competitiveness"
        
        return {
            'key_findings': findings,
            'overall_recommendation': overall_recommendation,
            'competitive_score': market_position.get('competitive_score', 0.5),
            'risk_level': basic_analysis.get('risk_assessment', {}).get('overall_risk', 'medium'),
            'priority_actions': [rec['recommendation'] for rec in 
                               basic_analysis.get('strategic_recommendations', [])[:3]]
        }
    
    def _analyze_quality_distribution(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality score distribution among competitors"""
        
        quality_scores = [p['avg_quality_score'] for p in profiles]
        
        return {
            'mean': np.mean(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores),
            'percentiles': {
                '25th': np.percentile(quality_scores, 25),
                '50th': np.percentile(quality_scores, 50),
                '75th': np.percentile(quality_scores, 75)
            }
        }
    
    def _analyze_pricing_strategies(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of pricing strategies"""
        
        strategies = [p['pricing_strategy'] for p in profiles]
        strategy_counts = {}
        
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'distribution': strategy_counts,
            'dominant_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0],
            'strategy_diversity': len(strategy_counts)
        }
    
    def _calculate_market_concentration(self, profiles: List[Dict[str, Any]]) -> float:
        """Calculate market concentration ratio"""
        
        market_shares = [p['market_share'] for p in profiles]
        sorted_shares = sorted(market_shares, reverse=True)
        
        # Top 3 concentration ratio
        top3_concentration = sum(sorted_shares[:3])
        
        return min(1.0, top3_concentration)