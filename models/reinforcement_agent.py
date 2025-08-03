"""
Reinforcement Learning Agent for Proposal Optimization
Implements adaptive learning strategies for dynamic bidding optimization
Based on the RL methodology described in the research paper
"""

import numpy as np
import pandas as pd
from collections import deque, defaultdict
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional
import random
from abc import ABC, abstractmethod

# RL Libraries
try:
    import gymnasium as gym
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    HAS_STABLE_BASELINES = True
except ImportError:
    HAS_STABLE_BASELINES = False
    logging.warning("stable-baselines3 not available, using custom RL implementation")

logger = logging.getLogger(__name__)

class BiddingEnvironment:
    """
    Custom bidding environment for reinforcement learning
    Simulates the competitive bidding process
    """
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        """Initialize the bidding environment"""
        self.historical_data = historical_data
        self.current_state = None
        self.episode_length = 0
        self.max_episode_length = 100
        self.action_space_size = 21  # 21 discrete price multipliers (0.8 to 1.2)
        self.state_space_size = 15   # Number of state features
        
        # Environment parameters
        self.base_reward = 100
        self.cost_penalty = 0.1
        self.competition_factor = 0.2
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.episode_length = 0
        self.current_state = self._generate_initial_state()
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        
        # Convert action to price multiplier
        price_multiplier = 0.8 + (action / 20.0) * 0.4  # Maps 0-20 to 0.8-1.2
        
        # Simulate competitive bidding
        reward = self._calculate_reward(self.current_state, price_multiplier)
        
        # Update state (next bidding scenario)
        next_state = self._generate_next_state()
        
        # Check if episode is done
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        
        # Additional info
        info = {
            'price_multiplier': price_multiplier,
            'win_probability': self._calculate_win_probability(self.current_state, price_multiplier),
            'episode_length': self.episode_length
        }
        
        self.current_state = next_state
        return next_state, reward, done, info
    
    def _generate_initial_state(self) -> np.ndarray:
        """Generate initial state for episode"""
        
        # State features: [quality_score, delivery_time, num_competitors, complexity, 
        #                 market_share, historical_win_rate, avg_competitor_quality,
        #                 avg_competitor_price, seasonal_factor, urgency,
        #                 client_relationship, past_performance, risk_level,
        #                 market_volatility, economic_indicator]
        
        state = np.array([
            np.random.uniform(3, 10),    # quality_score
            np.random.uniform(5, 60),    # delivery_time
            np.random.randint(2, 10),    # num_competitors
            np.random.uniform(1, 10),    # complexity
            np.random.uniform(0.05, 0.3), # market_share
            np.random.uniform(0.2, 0.8), # historical_win_rate
            np.random.uniform(4, 9),     # avg_competitor_quality
            np.random.uniform(0.9, 1.1), # avg_competitor_price
            np.random.uniform(0.8, 1.2), # seasonal_factor
            np.random.uniform(0, 1),     # urgency
            np.random.uniform(0, 1),     # client_relationship
            np.random.uniform(0.3, 1.0), # past_performance
            np.random.uniform(0, 1),     # risk_level
            np.random.uniform(0.8, 1.2), # market_volatility
            np.random.uniform(0.9, 1.1)  # economic_indicator
        ])
        
        return state
    
    def _generate_next_state(self) -> np.ndarray:
        """Generate next state with some correlation to current state"""
        
        # Slightly modify current state to simulate market evolution
        next_state = self.current_state.copy()
        
        # Add some noise to simulate market changes
        noise = np.random.normal(0, 0.05, size=next_state.shape)
        next_state += noise
        
        # Ensure bounds
        next_state = np.clip(next_state, 0, None)
        
        # Some features have specific bounds
        next_state[0] = np.clip(next_state[0], 1, 10)    # quality_score
        next_state[1] = np.clip(next_state[1], 1, 90)    # delivery_time
        next_state[2] = np.clip(next_state[2], 1, 15)    # num_competitors
        next_state[4] = np.clip(next_state[4], 0, 1)     # market_share
        next_state[5] = np.clip(next_state[5], 0, 1)     # historical_win_rate
        
        return next_state
    
    def _calculate_reward(self, state: np.ndarray, price_multiplier: float) -> float:
        """Calculate reward for the given state and action"""
        
        # Extract state features
        quality_score = state[0]
        delivery_time = state[1]
        num_competitors = state[2]
        complexity = state[3]
        avg_competitor_price = state[7]
        
        # Calculate win probability
        win_prob = self._calculate_win_probability(state, price_multiplier)
        
        # Base value of the contract
        contract_value = self.base_reward * (1 + complexity / 10)
        
        # Cost calculation
        cost = contract_value * price_multiplier * 0.8  # Assume 80% cost ratio
        
        # Profit if win
        profit = contract_value - cost
        
        # Expected reward = win_probability * profit
        expected_reward = win_prob * profit
        
        # Penalties for extreme actions
        if price_multiplier < 0.85 or price_multiplier > 1.15:
            expected_reward *= 0.8  # Penalty for extreme pricing
        
        return expected_reward
    
    def _calculate_win_probability(self, state: np.ndarray, price_multiplier: float) -> float:
        """Calculate probability of winning given state and price"""
        
        quality_score = state[0]
        delivery_time = state[1]
        num_competitors = state[2]
        avg_competitor_price = state[7]
        client_relationship = state[10]
        past_performance = state[11]
        
        # Quality factor (0-1)
        quality_factor = quality_score / 10.0
        
        # Delivery factor (0-1, faster is better)
        delivery_factor = max(0, 1 - (delivery_time - 5) / 85.0)
        
        # Price competitiveness factor
        price_competitiveness = max(0, 1 - abs(price_multiplier - avg_competitor_price))
        
        # Competition factor (more competitors = lower win probability)
        competition_factor = max(0.1, 1 - (num_competitors - 2) / 13.0)
        
        # Relationship factor
        relationship_factor = client_relationship
        
        # Performance factor
        performance_factor = past_performance
        
        # Combine factors with weights
        win_prob = (quality_factor * 0.25 +
                   delivery_factor * 0.15 +
                   price_competitiveness * 0.30 +
                   competition_factor * 0.10 +
                   relationship_factor * 0.10 +
                   performance_factor * 0.10)
        
        return np.clip(win_prob, 0.01, 0.99)

class QLearningAgent:
    """
    Q-Learning agent for discrete action spaces
    Custom implementation when stable-baselines3 is not available
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.995):
        """Initialize Q-Learning agent"""
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Q-table approximation using dictionary
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
    def _discretize_state(self, state: np.ndarray) -> str:
        """Convert continuous state to discrete representation"""
        
        # Simple discretization - divide each feature into bins
        discretized = []
        for i, feature in enumerate(state):
            if i == 0:  # quality_score (1-10)
                bins = np.linspace(1, 10, 5)
            elif i == 1:  # delivery_time (1-90)
                bins = np.linspace(1, 90, 5)
            elif i == 2:  # num_competitors (1-15)
                bins = np.linspace(1, 15, 4)
            else:  # Other features (0-1 typically)
                bins = np.linspace(0, 1, 3)
            
            discretized.append(str(np.digitize(feature, bins)))
        
        return '_'.join(discretized)
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_key = self._discretize_state(state)
        q_values = self.q_table[state_key]
        return np.argmax(q_values)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Experience replay training"""
        
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self._discretize_state(state)
            next_state_key = self._discretize_state(next_state)
            
            target = reward
            if not done:
                target += self.discount_factor * np.max(self.q_table[next_state_key])
            
            self.q_table[state_key][action] += self.learning_rate * (
                target - self.q_table[state_key][action]
            )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class ReinforcementAgent:
    """
    Main reinforcement learning agent for proposal optimization
    Combines multiple RL approaches and adaptive learning strategies
    """
    
    def __init__(self):
        """Initialize the reinforcement learning agent"""
        
        self.environment = None
        self.agent = None
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = {}
        
        # Multi-armed bandit for strategy selection
        self.strategy_bandits = {}
        self.strategy_counts = defaultdict(int)
        self.strategy_rewards = defaultdict(list)
        
        # Online learning parameters
        self.learning_rate = 0.01
        self.adaptation_window = 100
        self.recent_performance = deque(maxlen=self.adaptation_window)
        
        logger.info("Reinforcement learning agent initialized")
    
    def initialize_with_data(self, historical_data: pd.DataFrame):
        """Initialize agent with historical bidding data"""
        try:
            self.environment = BiddingEnvironment(historical_data)
            
            # Initialize appropriate agent based on available libraries
            if HAS_STABLE_BASELINES:
                self._initialize_stable_baselines_agent()
            else:
                self._initialize_custom_agent()
            
            logger.info("RL agent initialized with historical data")
            
        except Exception as e:
            logger.error(f"RL agent initialization error: {str(e)}")
    
    def _initialize_stable_baselines_agent(self):
        """Initialize using stable-baselines3"""
        
        # Create a simple gym environment wrapper
        class GymBiddingEnv(gym.Env):
            def __init__(self, bidding_env):
                super(GymBiddingEnv, self).__init__()
                self.bidding_env = bidding_env
                self.action_space = gym.spaces.Discrete(bidding_env.action_space_size)
                self.observation_space = gym.spaces.Box(
                    low=0, high=np.inf, shape=(bidding_env.state_space_size,), dtype=np.float32
                )
            
            def reset(self):
                return self.bidding_env.reset().astype(np.float32)
            
            def step(self, action):
                next_state, reward, done, info = self.bidding_env.step(action)
                return next_state.astype(np.float32), reward, done, info
        
        # Create gym environment
        gym_env = GymBiddingEnv(self.environment)
        
        # Initialize DQN agent
        self.agent = DQN(
            "MlpPolicy",
            gym_env,
            learning_rate=0.0005,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.95,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=0
        )
    
    def _initialize_custom_agent(self):
        """Initialize custom Q-learning agent"""
        
        self.agent = QLearningAgent(
            state_size=self.environment.state_space_size,
            action_size=self.environment.action_space_size,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995
        )
    
    def train(self, data: pd.DataFrame, episodes: int = 1000) -> Dict[str, Any]:
        """Train the RL agent"""
        try:
            if self.environment is None:
                self.initialize_with_data(data)
            
            logger.info(f"Starting RL training for {episodes} episodes...")
            
            if HAS_STABLE_BASELINES and hasattr(self.agent, 'learn'):
                # Train using stable-baselines3
                self.agent.learn(total_timesteps=episodes * 100)
                
            else:
                # Train using custom agent
                episode_rewards = []
                
                for episode in range(episodes):
                    state = self.environment.reset()
                    total_reward = 0
                    done = False
                    
                    while not done:
                        action = self.agent.act(state)
                        next_state, reward, done, info = self.environment.step(action)
                        
                        self.agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        total_reward += reward
                    
                    # Train agent with experience replay
                    self.agent.replay()
                    
                    episode_rewards.append(total_reward)
                    
                    # Log progress
                    if episode % 100 == 0:
                        avg_reward = np.mean(episode_rewards[-100:])
                        logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
                
                self.training_history = episode_rewards
            
            self.is_trained = True
            
            # Calculate performance metrics
            self.performance_metrics = self._calculate_training_metrics()
            
            logger.info("RL training completed successfully")
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"RL training error: {str(e)}")
            raise
    
    def get_recommendation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get RL agent recommendation for bidding strategy"""
        try:
            if not self.is_trained:
                return self._get_heuristic_recommendation(features)
            
            # Convert features to state representation
            state = self._features_to_state(features)
            
            if HAS_STABLE_BASELINES and hasattr(self.agent, 'predict'):
                # Use stable-baselines3 agent
                action, _ = self.agent.predict(state, deterministic=True)
                action = int(action)
            else:
                # Use custom agent
                action = self.agent.act(state)
            
            # Convert action to price multiplier
            price_multiplier = 0.8 + (action / 20.0) * 0.4
            
            # Calculate additional metrics
            win_probability = self.environment._calculate_win_probability(state, price_multiplier)
            expected_reward = self.environment._calculate_reward(state, price_multiplier)
            
            # Get strategy explanation
            strategy_explanation = self._explain_strategy(features, action, price_multiplier)
            
            # Multi-armed bandit recommendation
            bandit_recommendation = self._get_bandit_recommendation(features)
            
            recommendation = {
                'action': f'price_multiplier_{price_multiplier:.3f}',
                'price_multiplier': float(price_multiplier),
                'win_probability': float(win_probability),
                'expected_reward': float(expected_reward),
                'confidence': self._calculate_recommendation_confidence(),
                'strategy_explanation': strategy_explanation,
                'bandit_recommendation': bandit_recommendation,
                'risk_assessment': self._assess_strategy_risk(price_multiplier, features)
            }
            
            # Update recent performance tracking
            self.recent_performance.append({
                'recommendation': recommendation,
                'timestamp': pd.Timestamp.now()
            })
            
            return recommendation
            
        except Exception as e:
            logger.error(f"RL recommendation error: {str(e)}")
            return self._get_heuristic_recommendation(features)
    
    def _features_to_state(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to state array"""
        
        state = np.array([
            features.get('quality_score', 5),
            features.get('delivery_time', 30),
            features.get('num_bids', 5),
            features.get('complexity', 5),
            features.get('market_share', 0.1),
            features.get('provider_experience', 10) / 50.0,  # Normalize
            features.get('avg_competitor_quality', 5),
            features.get('avg_competitor_price', 1.0),
            features.get('seasonal_factor', 1.0),
            features.get('urgency', 0.5),
            features.get('client_relationship', 0.5),
            features.get('past_performance', 0.7),
            features.get('risk_level', 0.5),
            features.get('market_volatility', 1.0),
            features.get('economic_indicator', 1.0)
        ])
        
        return state
    
    def _get_heuristic_recommendation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback heuristic recommendation when RL agent is not trained"""
        
        # Simple heuristic strategy
        quality_score = features.get('quality_score', 5)
        num_competitors = features.get('num_bids', 5)
        complexity = features.get('complexity', 5)
        
        # Base price multiplier
        base_multiplier = 1.0
        
        # Adjust based on quality
        if quality_score > 7:
            base_multiplier += 0.05  # Premium for high quality
        elif quality_score < 4:
            base_multiplier -= 0.05  # Discount for low quality
        
        # Adjust based on competition
        if num_competitors > 7:
            base_multiplier -= 0.03  # More aggressive pricing
        elif num_competitors < 3:
            base_multiplier += 0.03  # Less aggressive pricing
        
        # Adjust based on complexity
        if complexity > 7:
            base_multiplier += 0.02  # Higher price for complex projects
        
        price_multiplier = np.clip(base_multiplier, 0.8, 1.2)
        
        return {
            'action': f'heuristic_price_multiplier_{price_multiplier:.3f}',
            'price_multiplier': price_multiplier,
            'win_probability': 0.5,  # Default estimate
            'expected_reward': 50.0,  # Default estimate
            'confidence': 0.3,  # Low confidence for heuristic
            'strategy_explanation': 'Heuristic-based recommendation (RL agent not trained)',
            'bandit_recommendation': None,
            'risk_assessment': 'medium'
        }
    
    def _explain_strategy(self, features: Dict[str, Any], action: int, price_multiplier: float) -> str:
        """Explain the reasoning behind the strategy"""
        
        explanations = []
        
        # Price positioning explanation
        if price_multiplier < 0.9:
            explanations.append("Aggressive pricing strategy to maximize win probability")
        elif price_multiplier > 1.1:
            explanations.append("Premium pricing strategy focusing on value and margins")
        else:
            explanations.append("Balanced pricing strategy considering market conditions")
        
        # Competition factor
        num_competitors = features.get('num_bids', 5)
        if num_competitors > 6:
            explanations.append(f"High competition ({num_competitors} bidders) influences pricing")
        
        # Quality factor
        quality_score = features.get('quality_score', 5)
        if quality_score > 7:
            explanations.append("High quality score supports premium positioning")
        elif quality_score < 4:
            explanations.append("Lower quality score requires competitive pricing")
        
        return "; ".join(explanations)
    
    def _get_bandit_recommendation(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get recommendation from multi-armed bandit for strategy selection"""
        
        try:
            # Define strategy arms
            strategies = ['aggressive', 'balanced', 'premium', 'value_based']
            
            # Calculate UCB (Upper Confidence Bound) for each strategy
            total_plays = sum(self.strategy_counts[s] for s in strategies)
            
            if total_plays == 0:
                # Random selection for cold start
                selected_strategy = random.choice(strategies)
            else:
                ucb_values = {}
                for strategy in strategies:
                    if self.strategy_counts[strategy] == 0:
                        ucb_values[strategy] = float('inf')
                    else:
                        mean_reward = np.mean(self.strategy_rewards[strategy])
                        confidence = np.sqrt(2 * np.log(total_plays) / self.strategy_counts[strategy])
                        ucb_values[strategy] = mean_reward + confidence
                
                selected_strategy = max(ucb_values.items(), key=lambda x: x[1])[0]
            
            # Convert strategy to price multiplier
            strategy_multipliers = {
                'aggressive': 0.85,
                'balanced': 1.0,
                'premium': 1.15,
                'value_based': 0.95
            }
            
            return {
                'strategy': selected_strategy,
                'price_multiplier': strategy_multipliers[selected_strategy],
                'confidence': self.strategy_counts[selected_strategy] / max(1, total_plays),
                'exploration_factor': ucb_values.get(selected_strategy, 0) if total_plays > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Bandit recommendation error: {str(e)}")
            return None
    
    def update_strategy_performance(self, strategy: str, reward: float):
        """Update multi-armed bandit with strategy performance"""
        
        self.strategy_counts[strategy] += 1
        self.strategy_rewards[strategy].append(reward)
        
        # Keep only recent rewards (sliding window)
        if len(self.strategy_rewards[strategy]) > 100:
            self.strategy_rewards[strategy] = self.strategy_rewards[strategy][-100:]
    
    def _calculate_recommendation_confidence(self) -> float:
        """Calculate confidence in the recommendation"""
        
        if not self.is_trained:
            return 0.3
        
        # Base confidence from training performance
        base_confidence = 0.7
        
        # Adjust based on recent performance
        if len(self.recent_performance) > 10:
            recent_rewards = [p.get('expected_reward', 0) for p in self.recent_performance]
            reward_stability = 1 - (np.std(recent_rewards) / (np.mean(recent_rewards) + 1))
            base_confidence *= reward_stability
        
        return np.clip(base_confidence, 0.1, 0.95)
    
    def _assess_strategy_risk(self, price_multiplier: float, features: Dict[str, Any]) -> str:
        """Assess risk level of the recommended strategy"""
        
        risk_factors = 0
        
        # Extreme pricing risk
        if price_multiplier < 0.85 or price_multiplier > 1.15:
            risk_factors += 1
        
        # High competition risk
        if features.get('num_bids', 5) > 7:
            risk_factors += 1
        
        # Low quality with premium pricing risk
        if price_multiplier > 1.05 and features.get('quality_score', 5) < 6:
            risk_factors += 1
        
        # Market volatility risk
        if features.get('market_volatility', 1.0) > 1.1:
            risk_factors += 1
        
        if risk_factors >= 3:
            return 'high'
        elif risk_factors >= 2:
            return 'medium'
        else:
            return 'low'
    
    def optimize_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize bidding strategy using RL and game theory"""
        try:
            # Get RL recommendation
            rl_recommendation = self.get_recommendation(data.get('features', {}))
            
            # Simulate different strategies
            strategies = self._simulate_strategies(data)
            
            # Multi-objective optimization
            optimal_strategy = self._multi_objective_optimization(strategies)
            
            # Risk-adjusted recommendation
            risk_adjusted = self._risk_adjust_strategy(optimal_strategy, data)
            
            return {
                'rl_recommendation': rl_recommendation,
                'simulated_strategies': strategies,
                'optimal_strategy': optimal_strategy,
                'risk_adjusted_strategy': risk_adjusted,
                'meta_recommendation': self._generate_meta_recommendation(
                    rl_recommendation, optimal_strategy, risk_adjusted
                )
            }
            
        except Exception as e:
            logger.error(f"Strategy optimization error: {str(e)}")
            raise
    
    def _simulate_strategies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate different bidding strategies"""
        
        strategies = []
        base_features = data.get('features', {})
        
        # Strategy variations
        price_multipliers = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
        
        for multiplier in price_multipliers:
            # Create modified features
            modified_features = base_features.copy()
            
            # Simulate strategy
            state = self._features_to_state(modified_features)
            win_prob = self.environment._calculate_win_probability(state, multiplier)
            expected_reward = self.environment._calculate_reward(state, multiplier)
            
            strategies.append({
                'price_multiplier': multiplier,
                'win_probability': win_prob,
                'expected_reward': expected_reward,
                'risk_score': self._calculate_strategy_risk_score(multiplier, modified_features),
                'strategy_type': self._classify_strategy_type(multiplier)
            })
        
        return strategies
    
    def _multi_objective_optimization(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform multi-objective optimization of strategies"""
        
        # Objectives: maximize expected reward, maximize win probability, minimize risk
        best_strategy = None
        best_score = -float('inf')
        
        for strategy in strategies:
            # Weighted score combining objectives
            score = (strategy['expected_reward'] * 0.4 +
                    strategy['win_probability'] * 100 * 0.4 +
                    (1 - strategy['risk_score']) * 100 * 0.2)
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        if best_strategy:
            best_strategy['optimization_score'] = best_score
        
        return best_strategy
    
    def _risk_adjust_strategy(self, strategy: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply risk adjustment to strategy"""
        
        if not strategy:
            return strategy
        
        adjusted_strategy = strategy.copy()
        
        # Risk factors
        market_volatility = data.get('features', {}).get('market_volatility', 1.0)
        competitive_intensity = data.get('competitive_intensity', 0.5)
        
        # Adjust price multiplier based on risk
        risk_adjustment = 0
        
        if market_volatility > 1.1:
            risk_adjustment -= 0.02  # More conservative in volatile markets
        
        if competitive_intensity > 0.7:
            risk_adjustment -= 0.01  # More aggressive in high competition
        
        adjusted_multiplier = strategy['price_multiplier'] + risk_adjustment
        adjusted_strategy['price_multiplier'] = np.clip(adjusted_multiplier, 0.8, 1.2)
        adjusted_strategy['risk_adjustment'] = risk_adjustment
        
        return adjusted_strategy
    
    def _generate_meta_recommendation(self, rl_rec: Dict[str, Any], 
                                    optimal_strategy: Dict[str, Any],
                                    risk_adjusted: Dict[str, Any]) -> Dict[str, Any]:
        """Generate meta-recommendation combining all approaches"""
        
        # Weight different recommendations
        rl_weight = 0.4 if self.is_trained else 0.2
        optimal_weight = 0.4
        risk_weight = 0.2 if self.is_trained else 0.4
        
        # Weighted average of price multipliers
        weighted_multiplier = (
            rl_rec['price_multiplier'] * rl_weight +
            optimal_strategy.get('price_multiplier', 1.0) * optimal_weight +
            risk_adjusted.get('price_multiplier', 1.0) * risk_weight
        )
        
        return {
            'recommended_price_multiplier': weighted_multiplier,
            'confidence': (rl_rec['confidence'] + 0.8) / 2,  # Average with optimal confidence
            'strategy_consensus': self._assess_strategy_consensus(rl_rec, optimal_strategy, risk_adjusted),
            'recommendation_source': 'meta_ensemble'
        }
    
    def _calculate_training_metrics(self) -> Dict[str, Any]:
        """Calculate training performance metrics"""
        
        if not self.training_history:
            return {}
        
        rewards = np.array(self.training_history)
        
        return {
            'total_episodes': len(rewards),
            'final_average_reward': np.mean(rewards[-100:]),
            'best_episode_reward': np.max(rewards),
            'learning_curve_slope': self._calculate_learning_slope(rewards),
            'convergence_episode': self._find_convergence_episode(rewards),
            'training_stability': 1 - (np.std(rewards[-100:]) / (np.mean(rewards[-100:]) + 1))
        }
    
    def _calculate_learning_slope(self, rewards: np.ndarray) -> float:
        """Calculate learning curve slope"""
        
        if len(rewards) < 100:
            return 0
        
        # Compare first 100 and last 100 episodes
        early_mean = np.mean(rewards[:100])
        late_mean = np.mean(rewards[-100:])
        
        return (late_mean - early_mean) / len(rewards)
    
    def _find_convergence_episode(self, rewards: np.ndarray) -> int:
        """Find episode where training converged"""
        
        if len(rewards) < 200:
            return len(rewards)
        
        window_size = 50
        threshold = 0.05  # 5% change threshold
        
        for i in range(window_size, len(rewards) - window_size):
            before_window = rewards[i-window_size:i]
            after_window = rewards[i:i+window_size]
            
            if np.abs(np.mean(after_window) - np.mean(before_window)) < threshold:
                return i
        
        return len(rewards)
    
    def _calculate_strategy_risk_score(self, price_multiplier: float, features: Dict[str, Any]) -> float:
        """Calculate risk score for a strategy"""
        
        risk_score = 0
        
        # Price extremity risk
        if price_multiplier < 0.9 or price_multiplier > 1.1:
            risk_score += 0.3
        
        # Market condition risk
        market_volatility = features.get('market_volatility', 1.0)
        if market_volatility > 1.1:
            risk_score += 0.2
        
        # Competition risk
        num_competitors = features.get('num_bids', 5)
        if num_competitors > 7:
            risk_score += 0.2
        
        # Quality-price mismatch risk
        quality_score = features.get('quality_score', 5)
        if price_multiplier > 1.05 and quality_score < 6:
            risk_score += 0.3
        
        return min(1.0, risk_score)
    
    def _classify_strategy_type(self, price_multiplier: float) -> str:
        """Classify strategy type based on price multiplier"""
        
        if price_multiplier < 0.9:
            return 'aggressive'
        elif price_multiplier < 0.95:
            return 'competitive'
        elif price_multiplier < 1.05:
            return 'balanced'
        elif price_multiplier < 1.1:
            return 'premium'
        else:
            return 'luxury'
    
    def _assess_strategy_consensus(self, rl_rec: Dict[str, Any], 
                                 optimal: Dict[str, Any], risk_adj: Dict[str, Any]) -> str:
        """Assess consensus among different recommendation approaches"""
        
        multipliers = [
            rl_rec['price_multiplier'],
            optimal.get('price_multiplier', 1.0),
            risk_adj.get('price_multiplier', 1.0)
        ]
        
        std_dev = np.std(multipliers)
        
        if std_dev < 0.02:
            return 'strong_consensus'
        elif std_dev < 0.05:
            return 'moderate_consensus'
        else:
            return 'weak_consensus'
    
    def save_agent(self, filepath: str):
        """Save trained RL agent"""
        try:
            if HAS_STABLE_BASELINES and hasattr(self.agent, 'save'):
                self.agent.save(filepath)
            else:
                # Save custom agent
                agent_data = {
                    'q_table': dict(self.agent.q_table),
                    'training_history': self.training_history,
                    'performance_metrics': self.performance_metrics,
                    'strategy_counts': dict(self.strategy_counts),
                    'strategy_rewards': dict(self.strategy_rewards)
                }
                
                with open(f"{filepath}.pkl", 'wb') as f:
                    pickle.dump(agent_data, f)
            
            logger.info(f"RL agent saved to {filepath}")
            
        except Exception as e:
            logger.error(f"RL agent save error: {str(e)}")
    
    def load_agent(self, filepath: str):
        """Load trained RL agent"""
        try:
            if HAS_STABLE_BASELINES:
                self.agent = DQN.load(filepath)
            else:
                # Load custom agent
                with open(f"{filepath}.pkl", 'rb') as f:
                    agent_data = pickle.load(f)
                
                # Restore agent state
                self.agent.q_table = defaultdict(lambda: np.zeros(self.agent.action_size))
                self.agent.q_table.update(agent_data['q_table'])
                self.training_history = agent_data['training_history']
                self.performance_metrics = agent_data['performance_metrics']
                self.strategy_counts = defaultdict(int)
                self.strategy_counts.update(agent_data['strategy_counts'])
                self.strategy_rewards = defaultdict(list)
                self.strategy_rewards.update(agent_data['strategy_rewards'])
            
            self.is_trained = True
            logger.info(f"RL agent loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"RL agent load error: {str(e)}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the RL agent"""
        return {
            'is_trained': self.is_trained,
            'agent_type': 'stable_baselines_dqn' if HAS_STABLE_BASELINES else 'custom_qlearning',
            'training_episodes': len(self.training_history),
            'performance_metrics': self.performance_metrics,
            'strategy_counts': dict(self.strategy_counts),
            'recent_performance_window': len(self.recent_performance)
        }
                    