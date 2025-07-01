# reinforcement_learning.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class BiddingState:
    """Represents the state of a bidding environment"""
    project_features: np.ndarray
    competition_features: np.ndarray
    provider_features: np.ndarray
    market_conditions: np.ndarray
    time_remaining: float
    budget_constraints: np.ndarray

@dataclass
class BiddingAction:
    """Represents a bidding action"""
    price_multiplier: float  # Multiplier for base price estimate
    delivery_time: int
    quality_emphasis: float  # How much to emphasize quality in proposal

class BiddingEnvironment:
    """
    Simulated bidding environment for training RL agents
    Implements the competitive dynamics from the paper
    """
    
    def __init__(self, historical_data: List[Dict], max_competitors: int = 10):
        self.historical_data = historical_data
        self.max_competitors = max_competitors
        self.current_project = None
        self.competitors = []
        self.market_conditions = {}
        self.reset()
    
    def reset(self) -> BiddingState:
        """Reset environment to initial state"""
        # Sample a random project from historical data
        self.current_project = random.choice(self.historical_data)
        
        # Generate competitors based on historical patterns
        self._generate_competitors()
        
        # Calculate market conditions
        self._update_market_conditions()
        
        return self._get_state()
    
    def _generate_competitors(self):
        """Generate competitor profiles based on historical data"""
        num_competitors = random.randint(2, self.max_competitors)
        self.competitors = []
        
        for _ in range(num_competitors):
            # Sample competitor characteristics from historical distribution
            competitor = {
                'reputation_score': random.uniform(0.3, 1.0),
                'success_rate': random.uniform(0.1, 0.8),
                'avg_delivery_time': random.randint(10, 90),
                'quality_score': random.uniform(0.5, 1.0),
                'price_aggressiveness': random.uniform(0.7, 1.3)  # Multiplier for competitive pricing
            }
            self.competitors.append(competitor)
    
    def _update_market_conditions(self):
        """Update market conditions based on current competition"""
        competitor_scores = [c['reputation_score'] for c in self.competitors]
        
        self.market_conditions = {
            'competition_intensity': len(self.competitors) / self.max_competitors,
            'avg_competitor_reputation': np.mean(competitor_scores),
            'reputation_variance': np.var(competitor_scores),
            'price_pressure': np.mean([c['price_aggressiveness'] for c in self.competitors])
        }
    
    def _get_state(self) -> BiddingState:
        """Get current state representation"""
        # Project features
        project_features = np.array([
            self.current_project.get('complexity', 5) / 10.0,
            self.current_project.get('budget_max', 100000) / 1000000.0,
            (self.current_project.get('budget_max', 100000) - 
             self.current_project.get('budget_min', 50000)) / 1000000.0,
        ])
        
        # Competition features
        competition_features = np.array([
            self.market_conditions['competition_intensity'],
            self.market_conditions['avg_competitor_reputation'],
            self.market_conditions['reputation_variance'],
            self.market_conditions['price_pressure']
        ])
        
        # Provider features (agent's characteristics)
        provider_features = np.array([0.7, 0.6, 0.8])  # reputation, success_rate, quality
        
        # Market conditions
        market_features = np.array([
            random.uniform(0, 1),  # market volatility
            random.uniform(0, 1),  # demand level
        ])
        
        # Time remaining (normalized)
        time_remaining = random.uniform(0.1, 1.0)
        
        # Budget constraints
        budget_constraints = np.array([
            self.current_project.get('budget_min', 50000) / 1000000.0,
            self.current_project.get('budget_max', 100000) / 1000000.0
        ])
        
        return BiddingState(
            project_features=project_features,
            competition_features=competition_features,
            provider_features=provider_features,
            market_conditions=market_features,
            time_remaining=time_remaining,
            budget_constraints=budget_constraints
        )
    
    def step(self, action: BiddingAction) -> Tuple[BiddingState, float, bool, Dict]:
        """
        Execute action and return next state, reward, done flag, and info
        """
        # Calculate agent's bid
        base_price = self.current_project.get('budget_max', 100000) * 0.7  # Conservative base
        agent_price = base_price * action.price_multiplier
        
        # Generate competitor bids
        competitor_bids = []
        for competitor in self.competitors:
            comp_price = base_price * competitor['price_aggressiveness'] * random.uniform(0.8, 1.2)
            competitor_bids.append({
                'price': comp_price,
                'delivery_time': competitor['avg_delivery_time'],
                'quality_score': competitor['quality_score'],
                'reputation': competitor['reputation_score']
            })
        
        # Determine winner using scoring function
        agent_score = self._calculate_bid_score(agent_price, action.delivery_time, 
                                              action.quality_emphasis, is_agent=True)
        
        competitor_scores = []
        for bid in competitor_bids:
            score = self._calculate_bid_score(bid['price'], bid['delivery_time'], 
                                            bid['quality_score'], is_agent=False,
                                            reputation=bid['reputation'])
            competitor_scores.append(score)
        
        # Agent wins if they have the highest score
        agent_wins = agent_score > max(competitor_scores) if competitor_scores else True
        
        # Calculate reward
        reward = self._calculate_reward(agent_wins, agent_price, action, competitor_bids)
        
        # Episode ends after each bid
        done = True
        
        info = {
            'agent_wins': agent_wins,
            'agent_price': agent_price,
            'agent_score': agent_score,
            'num_competitors': len(competitor_bids),
            'winning_price': min([bid['price'] for bid in competitor_bids] + [agent_price]) if agent_wins else min([bid['price'] for bid in competitor_bids])
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_bid_score(self, price: float, delivery_time: int, quality: float, 
                           is_agent: bool, reputation: float = 0.7) -> float:
        """
        Calculate bid score based on multiple factors
        Higher score = better bid
        """
        budget_max = self.current_project.get('budget_max', 100000)
        
        # Price score (lower price = higher score, but not too low)
        price_ratio = price / budget_max
        if price_ratio < 0.5:
            price_score = 0.3  # Penalty for too low pricing (sustainability concern)
        else:
            price_score = max(0, 1 - (price_ratio - 0.5) * 2)
        
        # Delivery time score (faster = better, but realistic)
        max_delivery = 90
        delivery_score = max(0, 1 - delivery_time / max_delivery)
        
        # Quality score
        quality_score = quality
        
        # Reputation score
        reputation_score = reputation
        
        # Weighted combination
        total_score = (0.3 * price_score + 
                      0.2 * delivery_score + 
                      0.25 * quality_score + 
                      0.25 * reputation_score)
        
        return total_score
    
    def _calculate_reward(self, won: bool, price: float, action: BiddingAction, 
                         competitor_bids: List[Dict]) -> float:
        """
        Calculate reward for the agent's action
        """
        if not won:
            return -0.1  # Small penalty for losing
        
        # Base reward for winning
        reward = 1.0
        
        # Profit margin reward
        budget_max = self.current_project.get('budget_max', 100000)
        profit_margin = (price - budget_max * 0.6) / (budget_max * 0.6)  # Assume 60% cost
        reward += profit_margin * 0.5  # Bonus for higher profit margin
        
        # Efficiency bonus (winning with competitive price)
        if competitor_bids:
            min_competitor_price = min([bid['price'] for bid in competitor_bids])
            if price <= min_competitor_price * 1.1:  # Within 10% of lowest competitor
                reward += 0.3
        
        # Quality emphasis bonus
        reward += action.quality_emphasis * 0.1
        
        # Delivery time bonus (faster delivery)
        delivery_bonus = max(0, (60 - action.delivery_time) / 60 * 0.2)
        reward += delivery_bonus
        
        return reward

class DQNAgent:
    """
    Deep Q-Network agent for bidding optimization
    Implements the DQN approach described in the paper
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95  # discount factor
        self.update_target_freq = 100
        self.training_step = 0
        
        # Build neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the neural network for Q-value estimation"""
        model = Sequential([
            Dense(128, activation='relu', input_dim=self.state_size),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Target Q values
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.update_target_freq == 0:
            self.update_target_network()
    
    def save(self, filepath: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.q_network.save(filepath)
        
        # Save agent parameters
        params = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'training_step': self.training_step
        }
        
        params_path = filepath.replace('.h5', '_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
        
        logger.info(f"DQN agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        self.q_network = tf.keras.models.load_model(filepath)
        self.target_network = self._build_model()
        self.update_target_network()
        
        # Load parameters
        params_path = filepath.replace('.h5', '_params.pkl')
        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
            
            self.epsilon = params.get('epsilon', self.epsilon)
            self.training_step = params.get('training_step', 0)
        
        logger.info(f"DQN agent loaded from {filepath}")

class MultiAgentBiddingEnvironment:
    """
    Multi-agent environment for studying competitive dynamics
    Implements game-theoretic learning from the paper
    """
    
    def __init__(self, num_agents: int, historical_data: List[Dict]):
        self.num_agents = num_agents
        self.historical_data = historical_data
        self.agents = []
        self.current_project = None
        self.auction_history = []
        
    def add_agent(self, agent: DQNAgent, agent_id: str):
        """Add an agent to the environment"""
        self.agents.append({'agent': agent, 'id': agent_id, 'wins': 0, 'total_profit': 0})
    
    def run_auction(self) -> Dict:
        """Run a single auction with all agents"""
        # Sample project
        self.current_project = random.choice(self.historical_data)
        
        # Get bids from all agents
        bids = []
        state = self._get_state()
        
        for i, agent_info in enumerate(self.agents):
            agent = agent_info['agent']
            
            # Get action from agent
            action_index = agent.act(state)
            action = self._index_to_action(action_index)
            
            # Calculate bid details
            base_price = self.current_project.get('budget_max', 100000) * 0.7
            bid_price = base_price * action.price_multiplier
            
            bid = {
                'agent_id': agent_info['id'],
                'agent_index': i,
                'price': bid_price,
                'delivery_time': action.delivery_time,
                'quality_emphasis': action.quality_emphasis,
                'action_index': action_index
            }
            bids.append(bid)
        
        # Determine winner
        winner_info = self._determine_winner(bids)
        
        # Calculate rewards and update agents
        rewards = self._calculate_rewards(bids, winner_info)
        
        # Store experiences and train agents
        next_state = self._get_state()  # In this case, same as current state
        
        for i, (agent_info, reward) in enumerate(zip(self.agents, rewards)):
            agent = agent_info['agent']
            action_index = bids[i]['action_index']
            
            # Store experience
            agent.remember(state, action_index, reward, next_state, True)
            
            # Train agent
            agent.replay()
            
            # Update statistics
            if i == winner_info['winner_index']:
                agent_info['wins'] += 1
                agent_info['total_profit'] += winner_info['profit']
        
        # Record auction results
        auction_result = {
            'project_id': self.current_project.get('id', 'unknown'),
            'winner_id': winner_info['winner_id'],
            'winning_price': winner_info['winning_price'],
            'num_bidders': len(bids),
            'bids': bids
        }
        
        self.auction_history.append(auction_result)
        
        return auction_result
    
    def _get_state(self) -> np.ndarray:
        """Get state representation for agents"""
        return np.array([
            self.current_project.get('complexity', 5) / 10.0,
            self.current_project.get('budget_max', 100000) / 1000000.0,
            len(self.agents) / 10.0,  # Competition level
            random.uniform(0, 1),  # Market volatility
            random.uniform(0, 1),  # Demand level
        ])
    
    def _index_to_action(self, action_index: int) -> BiddingAction:
        """Convert action index to BiddingAction"""
        # Define action space (discrete)
        price_multipliers = [0.7, 0.8, 0.9, 1.0, 1.1]
        delivery_times = [15, 30, 45, 60]
        quality_levels = [0.6, 0.8, 1.0]
        
        # Map index to action components
        num_price = len(price_multipliers)
        num_delivery = len(delivery_times)
        num_quality = len(quality_levels)
        
        price_idx = action_index % num_price
        delivery_idx = (action_index // num_price) % num_delivery
        quality_idx = (action_index // (num_price * num_delivery)) % num_quality
        
        return BiddingAction(
            price_multiplier=price_multipliers[price_idx],
            delivery_time=delivery_times[delivery_idx],
            quality_emphasis=quality_levels[quality_idx]
        )
    
    def _determine_winner(self, bids: List[Dict]) -> Dict:
        """Determine auction winner based on scoring"""
        best_score = -1
        winner_index = 0
        
        for i, bid in enumerate(bids):
            score = self._calculate_bid_score(bid)
            if score > best_score:
                best_score = score
                winner_index = i
        
        winning_bid = bids[winner_index]
        base_cost = self.current_project.get('budget_max', 100000) * 0.6  # Assume 60% cost
        profit = winning_bid['price'] - base_cost
        
        return {
            'winner_index': winner_index,
            'winner_id': winning_bid['agent_id'],
            'winning_price': winning_bid['price'],
            'profit': profit,
            'best_score': best_score
        }
    
    def _calculate_bid_score(self, bid: Dict) -> float:
        """Calculate bid score for winner determination"""
        budget_max = self.current_project.get('budget_max', 100000)
        
        # Price score
        price_ratio = bid['price'] / budget_max
        price_score = max(0, 1 - (price_ratio - 0.5) * 2) if price_ratio >= 0.5 else 0.3
        
        # Delivery score
        delivery_score = max(0, 1 - bid['delivery_time'] / 90)
        
        # Quality score
        quality_score = bid['quality_emphasis']
        
        # Combined score
        total_score = 0.4 * price_score + 0.3 * delivery_score + 0.3 * quality_score
        return total_score
    
    def _calculate_rewards(self, bids: List[Dict], winner_info: Dict) -> List[float]:
        """Calculate rewards for all agents"""
        rewards = []
        
        for i, bid in enumerate(bids):
            if i == winner_info['winner_index']:
                # Winner gets positive reward based on profit
                base_reward = 1.0
                profit_bonus = winner_info['profit'] / 100000.0  # Normalized profit bonus
                reward = base_reward + profit_bonus
            else:
                # Losers get small negative reward
                reward = -0.1
                
                # But get bonus for competitive bids
                price_competitiveness = 1 - (bid['price'] - winner_info['winning_price']) / winner_info['winning_price']
                if price_competitiveness > 0:
                    reward += price_competitiveness * 0.05
            
            rewards.append(reward)
        
        return rewards
    
    def get_agent_statistics(self) -> List[Dict]:
        """Get performance statistics for all agents"""
        stats = []
        total_auctions = len(self.auction_history)
        
        for agent_info in self.agents:
            win_rate = agent_info['wins'] / total_auctions if total_auctions > 0 else 0
            avg_profit = agent_info['total_profit'] / agent_info['wins'] if agent_info['wins'] > 0 else 0
            
            stats.append({
                'agent_id': agent_info['id'],
                'wins': agent_info['wins'],
                'win_rate': win_rate,
                'total_profit': agent_info['total_profit'],
                'average_profit': avg_profit
            })
        
        return stats

class ReinforcementLearningTrainer:
    """
    Main trainer for reinforcement learning agents
    """
    
    def __init__(self, historical_data: List[Dict]):
        self.historical_data = historical_data
        self.single_agent_env = BiddingEnvironment(historical_data)
        self.multi_agent_env = None
        
    def train_single_agent(self, episodes: int = 1000) -> DQNAgent:
        """Train a single DQN agent against simulated competition"""
        state_size = 13  # Size of state representation
        action_size = 60  # 5 * 4 * 3 = price * delivery * quality combinations
        
        agent = DQNAgent(state_size, action_size)
        
        scores = deque(maxlen=100)
        
        for episode in range(episodes):
            state = self.single_agent_env.reset()
            state_vector = self._state_to_vector(state)
            
            action_index = agent.act(state_vector)
            action = self._index_to_action(action_index)
            
            next_state, reward, done, info = self.single_agent_env.step(action)
            next_state_vector = self._state_to_vector(next_state)
            
            agent.remember(state_vector, action_index, reward, next_state_vector, done)
            
            if done:
                scores.append(reward)
                
                if episode % 100 == 0:
                    avg_score = np.mean(scores)
                    logger.info(f"Episode {episode}, Average Score: {avg_score:.2f}, "
                              f"Epsilon: {agent.epsilon:.2f}")
            
            # Train the agent
            if len(agent.memory) > 32:
                agent.replay(32)
        
        return agent
    
    def train_multi_agent(self, num_agents: int = 4, episodes: int = 1000) -> List[DQNAgent]:
        """Train multiple agents in competitive environment"""
        self.multi_agent_env = MultiAgentBiddingEnvironment(num_agents, self.historical_data)
        
        # Create and add agents
        agents = []
        state_size = 5  # Simplified state for multi-agent
        action_size = 60
        
        for i in range(num_agents):
            agent = DQNAgent(state_size, action_size)
            agent_id = f"agent_{i}"
            self.multi_agent_env.add_agent(agent, agent_id)
            agents.append(agent)
        
        # Training loop
        for episode in range(episodes):
            auction_result = self.multi_agent_env.run_auction()
            
            if episode % 100 == 0:
                stats = self.multi_agent_env.get_agent_statistics()
                logger.info(f"Episode {episode}:")
                for stat in stats:
                    logger.info(f"  {stat['agent_id']}: Win Rate = {stat['win_rate']:.3f}, "
                              f"Avg Profit = {stat['average_profit']:.0f}")
        
        return agents
    
    def _state_to_vector(self, state: BiddingState) -> np.ndarray:
        """Convert BiddingState to numpy vector"""
        return np.concatenate([
            state.project_features,
            state.competition_features,
            state.provider_features,
            state.market_conditions,
            [state.time_remaining],
            state.budget_constraints
        ])
    
    def _index_to_action(self, action_index: int) -> BiddingAction:
        """Convert action index to BiddingAction"""
        price_multipliers = [0.7, 0.8, 0.9, 1.0, 1.1]
        delivery_times = [15, 30, 45, 60]
        quality_levels = [0.6, 0.8, 1.0]
        
        num_price = len(price_multipliers)
        num_delivery = len(delivery_times)
        
        price_idx = action_index % num_price
        delivery_idx = (action_index // num_price) % num_delivery
        quality_idx = (action_index // (num_price * num_delivery)) % len(quality_levels)
        
        return BiddingAction(
            price_multiplier=price_multipliers[price_idx],
            delivery_time=delivery_times[delivery_idx],
            quality_emphasis=quality_levels[quality_idx]
        )

class AdaptiveBiddingStrategy:
    """
    Adaptive bidding strategy that combines supervised learning and RL
    Implements the hybrid approach from the paper
    """
    
    def __init__(self, supervised_model, rl_agent: DQNAgent):
        self.supervised_model = supervised_model
        self.rl_agent = rl_agent
        self.adaptation_weight = 0.5  # Balance between supervised and RL predictions
        
    def predict_optimal_bid(self, project_data: Dict, competition_data: List[Dict],
                          provider_data: Dict) -> Dict:
        """
        Generate optimal bid using hybrid approach
        """
        # Get supervised learning prediction
        supervised_pred = self._get_supervised_prediction(project_data, competition_data, provider_data)
        
        # Get RL agent recommendation
        rl_recommendation = self._get_rl_recommendation(project_data, competition_data)
        
        # Combine predictions
        optimal_bid = self._combine_predictions(supervised_pred, rl_recommendation)
        
        return optimal_bid
    
    def _get_supervised_prediction(self, project_data: Dict, competition_data: List[Dict],
                                 provider_data: Dict) -> Dict:
        """Get prediction from supervised model"""
        # This would use the feature engineering and supervised model
        # For now, return a placeholder
        return {
            'win_probability': 0.6,
            'suggested_price': project_data.get('budget_max', 100000) * 0.8,
            'confidence': 0.7
        }
    
    def _get_rl_recommendation(self, project_data: Dict, competition_data: List[Dict]) -> Dict:
        """Get recommendation from RL agent"""
        # Create state representation
        state = np.array([
            project_data.get('complexity', 5) / 10.0,
            project_data.get('budget_max', 100000) / 1000000.0,
            len(competition_data) / 10.0,
            random.uniform(0, 1),  # Market conditions
            random.uniform(0, 1)   # Demand level
        ])
        
        # Get action from RL agent
        action_index = self.rl_agent.act(state)
        action = self._index_to_action(action_index)
        
        base_price = project_data.get('budget_max', 100000) * 0.7
        
        return {
            'price_multiplier': action.price_multiplier,
            'suggested_price': base_price * action.price_multiplier,
            'delivery_time': action.delivery_time,
            'quality_emphasis': action.quality_emphasis
        }
    
    def _combine_predictions(self, supervised_pred: Dict, rl_recommendation: Dict) -> Dict:
        """Combine supervised and RL predictions"""
        # Weighted combination of price suggestions
        combined_price = (self.adaptation_weight * supervised_pred['suggested_price'] + 
                         (1 - self.adaptation_weight) * rl_recommendation['suggested_price'])
        
        return {
            'optimal_price': combined_price,
            'delivery_time': rl_recommendation['delivery_time'],
            'quality_emphasis': rl_recommendation['quality_emphasis'],
            'win_probability': supervised_pred['win_probability'],
            'confidence': supervised_pred['confidence'] * self.adaptation_weight + 
                         (1 - self.adaptation_weight) * 0.6,  # RL confidence estimate
            'strategy_weights': {
                'supervised': self.adaptation_weight,
                'reinforcement': 1 - self.adaptation_weight
            }
        }
    
    def _index_to_action(self, action_index: int) -> BiddingAction:
        """Convert action index to BiddingAction"""
        price_multipliers = [0.7, 0.8, 0.9, 1.0, 1.1]
        delivery_times = [15, 30, 45, 60]
        quality_levels = [0.6, 0.8, 1.0]
        
        num_price = len(price_multipliers)
        num_delivery = len(delivery_times)
        
        price_idx = action_index % num_price
        delivery_idx = (action_index // num_price) % num_delivery
        quality_idx = (action_index // (num_price * num_delivery)) % len(quality_levels)
        
        return BiddingAction(
            price_multiplier=price_multipliers[price_idx],
            delivery_time=delivery_times[delivery_idx],
            quality_emphasis=quality_levels[quality_idx]
        )
    
    def update_adaptation_weight(self, recent_performance: Dict):
        """
        Dynamically adjust the balance between supervised and RL components
        based on recent performance
        """
        if recent_performance.get('supervised_accuracy', 0) > recent_performance.get('rl_performance', 0):
            self.adaptation_weight = min(0.8, self.adaptation_weight + 0.1)
        else:
            self.adaptation_weight = max(0.2, self.adaptation_weight - 0.1)
        
        logger.info(f"Updated adaptation weight to {self.adaptation_weight:.2f}")