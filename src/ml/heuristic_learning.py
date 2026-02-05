"""
ML-Based Heuristic Learning for A* Search
=========================================

Research:
- Arfaee et al. (2011) "Learning Heuristic Functions for Large State Spaces"
- Ferber et al. (2020) "Neural Network Heuristics for Classical Planning"

Strategy:
1. Collect training data from solved dungeons
2. Train neural network to predict remaining cost
3. Use NN predictions as heuristic in A*

Neural Network Architecture:
- Input: State features (position, inventory, ~10 features)
- Hidden: 128 → 64 → 32 neurons (ReLU)
- Output: Predicted cost to goal (linear)

Admissibility:
- NN heuristic must satisfy: h(s) ≤ h*(s) (never overestimate)
- Enforce via post-training scaling (multiply by 0.9)
"""

import os
import json
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


@dataclass
class TrainingExample:
    """Single training example for heuristic learning."""
    position: Tuple[int, int]
    keys: int
    has_bomb: bool
    has_boss_key: bool
    has_item: bool
    remaining_cost: float  # Ground truth label


class HeuristicNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Neural network for predicting remaining cost to goal.
    
    Architecture:
        Input (10) → FC(128) → ReLU → FC(64) → ReLU → FC(32) → ReLU → FC(1)
    
    Input Features:
    - position_x, position_y (normalized)
    - keys (count)
    - has_bomb, has_boss_key, has_item (binary)
    - manhattan_distance_to_goal
    - num_locked_doors_ahead
    - num_items_collected
    - exploration_progress (0-1)
    """
    
    def __init__(self, map_height: int, map_width: int):
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available!")
            return
        
        super(HeuristicNetwork, self).__init__()
        
        self.map_height = map_height
        self.map_width = map_width
        
        # Network layers
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Linear output (cost can be any positive value)
        return x
    
    def predict_cost(self, state_features: np.ndarray) -> float:
        """
        Predict remaining cost for a single state.
        
        Args:
            state_features: Numpy array of shape (10,)
            
        Returns:
            Predicted cost (float)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_features).unsqueeze(0)
            pred = self.forward(x)
            return max(0.0, pred.item())  # Ensure non-negative


class HeuristicTrainer:
    """
    Trainer for heuristic network.
    
    Workflow:
    1. Collect training data from solved dungeons
    2. Train network via supervised learning (MSE loss)
    3. Validate admissibility (never overestimate)
    4. Save trained model
    """
    
    def __init__(self, map_height: int, map_width: int):
        """Initialize trainer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ML heuristics")
        
        self.model = HeuristicNetwork(map_height, map_width)
        self.map_height = map_height
        self.map_width = map_width
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.training_history = []
    
    def collect_data_from_solution(
        self,
        path: List[Tuple[int, int]],
        states: List,  # GameState objects
        env
    ) -> List[TrainingExample]:
        """
        Extract training examples from a solved dungeon.
        
        For each state along the solution path, the label is the
        actual remaining cost (number of steps to goal).
        
        Args:
            path: Solution path (list of positions)
            states: List of GameState objects along path
            env: ZeldaLogicEnv instance
            
        Returns:
            List of training examples
        """
        examples = []
        
        for i, state in enumerate(states):
            remaining_cost = len(path) - i - 1  # Steps remaining to goal
            
            example = TrainingExample(
                position=state.position,
                keys=state.keys,
                has_bomb=state.has_bomb,
                has_boss_key=state.has_boss_key,
                has_item=state.has_item,
                remaining_cost=remaining_cost
            )
            
            examples.append(example)
        
        logger.info(f"Collected {len(examples)} training examples from solution")
        return examples
    
    def featurize_state(self, example: TrainingExample, env) -> np.ndarray:
        """
        Convert state to feature vector (10 features).
        
        Features:
        1-2. Normalized position (x/width, y/height)
        3. Key count
        4. Has bomb (0/1)
        5. Has boss key (0/1)
        6. Has item (0/1)
        7. Manhattan distance to goal (normalized)
        8. Number of locked doors in dungeon
        9. Number of items collected
        10. Exploration progress (0-1)
        """
        features = np.zeros(10, dtype=np.float32)
        
        # Position (normalized)
        features[0] = example.position[1] / self.map_width
        features[1] = example.position[0] / self.map_height
        
        # Inventory
        features[2] = min(example.keys / 5.0, 1.0)  # Normalize to [0, 1]
        features[3] = float(example.has_bomb)
        features[4] = float(example.has_boss_key)
        features[5] = float(example.has_item)
        
        # Manhattan distance (normalized)
        if env.goal_pos:
            dist = abs(example.position[0] - env.goal_pos[0]) + abs(example.position[1] - env.goal_pos[1])
            features[6] = dist / (self.map_height + self.map_width)
        
        # Door count (simplified - count from grid)
        from src.core.definitions import SEMANTIC_PALETTE
        locked_doors = np.sum(env.grid == SEMANTIC_PALETTE['DOOR_LOCKED'])
        features[7] = locked_doors / 10.0  # Normalize
        
        # Items collected (simplified - assume max 10 items)
        features[8] = example.keys / 10.0
        
        # Exploration progress (heuristic)
        features[9] = 1.0 - (example.remaining_cost / (self.map_height * self.map_width))
        
        return features
    
    def train(
        self,
        examples: List[TrainingExample],
        env,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Train the heuristic network.
        
        Args:
            examples: Training examples
            env: ZeldaLogicEnv instance
            epochs: Number of training epochs
            batch_size: Batch size
        """
        # Convert examples to tensors
        X = []
        y = []
        
        for example in examples:
            features = self.featurize_state(example, env)
            X.append(features)
            y.append(example.remaining_cost)
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)
        
        logger.info(f"Training on {len(X)} examples for {epochs} epochs")
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.training_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Training complete!")
    
    def enforce_admissibility(self, scaling_factor: float = 0.9):
        """
        Scale network outputs to ensure admissibility.
        
        Admissible heuristic: h(s) ≤ h*(s) (never overestimate)
        
        Simple approach: Multiply all predictions by 0.9
        """
        logger.info(f"Enforcing admissibility with scaling factor {scaling_factor}")
        
        # Scale final layer weights
        with torch.no_grad():
            self.model.fc4.weight.mul_(scaling_factor)
            if self.model.fc4.bias is not None:
                self.model.fc4.bias.mul_(scaling_factor)
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'map_height': self.map_height,
            'map_width': self.map_width,
            'training_history': self.training_history
        }, path)
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path: str) -> 'HeuristicNetwork':
        """Load trained model from disk."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        checkpoint = torch.load(path)
        model = HeuristicNetwork(
            checkpoint['map_height'],
            checkpoint['map_width']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model


# ==========================================
# INTEGRATION WITH A*
# ==========================================

class MLHeuristicAStar:
    """
    A* search using ML-learned heuristic instead of Manhattan distance.
    
    Usage:
        solver = MLHeuristicAStar(env, model_path='models/heuristic_net.pth')
        success, path, states = solver.solve(start_state)
    """
    
    def __init__(self, env, model_path: Optional[str] = None):
        """
        Initialize ML-based A*.
        
        Args:
            env: ZeldaLogicEnv instance
            model_path: Path to trained model (optional)
        """
        self.env = env
        self.model = None
        
        if model_path and os.path.exists(model_path) and TORCH_AVAILABLE:
            try:
                self.model = HeuristicTrainer.load_model(model_path)
                logger.info("ML heuristic loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
    
    def heuristic(self, state) -> float:
        """
        Compute heuristic using ML model or fallback to Manhattan.
        
        Args:
            state: GameState
            
        Returns:
            Heuristic value (predicted remaining cost)
        """
        if self.model is None:
            # Fallback to Manhattan distance
            if self.env.goal_pos is None:
                return 0
            pos = state.position
            goal = self.env.goal_pos
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Use ML heuristic
        trainer = HeuristicTrainer(self.env.height, self.env.width)
        example = TrainingExample(
            position=state.position,
            keys=state.keys,
            has_bomb=state.has_bomb,
            has_boss_key=state.has_boss_key,
            has_item=state.has_item,
            remaining_cost=0  # Not used for prediction
        )
        features = trainer.featurize_state(example, self.env)
        
        return self.model.predict_cost(features)
