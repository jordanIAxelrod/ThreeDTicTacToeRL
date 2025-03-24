"""Deep Q-Network (DQN) agent implementation for 3D Tic-Tac-Toe.

This module implements a DQN agent that learns to play 3D Tic-Tac-Toe using
reinforcement learning with experience replay and target networks.
"""

from typing import List, Tuple, Optional, Deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from src.game import ThreeDTicTacToe
from src.utils import legal_moves, print_board
import matplotlib.pyplot as plt

# Set device for PyTorch operations
device = "cpu"  #torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """Deep Q-Network architecture for 3D Tic-Tac-Toe.
    
    The network uses 3D convolutions to process the game state and outputs
    Q-values for each possible action.
    """
    
    def __init__(self) -> None:
        """Initialize the DQN architecture."""
        super(DQN, self).__init__()
        # Input shape: (batch_size, 1, 3, 3, 3)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=2, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=2, stride=1, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(64)
        
        # Calculate the size after convolutions
        self.fc1 = nn.Linear(64 * 7 ** 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)  # 9 possible moves (x, y positions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 3, 3, 3)
            
        Returns:
            Q-values for each possible action
        """
        x = x.view(-1, 1, 3, 3, 3).to(device)
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Fourth conv block
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class TicTacToeAgent:
    """Reinforcement Learning Agent for 3D Tic-Tac-Toe using Double DQN."""
    
    def __init__(
        self,
        number: int,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.02,
        model_path: Optional[str] = None
    ) -> None:
        """Initialize the agent.
        
        Args:
            number: Player number (1 or -1)
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum exploration rate
            model_path: Path to load a pre-trained model
        """
        self.number = number
        self.online_network = DQN().to(device)
        self.target_network = DQN().to(device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory: Deque[Tuple] = deque(maxlen=2000)
        self.temperature = 3.0
        self.illegal_moves_count = 0  # Track illegal moves per episode

        self.game_symmetry = [0, 0]  # the first digit is the rotation from 0 - 2, second is to flip the board 0 or 1.

        if model_path:
            self.online_network.load_state_dict(torch.load(model_path))
            self.epsilon = 0  # Disable exploration when playing

    def update_target_network(self) -> None:
        """Update the target network with the online network's weights."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def into_symmetric_space(self, state: torch.Tensor) -> torch.Tensor:
        """Apply symmetry to the state.
        
        Args:
            state: Current game state
            
        Returns:
            Symmetric game state
        """

        state = state.rot90(self.game_symmetry[0])
        if self.game_symmetry[1]:
            state = state.transpose(1, 0)
        return state

    def out_of_symmetric_space(self, state: torch.Tensor) -> torch.Tensor:
        """Convert a symmetric state back to the original space.
        
        Args:
            state: Symmetric game state
            
        """
        if self.game_symmetry[1]:
            state = state.transpose(1, 0)
        state = state.rot90(-self.game_symmetry[0])
        return state

    def get_q_values(self, state: torch.Tensor, legal_moves: List[int]) -> torch.Tensor:
        """Get Q-values for legal moves.
        
        Args:
            state: Current game state tensor
            legal_moves: List of legal move indices
            
        Returns:
            Q-values for legal moves
        """
        q_values = self.online_network((state * self.number).float()).squeeze()
        return q_values[legal_moves]

    def select_action(self, game: ThreeDTicTacToe, first_move: bool = False) -> int:
        """Select an action using epsilon-greedy policy.
        
        Args:
            game: Current game instance
            
        Returns:
            Selected action index
        """
       
        q_values = self.online_network((game.get_state() * self.number).float()).squeeze()
        if random.random() < self.epsilon or first_move:
            action = self.sample_action(list(range(9)), q_values)
        else:
            action = torch.argmax(q_values).item()

        return action

    def sample_action(self, legal_moves: List[int], q_values: torch.Tensor) -> int:
        """Sample an action using uniform distribution.
        
        Args:
            legal_moves: List of legal move indices
            q_values: Q-values for legal moves (not used in uniform sampling)
            
        Returns:
            Sampled action index
        """
        return random.choice(legal_moves)

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        """Store an experience tuple in the replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is finished
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size: int = 256) -> None:
        """Train the agent using experience replay.
        
        Args:
            batch_size: Number of experiences to sample for training
        """
    
        while len(self.memory) >= batch_size:
            # Take exactly batch_size experiences
            batch = []
            for _ in range(batch_size):
                batch.append(self.memory.popleft())
            
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.stack(states).float().to(device)
            next_states = torch.stack(next_states).float().to(device)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)    

            # Get current Q values
            q_values = self.online_network((states * self.number).float()).gather(1, actions)

            # Compute target Q values using Double DQN
            with torch.no_grad():
                # Simulate opponent's move
                flipped_next_states = -next_states  # 1 -1 1 -> -1 1 -1
                opponent_best_actions = self._get_opponent_actions(flipped_next_states)
                
                # Get true next states after opponent's move
                true_next_states = self._simulate_opponent_move(flipped_next_states, opponent_best_actions, dones)  # true next states from current agent perspective
                # Compute target Q-values
                next_q_values = self.target_network((true_next_states * self.number).float()).max(dim=1, keepdim=True)[0]
                target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

            # Compute loss and optimize
            loss = self.criterion(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def _get_opponent_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Get opponent's best actions for each state.
        
        Args:
            states: Batch of states from opponent's perspective
            
        Returns:
            Tensor of opponent's best actions
        """
        opponent_actions = []
        for state in states:
            lm = [i * 3 + j for i, j in legal_moves(state.cpu().numpy().reshape(3,3,3))]
            if not lm:
                opponent_actions.append(0)
            else:
                opponent_q_values = self.get_q_values(state, lm)
                best_action = lm[opponent_q_values.argmax().item()]
                opponent_actions.append(best_action)
        return torch.tensor(opponent_actions, dtype=torch.int64).unsqueeze(1)

    def _simulate_opponent_move(self, states: torch.Tensor, actions: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Simulate opponent's move to get true next states.
        
        Args:
            states: Current states from opponents view
            actions: Opponent's actions
            
        Returns:
            States after opponent's moves from current agent perspective
        """
        simulated_states = states.clone()
        for idx in range(len(states)):
            if dones[idx].item():  # Skip if game is done
                continue
                
            board = simulated_states[idx].view(3, 3, 3)
            x, y = divmod(actions[idx].item(), 3)
            
            # Find first empty z position
            for z in range(3):
                if board[x, y, z] == 0:
                    board[x, y, z] = self.number
                    break
                    
            simulated_states[idx] = -board.flatten()  # Reverse perspective back
            
        return simulated_states

    def save_model(self, filename: str) -> None:
        """Save the model's state dictionary.
        
        Args:
            filename: Path to save the model
        """
        torch.save(self.online_network.state_dict(), filename)

    def load_model(self, other_agent: 'TicTacToeAgent') -> None:
        """Load model weights from another agent.
        
        Args:
            other_agent: Agent to copy weights from
        """
        self.online_network.load_state_dict(other_agent.online_network.state_dict())
        self.target_network.load_state_dict(other_agent.target_network.state_dict())

    def load_model_from_file(self, filename: str) -> None:
        """Load model weights from a file.
        
        Args:
            filename: Path to load the model from
        """
        self.online_network.load_state_dict(torch.load(filename))

    def reset_illegal_moves_count(self):
        """Reset the illegal moves counter for a new episode."""
        self.illegal_moves_count = 0


def initialize_game() -> Tuple[ThreeDTicTacToe, torch.Tensor, bool, int]:
    """Initialize a new game.
    
    Returns:
        Tuple containing (game instance, initial state, done flag, starting player)
    """
    game = ThreeDTicTacToe()
    state = game.get_state()
    done = False
    player = random.choice([-1, 1])
    return game, state, done, player


def get_action(agent: TicTacToeAgent, game: ThreeDTicTacToe, player: int) -> Tuple[int, int, int]:
    """Get an action from the agent.
    
    Args:
        agent: The agent to get the action from
        game: Current game instance
        player: Current player number
        
    Returns:
        Tuple containing (action index, x coordinate, y coordinate)
    """
    action = agent.select_action(game)
    x, y = divmod(action, 3)
    return action, x, y

def get_reward(game: ThreeDTicTacToe, player: int, move: Tuple[int, int]) -> Tuple[float, bool]:
    """Calculate reward for a move.
    
    Args:
        game: Current game instance
        player: Current player number
        move: Move coordinates (x, y)
        
    Returns:
        Tuple containing (reward value, done flag, done_next_move flag)
    """
    reward = 0
    done = False
    done_next_move = False
    
    # Check opponent's winning potential before the move
    opponent_could_win = -player in game.check_two_in_a_row()
    
    # Make the move and get game state
    if not game.move(player, move):
        reward = -5.0  # Severe penalty for invalid moves
        return reward, done, done_next_move
        
    # Check if we won
    if game.check_win() == player:
        reward = 15.0 - game.board.abs().sum() / 27 * 9  # Higher reward for quick wins
        done = True
        return reward, done, done_next_move
        
    # Check if we successfully blocked opponent's winning move
    if opponent_could_win:
        # Check if opponent still has a winning move after our move
        if -player in game.check_two_in_a_row():
            reward = -12.0 + game.board.abs().sum() / 27 * 8  # Severe penalty for missing crucial blocks
            done_next_move = True
        else:
            reward = 8.0  # High reward for successful blocks
            
    # Check if we created a winning opportunity
    elif player in game.check_two_in_a_row():
        reward = 1.0  # small reward for creating winning opportunities dont want this to overpower blocking
        
    # Check if game ended in draw
    elif game.full_board():
        reward = 2.0  # Small positive reward for draws
        done = True
        
    # Evaluate position quality
    else:
        # Small positive reward for valid moves to encourage exploration
        reward += 0.1

    return reward, done, done_next_move


def train_agents(
    num_episodes: int = 10000,  # More episodes per epoch
    num_epochs: int = 40,
    target_update_frequency: int = 10,  # Less frequent target updates to allow learning to stabilize
    epsilon: float = 0.8,  # Higher initial exploration
    epsilon_min: float = 0.15,  # Slightly higher minimum exploration
    batch_size: int = 256,  # Larger batch size for more stable gradients
    learning_rate: float = 0.0003,  # Slightly lower learning rate
    from_pretrained: bool = False
) -> None:
    """Train two agents through self-play.
    
    Args:
        num_episodes: Number of episodes per epoch
        num_epochs: Number of training epochs
        target_update_frequency: How often to update target networks
        epsilon_min: Minimum exploration rate
        batch_size: Size of training batches
        learning_rate: Learning rate for the optimizer
    """
    agent1 = TicTacToeAgent(1, epsilon_min=epsilon_min, epsilon=epsilon, learning_rate=learning_rate)
    agent2 = TicTacToeAgent(-1, epsilon_min=epsilon_min, epsilon=epsilon, learning_rate=learning_rate)
    if from_pretrained:
        agent1.load_model_from_file("agent1_model.pth")
        agent2.load_model_from_file("agent2_model.pth")
    
    total_reward1 = 0
    total_reward2 = 0
    
    # Lists to store metrics
    agent1_illegal_moves = []
    agent2_illegal_moves = []
    agent1_wins = []
    agent2_wins = []
    
    for epoch in range(num_epochs):
        epoch_reward1 = 0
        epoch_reward2 = 0
        epoch_wins1 = 0
        epoch_wins2 = 0

        for episode in range(num_episodes):
            game, state, done, player = initialize_game()
            
            # Reset illegal moves counters at the start of each episode
            agent1.reset_illegal_moves_count()
            agent2.reset_illegal_moves_count()
            first_move = True

            while not done:
                agent = agent1 if player == 1 else agent2
                legal_moves_list = [i * 3 + j for i, j in game.legal_moves()]
                is_illegal = True
                
                # Inner loop to get a legal move
                while is_illegal:
                    action = agent.select_action(game, first_move)
                    first_move = False

                    is_illegal = False
                    if action not in legal_moves_list:
                        is_illegal = True    
                        agent.illegal_moves_count += 1
                    
                    # Get coordinates for the legal move
                    x, y = divmod(action, 3)
                    
                    # Get reward and store experience for legal move
                    reward, done, done_next_move = get_reward(game, player, (x, y))
                    next_state = game.get_state()
                    agent.store_experience(state, action, reward, next_state, done or done_next_move)
                    state = next_state
                    
                if player == 1:
                    total_reward1 += reward
                    epoch_reward1 += reward
                else:
                    total_reward2 += reward
                    epoch_reward2 += reward
                    
                player *= -1

            # Record metrics for this episode
            agent1_illegal_moves.append(agent1.illegal_moves_count)
            agent2_illegal_moves.append(agent2.illegal_moves_count)
            agent1_wins.append(epoch_wins1)
            agent2_wins.append(epoch_wins2)
            
            # Record wins
            winner = game.check_win()
            if winner == 1:
                epoch_wins1 += 1
            elif winner == -1:
                epoch_wins2 += 1

            # Train both agents more frequently
            agent1.train(batch_size=batch_size)
            agent2.train(batch_size=batch_size)

            if episode % target_update_frequency == 0:
                agent1.update_target_network()
                agent2.update_target_network()

            if episode % 100 == 0:
                print(f"Epoch {epoch}:")
                print(f"  Episode {episode}:")
                print(f"    Player 1 - Total Reward: {total_reward1}, Epoch Reward: {epoch_reward1}, Epsilon: {agent1.epsilon:.4f}, Illegal Moves: {agent1.illegal_moves_count}, Wins: {epoch_wins1}")
                print(f"    Player 2 - Total Reward: {total_reward2}, Epoch Reward: {epoch_reward2}, Epsilon: {agent2.epsilon:.4f}, Illegal Moves: {agent2.illegal_moves_count}, Wins: {epoch_wins2}")

        # Update weaker agent with stronger agent's weights
        if epoch_reward1 > epoch_reward2:
            agent2.load_model(agent1)
        else:
            agent1.load_model(agent2)

        # Save models after each epoch
        agent1.save_model("agent1_model.pth")
        agent2.save_model("agent2_model.pth")

        # Plot metrics at the end of each epoch
        plt.figure(figsize=(15, 10))
        
        # Plot illegal moves
        plt.subplot(2, 2, 1)
        plt.plot(agent1_illegal_moves, label='Agent 1')
        plt.plot(agent2_illegal_moves, label='Agent 2')
        plt.title('Illegal Moves per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Illegal Moves')
        plt.legend()
        
        # Plot win rates
        plt.subplot(2, 2, 2)
        plt.plot(agent1_wins, label='Agent 1')
        plt.plot(agent2_wins, label='Agent 2')
        plt.title('Wins per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Wins')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_metrics_epoch_{epoch}.png')
        plt.close()


if __name__ == "__main__":
    train_agents()
