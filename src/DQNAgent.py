import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from src.game import ThreeDTicTacToe
import torch.nn.functional as F
from utils import print_board, legal_moves


# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(27, 64)  # 3x3x3 board flattened
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)

        self.fc5 = nn.Linear(64, 9)  # 9 possible moves (x, y positions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))

        return self.fc5(x)


# Define the Reinforcement Learning Agent with Double DQN
class TicTacToeAgent:
    def __init__(self, number, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 model_path=None):
        self.number = number
        self.online_network = DQN()
        self.target_network = DQN()
        self.update_target_network()
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)
        self.temperature = 3.0

        if model_path:
            self.online_network.load_state_dict(torch.load(model_path))
            self.epsilon = 0  # Disable exploration when playing

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def get_q_values(self, state, legal_moves):
        """Get Q-values for legal moves."""

        # Always show the model that it is 1
        state_tensor = torch.tensor(state * self.number, dtype=torch.float32).flatten().unsqueeze(0)
        q_values = self.online_network(state_tensor).squeeze()
        return q_values[legal_moves]

    def select_action(self, game):

        legal_moves = [i * 3 + j for i, j in game.legal_moves()]
        if not legal_moves:
            raise ValueError("No legal moves available.")

        q_values = self.get_q_values(game.get_state(), legal_moves)

        if random.random() < self.epsilon:
            return self.sample_action(legal_moves, q_values)

        return legal_moves[torch.argmax(q_values).item()]

    def sample_action(self, legal_moves, q_values):
        """Sample an action using a softmax probability distribution with temperature scaling."""
        probabilities = F.softmax(q_values / self.temperature, dim=0)
        return random.choices(legal_moves, weights=probabilities.tolist(), k=1)[0]


    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=256):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).view(batch_size, -1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).view(batch_size, -1)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Get current Q values
        q_values = self.online_network(states).gather(1, actions)

        # --- Probabilistic State Transition ---
        with torch.no_grad():
            # Step 1: Opponent's Move Simulation
            flipped_next_states = -next_states  # View the next state as the opponent sees it
            opponent_best_actions = []
            for i, state in enumerate(flipped_next_states):
                lm = [i * 3 + j for i, j in legal_moves(state.numpy().reshape(3,3,3))]
                if not lm:  # Avoid empty moves (shouldn't happen in a valid game)
                    opponent_best_actions.append(0)  # Default to the first move if no legal move
                else:
                    opponent_q_values = self.get_q_values(state.numpy().reshape(3,3,3), lm)  # Get Q-values for legal moves
                    best_action = lm[opponent_q_values.argmax().item()]  # Pick the best move
                    opponent_best_actions.append(best_action)

            opponent_best_actions = torch.tensor(opponent_best_actions, dtype=torch.int64).unsqueeze(1)
            # Step 2: What Happens After the Opponent's Move?
            # This gives us the *actual* next state s' after both moves
            true_next_states = self._simulate_opponent_move(next_states, opponent_best_actions)

            # Step 3: Compute Target Q-values
            next_q_values = self.target_network(true_next_states).max(dim=1, keepdim=True)[0]  # Max Q-value for next state

            # Minimax Q-value update
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and optimize
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _simulate_opponent_move(self, next_states, opponent_best_actions):
        """ Simulates the opponent's best move to estimate the true next state s' """
        # This function should apply the opponent's action and return the resulting state
        # You will need access to the game environment for this part.

        simulated_states = next_states.clone()
        for i in range(len(next_states)):
            board = simulated_states[i].view(3, 3, 3)  # Convert to board shape
            x, y = divmod(opponent_best_actions[i].item(), 3)  # Get opponent's move position
            board[x, y] = -1  # Opponent makes a move
            simulated_states[i] = board.flatten()  # Convert back to tensor shape

        return simulated_states

    def save_model(self, filename):
        torch.save(self.online_network.state_dict(), filename)

    def load_model(self, other_agent):
        self.online_network.load_state_dict(other_agent.online_network.state_dict())
        self.target_network.load_state_dict(other_agent.target_network.state_dict())


# Training Loop - Self-Play
if __name__ == "__main__":
    agent1 = TicTacToeAgent(1, epsilon_min=.02)
    agent2 = TicTacToeAgent(-1, epsilon_min=.02)
    num_episodes = 10000
    num_epochs = 1
    target_update_frequency = 10
    total_reward1 = 0
    total_reward2 = 0
    for epoch in range(num_epochs):
        epoch_reward1 = 0
        epoch_reward2 = 0


        for episode in range(num_episodes):
            game = ThreeDTicTacToe()
            state = game.get_state().numpy()
            done = False
            player = random.choice([-1, 1])

            while not done:
                agent = agent1 if player == 1 else agent2
                action = agent.select_action(game)
                x, y = divmod(action, 3)

                reward = 0
                if not game.move(player, (x, y)):
                    reward = -1000
                    print("hello")
                    print(list(game.legal_moves()))
                    print(f"Invalid move: {x}, {y}")
                    print(game.get_state())
                    break

                if game.check_win() == player:
                    reward = 10
                    done = True
                    # print_board(game.get_state())
                elif game.check_two_in_a_row() == player * -1:
                    reward = -10
                elif game.full_board():
                    reward = 5
                    print("draw")
                    done = True

                next_state = game.get_state().numpy()
                agent.store_experience(state, action, reward, next_state, done)
                state = next_state
                if player == 1:
                    total_reward1 += reward
                    epoch_reward1 += reward
                else:
                    total_reward2 += reward
                    epoch_reward2 += reward
                player *= -1

            agent1.train()
            agent2.train()

            if episode % target_update_frequency == 0:
                agent1.update_target_network()
                agent2.update_target_network()

            if episode % 100 == 0:
                print(
                    f"Episode {episode}: Total Reward player 1 = {total_reward1}, {epoch_reward1}, Total Reward player 2 = {total_reward2}, {epoch_reward2}, Epsilon1 = {agent1.epsilon:.4f}, Epsilon2 = {agent2.epsilon:.4f}")

        if epoch_reward1 > epoch_reward2:
            agent2.load_model(agent1)
        else:
            agent1.load_model(agent2)

    agent1.save_model("agent1_model.pth")
    agent2.save_model("agent2_model.pth")