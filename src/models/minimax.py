from src.game import ThreeDTicTacToe
from typing import Dict, Tuple, List
import numpy as np
from src.utils import print_board
import json
import os

class MinimaxAgent:
    def __init__(self, player: int, max_depth: int = 9, cache_file: str = None):
        self.player = player
        self.max_depth = max_depth
        self.cache: Dict[str, Tuple[int, int]] = {}  # Cache for game states
        self.nodes_evaluated = 0
        self.cache_file = cache_file or f"minimax_cache_player_{player}.json"
        # Center positions are typically stronger
        self.move_priorities = {
            (1,1): 3,  # Center positions
            (0,1): 2, (1,0): 2, (1,2): 2, (2,1): 2,  # Edge centers
            (0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1   # Corners
        }
        # Try to load existing cache
        self.load_cache()

    def save_cache(self):
        """Save the cache to a file"""
        # Convert tuple values to lists for JSON serialization
        serializable_cache = {k: list(v) for k, v in self.cache.items()}
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache to {self.cache_file}: {e}")

    def load_cache(self):
        """Load the cache from a file if it exists"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    loaded_cache = json.load(f)
                    # Convert lists back to tuples
                    self.cache = {k: tuple(v) for k, v in loaded_cache.items()}
                print(f"Loaded {len(self.cache)} cached positions")
        except Exception as e:
            print(f"Warning: Could not load cache from {self.cache_file}: {e}")
            self.cache = {}

    def get_dynamic_depth(self, game: ThreeDTicTacToe) -> int:
        """Adjust search depth based on game progress"""
        empty_spaces = (game.board == 0).sum().item()
        total_spaces = 27  # 3x3x3
        progress = (total_spaces - empty_spaces) / total_spaces
        
        if progress < 0.2:  # Early game
            return min(7, self.max_depth)
        elif progress < 0.5:  # Mid game
            return min(8, self.max_depth)
        return self.max_depth  # Late game

    def get_move(self, game: ThreeDTicTacToe):
        self.nodes_evaluated = 0
        legal_moves = list(game.legal_moves())
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Sort moves by priority for better alpha-beta pruning
        legal_moves = self.order_moves(legal_moves)
        
        # Use dynamic depth based on game progress
        current_depth = self.get_dynamic_depth(game)
        
        for move in legal_moves:
            game_copy = game.clone()
            game_copy.move(self.player, move)
            
            # Skip symmetric positions we've already evaluated
            if self.is_symmetric_position_evaluated(game_copy):
                continue
                
            new_score = self.minimax(game_copy, -self.player, current_depth - 1, alpha, beta)
            print(move, new_score, game_copy.board)
            
            if new_score > best_score:
                best_score = new_score
                best_move = move
                alpha = max(alpha, best_score)
                
            
        print(f"Nodes evaluated: {self.nodes_evaluated}")
        # Save cache after making a move
        self.save_cache()
        return best_move

    def order_moves(self, moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Order moves by priority for better alpha-beta pruning"""
        def move_priority(move):
            x, y = move[0], move[1]
            return self.move_priorities.get((x,y), 0)
        
        return sorted(moves, key=move_priority, reverse=True)

    def get_canonical_board(self, board: np.ndarray) -> str:
        """Get canonical representation of board accounting for symmetries"""
        # For simplicity, just consider rotations around z-axis
        canonical = str(board.tobytes())
        rotated = np.rot90(board, k=1, axes=(0,1))
        canonical = min(canonical, str(rotated.tobytes()))
        rotated = np.rot90(board, k=2, axes=(0,1))
        canonical = min(canonical, str(rotated.tobytes()))
        rotated = np.rot90(board, k=3, axes=(0,1))
        canonical = min(canonical, str(rotated.tobytes()))
        return canonical

    def is_symmetric_position_evaluated(self, game: ThreeDTicTacToe) -> bool:
        """Check if we've already evaluated a symmetric position"""
        canonical = self.get_canonical_board(game.board.numpy())
        return canonical in self.cache

    def minimax(self, game: ThreeDTicTacToe, current_player: int, depth: int, alpha: float, beta: float) -> int:
        self.nodes_evaluated += 1
        # Check cache first
        game_state = str(game.board.numpy().tobytes())  # Convert tensor array to bytes for hashing
        if game_state in self.cache and self.cache[game_state][1] >= depth:
            return self.cache[game_state][0]
        
        winner = game.check_win()
        if winner != 0:
            print(depth, game.board)
            score = winner * self.player
            # Add depth bonus/penalty to incentivize faster wins/slower losses
            if score > 0:  # Winning
                score = score + (depth / self.max_depth)  # Bonus for winning quickly
            else:  # Losing
                score = score - (depth / self.max_depth)  # Less penalty for losing later
            self.cache[game_state] = (score, depth)
            return score 
            
        if depth == 0:
            # Evaluate current position
            score = self.evaluate_position(game)
            self.cache[game_state] = (score, depth)
            return score
            
        legal_moves = list(game.legal_moves())
        if not legal_moves:  # Draw
            self.cache[game_state] = (0, depth)
            return 0
            
        if current_player == self.player:
            # Maximizing player
            best_score = float('-inf')
            for move in legal_moves:
                game_copy = game.clone()
                game_copy.move(current_player, move)
                score = self.minimax(game_copy, -current_player, depth - 1, alpha, beta)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:  # Alpha-beta pruning
                    break
            self.cache[game_state] = (best_score, depth)
            return best_score
        else:
            # Minimizing player
            best_score = float('inf')
            for move in legal_moves:
                game_copy = game.clone()
                game_copy.move(current_player, move)
                score = self.minimax(game_copy, -current_player, depth - 1, alpha, beta)
                
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:  # Alpha-beta pruning
                    break
            self.cache[game_state] = (best_score, depth)
            return best_score

    def evaluate_position(self, game: ThreeDTicTacToe) -> int:
        """
        Evaluate the current position without going deeper into the tree.
        This is a simple heuristic that can be improved based on game-specific knowledge.
        """
        # Count potential winning lines for each player
        score = 0
        # Add game-specific evaluation logic here
        # For example, count number of pieces in a row, control of center, etc.
        return score
        
        
        
if __name__ == "__main__":
    game = ThreeDTicTacToe()
    agent1 = MinimaxAgent(1)
    game.move(-1, (0,0))
    game.move(-1, (0,0))
    game.move(1, (0,1))

    agent2 = MinimaxAgent(-1)
    while len(list(game.legal_moves())) > 0 and game.check_win() == 0:
        move = agent1.get_move(game)
        print(move)
        game.move(1, move)
        print_board(game.board)
        move = agent2.get_move(game)
        game.move(-1, move)
        print_board(game.board)
    

