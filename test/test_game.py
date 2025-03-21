import unittest
import torch
from src.game import ThreeDTicTacToe
from src.utils import print_board

class TestThreeDTicTacToe(unittest.TestCase):
    def setUp(self):
        """Initialize a new game before each test."""
        self.game = ThreeDTicTacToe()

    def test_initial_board_empty(self):
        """Test that the board is initialized as empty (all zeros)."""
        self.assertTrue(torch.all(self.game.get_state() == 0))

    def test_valid_move(self):
        """Test that a valid move is placed correctly."""
        result = self.game.move(1, (0, 0))
        self.assertTrue(result)
        self.assertEqual(self.game.get_state()[0, 0, 0].item(), 1)

    def test_invalid_move(self):
        """Test that a move in a full column is rejected."""
        for _ in range(3):
            self.game.move(1, (0, 0))
        result = self.game.move(-1, (0, 0))
        self.assertFalse(result)

    def test_legal_moves(self):
        """Test that legal_moves() correctly identifies open spaces."""
        self.game.move(1, (0, 0))
        self.game.move(1, (0, 0))
        self.game.move(1, (0, 0))
        legal_moves = list(self.game.legal_moves())
        self.assertIn((0, 1), legal_moves)
        self.assertIn((1, 0), legal_moves)
        self.assertNotIn((0, 0), legal_moves)

    def test_horizontal_wins(self):
        """Test all possible horizontal line wins."""
        # Test each layer
        for z in range(3):
            # Test each row
            for y in range(3):
                self.game = ThreeDTicTacToe()  # Reset for each test
                self.game.board = torch.zeros((3, 3, 3))
                # Place three in a row
                for x in range(3):
                    self.game.board[x, y, z] = 1
                self.assertEqual(self.game.check_win(), 1)

    def test_vertical_wins(self):
        """Test all possible vertical line wins."""
        # Test each layer
        for z in range(3):
            # Test each column
            for x in range(3):
                self.game = ThreeDTicTacToe()
                self.game.board = torch.zeros((3, 3, 3))
                # Place three in a column
                for y in range(3):
                    self.game.board[x, y, z] = 1
                self.assertEqual(self.game.check_win(), 1)

    def test_depth_wins(self):
        """Test all possible depth line wins."""
        # Test each row and column position
        for x in range(3):
            for y in range(3):
                self.game = ThreeDTicTacToe()
                self.game.board = torch.zeros((3, 3, 3))
                # Place three in depth
                for z in range(3):
                    self.game.board[x, y, z] = 1
                self.assertEqual(self.game.check_win(), 1)

    def test_2d_diagonal_wins(self):
        """Test all possible 2D diagonal wins (in each plane)."""
        # Test each layer in the z-axis
        for z in range(3):
            # Test main diagonal
            self.game = ThreeDTicTacToe()
            self.game.board = torch.zeros((3, 3, 3))
            for i in range(3):
                self.game.board[i, i, z] = 1
            self.assertEqual(self.game.check_win(), 1)

            # Test anti-diagonal
            self.game = ThreeDTicTacToe()
            self.game.board = torch.zeros((3, 3, 3))
            for i in range(3):
                self.game.board[i, 2-i, z] = 1
            self.assertEqual(self.game.check_win(), 1)

        # Test each layer in the y-axis
        for y in range(3):
            # Test main diagonal
            self.game = ThreeDTicTacToe()
            self.game.board = torch.zeros((3, 3, 3))
            for i in range(3):
                self.game.board[i, y, i] = 1
            self.assertEqual(self.game.check_win(), 1)

            # Test anti-diagonal
            self.game = ThreeDTicTacToe()
            self.game.board = torch.zeros((3, 3, 3))
            for i in range(3):
                self.game.board[i, y, 2-i] = 1
            self.assertEqual(self.game.check_win(), 1)

        # Test each layer in the x-axis
        for x in range(3):
            # Test main diagonal
            self.game = ThreeDTicTacToe()
            self.game.board = torch.zeros((3, 3, 3))
            for i in range(3):
                self.game.board[x, i, i] = 1
            self.assertEqual(self.game.check_win(), 1)

            # Test anti-diagonal
            self.game = ThreeDTicTacToe()
            self.game.board = torch.zeros((3, 3, 3))
            for i in range(3):
                self.game.board[x, i, 2-i] = 1
            self.assertEqual(self.game.check_win(), 1)

    def test_3d_diagonal_wins(self):
        """Test all possible 3D diagonal wins (corner to corner)."""
        # Main diagonal (0,0,0) to (2,2,2)
        self.game = ThreeDTicTacToe()
        self.game.board = torch.zeros((3, 3, 3))
        for i in range(3):
            self.game.board[i, i, i] = 1
        self.assertEqual(self.game.check_win(), 1)
        self.assertEqual(set(self.game.get_winning_coordinates()), {(0,0,0), (1,1,1), (2,2,2)})

        # Diagonal (0,0,2) to (2,2,0)
        self.game = ThreeDTicTacToe()
        self.game.board = torch.zeros((3, 3, 3))
        for i in range(3):
            self.game.board[i, i, 2-i] = 1
        self.assertEqual(self.game.check_win(), 1)
        self.assertEqual(set(self.game.get_winning_coordinates()), {(0,0,2), (1,1,1), (2,2,0)})

        # Diagonal (0,2,0) to (2,0,2)
        self.game = ThreeDTicTacToe()
        self.game.board = torch.zeros((3, 3, 3))
        for i in range(3):
            self.game.board[i, 2-i, i] = 1
        self.assertEqual(self.game.check_win(), 1)
        self.assertEqual(set(self.game.get_winning_coordinates()), {(0,2,0), (1,1,1), (2,0,2)})

        # Diagonal (2,0,0) to (0,2,2)
        self.game = ThreeDTicTacToe()
        self.game.board = torch.zeros((3, 3, 3))
        for i in range(3):
            self.game.board[2-i, i, i] = 1
        self.assertEqual(self.game.check_win(), 1)
        self.assertEqual(set(self.game.get_winning_coordinates()), {(2,0,0), (1,1,1), (0,2,2)})

    def test_negative_player_wins(self):
        """Test that player -1 can also win."""
        self.game = ThreeDTicTacToe()
        self.game.board = torch.zeros((3, 3, 3))
        # Test a diagonal win for player -1
        for i in range(3):
            self.game.board[i, 2-i, i] = -1
        self.assertEqual(self.game.check_win(), -1)
        self.assertEqual(set(self.game.get_winning_coordinates()), {(0,2,0), (1,1,1), (2,0,2)})

    def test_no_win(self):
        """Test that no win is correctly identified."""
        # Test empty board
        self.assertEqual(self.game.check_win(), 0)

        # Test board with moves but no win
        self.game.board = torch.Tensor([
            [[-1, 0, 0],
             [0, -1, 0],
             [0, 0, 0]],
            [[0, 1, 0],
             [1, 1, 0],
             [0, 0, -1]],
            [[-1, -1, 0],
             [0, 1, 0],
             [-1, 0, 0]]
        ])
        print_board(self.game.board)
        self.assertEqual(self.game.check_win(), 0)

    def test_full_board(self):
        """Test full board detection."""
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    self.game.board[x, y, z] = 1
        self.assertTrue(self.game.full_board())

    
if __name__ == "__main__":
    unittest.main()
