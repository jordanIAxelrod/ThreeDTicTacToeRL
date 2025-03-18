import unittest
import torch
from src.game import ThreeDTicTacToe

class TestThreeDTicTacToe(unittest.TestCase):
    def setUp(self):
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
        self.assertNotIn((0, 0), legal_moves)  # Already occupied

    def test_check_win_horizontal(self):
        """Test horizontal win detection."""
        self.game.move(1, (0, 0))
        self.game.move(1, (1, 0))
        self.game.move(1, (2, 0))
        self.assertEqual(self.game.check_win(), 1)

    def test_check_win_vertical(self):
        """Test vertical win detection."""
        self.game.move(1, (0, 0))
        self.game.move(1, (0, 1))
        self.game.move(1, (0, 2))
        self.assertEqual(self.game.check_win(), 1)

    def test_check_win_diagonal(self):
        """Test diagonal win detection."""
        self.game.move(1, (0, 0))
        self.game.move(1, (1, 1))
        self.game.move(1, (2, 2))
        self.assertEqual(self.game.check_win(), 1)

    def test_check_two_in_a_row(self):
        """Test two-in-a-row detection."""
        self.game.move(1, (0, 0))
        self.game.move(1, (1, 0))
        self.assertEqual(self.game.check_two_in_a_row(), 1)

    def test_full_board(self):
        """Test full board detection."""
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    self.game.board[x, y, z] = 1
        self.assertTrue(self.game.full_board())

if __name__ == "__main__":
    unittest.main()
