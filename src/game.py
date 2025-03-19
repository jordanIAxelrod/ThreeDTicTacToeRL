import torch


class ThreeDTicTacToe:
    def __init__(self):
        """Initialize a 3x3x3 tic-tac-toe board."""
        self.board = torch.zeros((3, 3, 3), dtype=torch.int8)
        self.winning_coordinates = None  # Store winning coordinates

    def move(self, player: int, position: tuple) -> bool:
        """
        Attempt to place a piece in the given (x, y) position on the top 3x3 layer.
        Pieces drop to the lowest available position in the column.

        :param player: 1 or -1 indicating the player's piece.
        :param position: (x, y) coordinates for the move.
        :return: True if the move was successful, False otherwise.
        """
        x, y = position
        column = self.board[x, y, :]

        # Check if the column is full (i.e., no zeros left)
        if (column == 0).sum() == 0:
            return False  # Column is full

        # Place the piece in the lowest available slot
        for i in range(3):
            if column[i] == 0:
                self.board[x, y, i] = player
                return True

        return False

    def legal_moves(self):
        for i in range(3):
            for j in range(3):
                if self.board[i, j, -1] == 0:
                    yield i, j

    def get_state(self) -> torch.Tensor:
        """Return the current state of the board."""
        return self.board.clone()

    def check_win(self) -> int:
        """
        Check if there is a winner.
        Returns 1 if player 1 wins, -1 if player -1 wins, and 0 otherwise.
        """
        for axis in range(3):
            winner = self._check_axis(axis)
            if winner:
                return winner

        return 0

    def get_winning_coordinates(self):
        """Return the coordinates of the winning line if there is one."""
        return self.winning_coordinates

    def _check_axis(self, axis: int) -> int:
        """
        Helper function to check for a win along a given axis.
        The function permutes the board so that the given axis is first,
        allowing for uniform row/column checks.
        """
        board = self.board.permute((axis,) + tuple(range(3))[:axis] + tuple(range(3))[axis + 1:])

        # Check rows and columns for a win
        for i in range(3):
            for j in range(3):
                if abs(board[i, j, :].sum()) == 3:  # Horizontal win in this plane
                    # Convert coordinates back to original orientation
                    if axis == 0:
                        self.winning_coordinates = [(i, j, k) for k in range(3)]
                    elif axis == 1:
                        self.winning_coordinates = [(j, i, k) for k in range(3)]
                    else:
                        self.winning_coordinates = [(j, k, i) for k in range(3)]
                    return int(board[i, j, 0])
                
                if abs(board[i, :, j].sum()) == 3:  # Vertical win in this plane
                    if axis == 0:
                        self.winning_coordinates = [(i, k, j) for k in range(3)]
                    elif axis == 1:
                        self.winning_coordinates = [(k, i, j) for k in range(3)]
                    else:
                        self.winning_coordinates = [(k, j, i) for k in range(3)]
                    return int(board[i, 0, j])
                
                if abs(board[:, i, j].sum()) == 3:  # Depth-wise win through layers
                    if axis == 0:
                        self.winning_coordinates = [(k, i, j) for k in range(3)]
                    elif axis == 1:
                        self.winning_coordinates = [(i, k, j) for k in range(3)]
                    else:
                        self.winning_coordinates = [(i, j, k) for k in range(3)]
                    return int(board[0, i, j])

        # Check diagonals within each plane
        for i in range(3):
            if abs(board[i].diagonal().sum()) == 3:  # Top-left to bottom-right diagonal
                if axis == 0:
                    self.winning_coordinates = [(i, k, k) for k in range(3)]
                elif axis == 1:
                    self.winning_coordinates = [(k, i, k) for k in range(3)]
                else:
                    self.winning_coordinates = [(k, k, i) for k in range(3)]
                return int(board[i, 0, 0])
            
            if abs(torch.flip(board[i], [1]).diagonal().sum()) == 3:  # Top-right to bottom-left diagonal
                if axis == 0:
                    self.winning_coordinates = [(i, k, 2-k) for k in range(3)]
                elif axis == 1:
                    self.winning_coordinates = [(k, i, 2-k) for k in range(3)]
                else:
                    self.winning_coordinates = [(k, 2-k, i) for k in range(3)]
                return int(board[i, 0, 2])

        # Check 3D diagonals across layers
        diag1 = board.diagonal(dim1=0, dim2=1).diagonal().sum()
        diag2 = board.diagonal(dim1=0, dim2=2).diagonal().sum()
        diag3 = board.diagonal(dim1=1, dim2=2).diagonal().sum()
        diag4 = board.flip(2).diagonal(dim1=0, dim2=1).diagonal().sum()

        # Check 3D diagonals
        if abs(diag1) == 3:
            self.winning_coordinates = [(i, i, i) for i in range(3)]
            return int(torch.sign(diag1))
        if abs(diag2) == 3:
            self.winning_coordinates = [(i, i, 2-i) for i in range(3)]
            return int(torch.sign(diag2))
        if abs(diag3) == 3:
            self.winning_coordinates = [(i, 2-i, i) for i in range(3)]
            return int(torch.sign(diag3))
        if abs(diag4) == 3:
            self.winning_coordinates = [(2-i, i, i) for i in range(3)]
            return int(torch.sign(diag4))

        return 0

    def check_two_in_a_row(self) -> list:
        """Check if a player has two out of three in a row without the other player's piece."""
        players_with_two_in_a_row = []

        for axis in range(3):
            board = self.board.permute((axis,) + tuple(range(3))[:axis] + tuple(range(3))[axis + 1:])

            for i in range(3):
                for j in range(3):
                    for line in [board[i, j, :], board[i, :, j], board[:, i, j]]:
                        if (line == -1).any() and (line == 1).any():
                            continue  # Skip if both players are in the row
                        if line.sum() == 2 and 1 not in players_with_two_in_a_row:
                            players_with_two_in_a_row.append(1)  # Player 1 has two in a row
                        if line.sum() == -2 and -1 not in players_with_two_in_a_row:
                            players_with_two_in_a_row.append(-1)  # Player -1 has two in a row

                # Check diagonals in the plane
                for diag in [board[i].diagonal(), torch.flip(board[i], [1]).diagonal()]:
                    if (diag == -1).any() and (diag == 1).any():
                        continue
                    if diag.sum() == 2 and 1 not in players_with_two_in_a_row:
                        players_with_two_in_a_row.append(1)
                    if diag.sum() == -2 and -1 not in players_with_two_in_a_row:
                        players_with_two_in_a_row.append(-1)

        # Check 3D diagonals
        diagonals = [
            self.board.diagonal(dim1=0, dim2=1).diagonal(),
            self.board.diagonal(dim1=0, dim2=2).diagonal(),
            self.board.diagonal(dim1=1, dim2=2).diagonal(),
            self.board.flip(2).diagonal(dim1=0, dim2=1).diagonal()
        ]

        for diag in diagonals:
            if (diag == -1).any() and (diag == 1).any():
                continue
            if diag.sum() == 2 and 1 not in players_with_two_in_a_row:
                players_with_two_in_a_row.append(1)
            if diag.sum() == -2 and -1 not in players_with_two_in_a_row:
                players_with_two_in_a_row.append(-1)

        return players_with_two_in_a_row  # Return list of players with two-in-a-row

    def full_board(self) -> bool:
        return not (self.board == 0).any()
