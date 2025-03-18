def print_board(board):
    """Pretty-print the 3D tic-tac-toe board."""
    for z in range(3):
        print(f"Layer {z}:")
        for y in range(3):
            print(" ".join(str(int(board[x, y, z])) for x in range(3)))
        print()

def legal_moves(board):
    for i in range(3):
        for j in range(3):
            if board[i, j, -1] == 0:
                yield i, j