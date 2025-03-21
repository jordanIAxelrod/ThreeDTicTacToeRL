from game import ThreeDTicTacToe
from models.DQNAgent import TicTacToeAgent
from utils import print_board



def get_player_move():
    """Prompt the player for a valid move."""
    while True:
        try:
            x, y = map(int, input("Enter your move (x y): ").split())
            if x in range(3) and y in range(3):
                return x, y
            print("Invalid input. Coordinates must be between 0 and 2.")
        except ValueError:
            print("Invalid input. Please enter two numbers separated by a space.")


def play_game(game, player, agent=None):
    """Runs the game loop, allowing play against another player or an AI agent."""
    while True:
        print_board(game.get_state())

        if agent and player == -1:
            action = agent.select_action(game)
            x, y = divmod(action, 3)
            print(f"AI chooses: {x}, {y}")
        else:
            x, y = get_player_move()

        if not game.move(player, (x, y)):
            print("Invalid move. Try again.")
            continue

        winner = game.check_win()
        if winner:
            print_board(game.get_state())
            print(f"Player {winner} wins!")
            break

        player *= -1


def main():
    """Main function to initialize and run the game."""
    game = ThreeDTicTacToe()
    player = 1

    if input("Do you want to play against the trained AI? (yes/no): ").strip().lower() == "yes":
        agent = TicTacToeAgent(model_path="agent1_model.pth")
        print(agent.epsilon)
        play_first = input("Do you want to play first? (yes/no): ").strip().lower() == "yes"
        player = -1 if not play_first else 1
    else:
        agent = None

    play_game(game, player, agent)


if __name__ == "__main__":
    main()
