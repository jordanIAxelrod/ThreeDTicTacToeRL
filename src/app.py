from flask import Flask, render_template, jsonify, request
import torch
from game import ThreeDTicTacToe
from DQNAgent import TicTacToeAgent
from time import sleep
app = Flask(__name__)

# Initialize the game and agent
game = None
agent = None

def initialize_game():
    global game, agent
    game = ThreeDTicTacToe()
    agent = TicTacToeAgent(-1, epsilon=0)  # Bot plays as -1
    try:
        agent.load_model_from_file("agent2_model.pth")  # Load the trained model
    except Exception as e:
        print(e)
        print("Warning: Could not load model file. Using untrained agent.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_game', methods=['POST'])
def start_game():
    initialize_game()
    return jsonify({
        'board': game.get_state().tolist(),
        'status': 'started'
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    print(data)
    x = data.get('x')
    y = data.get('y')
    
    # Player's move (player is 1)
    if not game.move(1, (x, y)):
        return jsonify({
            'error': 'Invalid move',
            'board': game.get_state().tolist()
        })

    status = check_game_status()
    if status:
        return jsonify({
            'board': game.get_state().tolist(),
            'status': status['result'],
            'winning_coordinates': status['winning_coordinates']
        })

    # Bot's move
    action = agent.select_action(game)
    bot_x, bot_y = divmod(action, 3)
    game.move(-1, (bot_x, bot_y))

    status = check_game_status()
    response = {
        'board': game.get_state().tolist(),
        'bot_move': {'x': bot_x, 'y': bot_y}
    }
    if status:
        response.update({
            'status': status['result'],
            'winning_coordinates': status['winning_coordinates']
        })
    return jsonify(response)

def check_game_status():
    winner = game.check_win()
    winning_coords = game.get_winning_coordinates()
    if winner == 1:
        return {'result': 'player_wins', 'winning_coordinates': winning_coords}
    elif winner == -1:
        return {'result': 'bot_wins', 'winning_coordinates': winning_coords}
    elif game.full_board():
        return {'result': 'draw', 'winning_coordinates': None}
    return None

if __name__ == '__main__':
    app.run(debug=True) 