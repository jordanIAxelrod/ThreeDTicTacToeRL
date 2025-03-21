from flask import Flask, render_template, jsonify, request
import torch
from game import ThreeDTicTacToe
from models.DQNAgent import TicTacToeAgent
from models.minimax import MinimaxAgent
from time import sleep
app = Flask(__name__)

# Initialize the game and agents
game = None
agent1 = None  # Player 1 AI
agent2 = None  # Player 2 AI

def initialize_game(ai_vs_ai=False, use_minimax=False, player_first=True):
    global game, agent1, agent2
    game = ThreeDTicTacToe()
    if ai_vs_ai:
        if use_minimax:
            agent1 = MinimaxAgent(1, cache_file="minimax_cache_player1_ai_vs_ai.json")  # First AI plays as 1
            agent2 = MinimaxAgent(-1, cache_file="minimax_cache_player2_ai_vs_ai.json")  # Second AI plays as -1
        else:
            agent1 = TicTacToeAgent(1, epsilon=0)  # First AI plays as 1
            agent2 = TicTacToeAgent(-1, epsilon=0)  # Second AI plays as -1
            try:
                agent1.load_model_from_file("agent1_model.pth")
                agent2.load_model_from_file("agent1_model.pth")  # Using same model for both AIs
            except Exception as e:
                print(e)
                print("Warning: Could not load model file. Using untrained agents.")
    else:
        agent1 = None
        if use_minimax:
            # Use different cache files for when AI plays first vs second
            cache_file = "minimax_cache_ai_first.json" if not player_first else "minimax_cache_ai_second.json"
            agent2 = MinimaxAgent(-1 if player_first else 1, cache_file=cache_file)
        else:
            agent2 = TicTacToeAgent(-1 if player_first else 1, epsilon=0)
            try:
                agent2.load_model_from_file("agent1_model.pth")
            except Exception as e:
                print(e)
                print("Warning: Could not load model file. Using untrained agent.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_game', methods=['POST'])
def start_game():
    data = request.json
    ai_vs_ai = data.get('ai_vs_ai', False)
    use_minimax = data.get('use_minimax', False)
    player_first = data.get('player_first', True)
    
    initialize_game(ai_vs_ai, use_minimax, player_first)
    
    # If AI goes first, make its move
    if not player_first and not ai_vs_ai:
        if use_minimax:
            move = agent2.get_move(game)
            bot_x, bot_y = move
        else:
            action = agent2.select_action(game)
            bot_x, bot_y = divmod(action, 3)
            
        # AI plays as 1 (X) when going first
        game.move(1, (bot_x, bot_y))
        
        status = check_game_status()
        response = {
            'board': game.get_state().tolist(),
            'status': 'started' if not status else status['result'],
            'ai_vs_ai': ai_vs_ai,
            'player_first': player_first,
            'bot_move': {'x': bot_x, 'y': bot_y}
        }
        if status:
            response['winning_coordinates'] = status['winning_coordinates']
        return jsonify(response)
    
    return jsonify({
        'board': game.get_state().tolist(),
        'status': 'started',
        'ai_vs_ai': ai_vs_ai,
        'player_first': player_first
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    x = data.get('x')
    y = data.get('y')
    ai_vs_ai = data.get('ai_vs_ai', False)
    use_minimax = data.get('use_minimax', False)
    player_first = data.get('player_first', True)
    
    if ai_vs_ai:
        # If player's move is provided in AI vs AI mode
        if x is not None and y is not None:
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
            
            # Second AI's move
            if use_minimax:
                move = agent2.get_move(game)
                ai2_x, ai2_y = move
            else:
                action = agent2.select_action(game)
                ai2_x, ai2_y = divmod(action, 3)
            game.move(-1, (ai2_x, ai2_y))
            
            status = check_game_status()
            response = {
                'board': game.get_state().tolist(),
                'ai2_move': {'x': ai2_x, 'y': ai2_y}
            }
            if status:
                response.update({
                    'status': status['result'],
                    'winning_coordinates': status['winning_coordinates']
                })
            return jsonify(response)
        
        # AI vs AI mode - both AIs move
        # First AI's move
        if use_minimax:
            move = agent1.get_move(game)
            ai1_x, ai1_y = move
        else:
            action = agent1.select_action(game)
            ai1_x, ai1_y = divmod(action, 3)
        game.move(1, (ai1_x, ai1_y))
        
        status = check_game_status()
        if status:
            return jsonify({
                'board': game.get_state().tolist(),
                'status': status['result'],
                'winning_coordinates': status['winning_coordinates'],
                'ai1_move': {'x': ai1_x, 'y': ai1_y}
            })
        
        # Second AI's move
        if use_minimax:
            move = agent2.get_move(game)
            ai2_x, ai2_y = move
        else:
            action = agent2.select_action(game)
            ai2_x, ai2_y = divmod(action, 3)
        game.move(-1, (ai2_x, ai2_y))
        
        status = check_game_status()
        response = {
            'board': game.get_state().tolist(),
            'ai1_move': {'x': ai1_x, 'y': ai1_y},
            'ai2_move': {'x': ai2_x, 'y': ai2_y}
        }
        if status:
            response.update({
                'status': status['result'],
                'winning_coordinates': status['winning_coordinates']
            })
        return jsonify(response)
    else:
        # Player vs AI mode
        if x is not None and y is not None:
            # Player's move
            # When player goes first, they are X (1), when they go second, they are O (-1)
            player_value = 1 if player_first else -1
            if not game.move(player_value, (x, y)):
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
            
            # If player goes second, don't make an AI move here
            if not player_first:
                return jsonify({
                    'board': game.get_state().tolist()
                })

            # If player goes first, make AI move
            if use_minimax:
                move = agent2.get_move(game)
                bot_x, bot_y = move
            else:
                action = agent2.select_action(game)
                bot_x, bot_y = divmod(action, 3)

            # When player goes first, AI is O (-1), when AI goes first, it is X (1)
            bot_value = -1 if player_first else 1
            game.move(bot_value, (bot_x, bot_y))

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
        
        # This section should only run when player goes second and it's AI's turn
        if not player_first:
            # Bot's move
            if use_minimax:
                move = agent2.get_move(game)
                bot_x, bot_y = move
            else:
                action = agent2.select_action(game)
                bot_x, bot_y = divmod(action, 3)

            # When player goes first, AI is O (-1), when AI goes first, it is X (1)
            bot_value = -1 if player_first else 1
            game.move(bot_value, (bot_x, bot_y))

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
        
        return jsonify({
            'board': game.get_state().tolist()
        })

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