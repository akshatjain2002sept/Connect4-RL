from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import os
from environment import Connect4Environment
from agent import DQNAgent

app = Flask(__name__)
CORS(app)

# Global variables
env = None
agent = None
game_state = {
    "board": None,
    "current_player": 1,
    "game_over": False,
    "winner": None,
    "ai_player": 2
}

def load_model(model_path: str = "models/connect4_model_final.pth"):
    """Load the trained model."""
    global agent
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Using random agent.")
        agent = None
        return False
    
    try:
        agent = DQNAgent()
        agent.load(model_path)
        agent.epsilon = 0  # No exploration during play
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        agent = None
        return False

def reset_game():
    """Reset the game state."""
    global env, game_state
    
    env = Connect4Environment()
    game_state = {
        "board": env.board.tolist(),
        "current_player": env.current_player,
        "game_over": env.game_over,
        "winner": env.winner,
        "ai_player": 2,
        "valid_actions": env.get_valid_actions()
    }

@app.route('/')
def index():
    """Serve the game interface."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Connect 4 vs AI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }
        .game-container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .board {
            display: grid;
            grid-template-columns: repeat(7, 60px);
            grid-gap: 5px;
            justify-content: center;
            background-color: #0066cc;
            padding: 10px;
            border-radius: 10px;
            margin: 20px auto;
        }
        .cell {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: white;
            border: 2px solid #004499;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .cell:hover {
            background-color: #e0e0e0;
        }
        .cell.player1 {
            background-color: #ff4444;
        }
        .cell.player2 {
            background-color: #ffff44;
        }
        .controls {
            margin: 20px 0;
        }
        button {
            background-color: #0066cc;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
        }
        button:hover {
            background-color: #004499;
        }
        .status {
            font-size: 18px;
            font-weight: bold;
            margin: 20px 0;
            color: #333;
        }
        .winner {
            font-size: 24px;
            color: #ff4444;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="game-container">
        <h1>Connect 4 vs AI</h1>
        <div class="status" id="status">Your turn! Click a column to drop your piece.</div>
        <div class="board" id="board"></div>
        <div class="controls">
            <button onclick="newGame()">New Game</button>
            <button onclick="setAIFirst()">AI Goes First</button>
        </div>
    </div>

    <script>
        let gameState = null;
        let aiFirst = false;

        async function fetchGameState() {
            const response = await fetch('/api/state');
            gameState = await response.json();
            updateDisplay();
        }

        async function makeMove(col) {
            if (gameState.game_over || gameState.current_player !== 1) return;
            
            const response = await fetch('/api/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({column: col})
            });
            
            gameState = await response.json();
            updateDisplay();
            
            // If game continues and it's AI's turn, make AI move
            if (!gameState.game_over && gameState.current_player === gameState.ai_player) {
                setTimeout(makeAIMove, 500);
            }
        }

        async function makeAIMove() {
            const response = await fetch('/api/ai_move', {method: 'POST'});
            gameState = await response.json();
            updateDisplay();
        }

        async function newGame() {
            const response = await fetch('/api/new_game', {method: 'POST'});
            gameState = await response.json();
            updateDisplay();
            
            // If AI goes first, make AI move
            if (aiFirst && gameState.current_player === gameState.ai_player) {
                setTimeout(makeAIMove, 500);
            }
        }

        function setAIFirst() {
            aiFirst = !aiFirst;
            newGame();
        }

        function updateDisplay() {
            const board = document.getElementById('board');
            const status = document.getElementById('status');
            
            // Clear board
            board.innerHTML = '';
            
            // Create board cells
            for (let row = 0; row < 6; row++) {
                for (let col = 0; col < 7; col++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    cell.onclick = () => makeMove(col);
                    
                    const cellValue = gameState.board[row][col];
                    if (cellValue === 1) {
                        cell.classList.add('player1');
                    } else if (cellValue === 2) {
                        cell.classList.add('player2');
                    }
                    
                    board.appendChild(cell);
                }
            }
            
            // Update status
            if (gameState.game_over) {
                if (gameState.winner === 1) {
                    status.innerHTML = '<span class="winner">🎉 You Win!</span>';
                } else if (gameState.winner === 2) {
                    status.innerHTML = '<span class="winner">🤖 AI Wins!</span>';
                } else {
                    status.innerHTML = '<span class="winner">🤝 It\'s a Tie!</span>';
                }
            } else {
                if (gameState.current_player === 1) {
                    status.textContent = 'Your turn! Click a column to drop your piece.';
                } else {
                    status.textContent = 'AI is thinking...';
                }
            }
        }

        // Initialize game
        fetchGameState();
    </script>
</body>
</html>
    """
    return html

@app.route('/api/state')
def get_state():
    """Get current game state."""
    return jsonify(game_state)

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game."""
    reset_game()
    return jsonify(game_state)

@app.route('/api/move', methods=['POST'])
def make_move():
    """Make a human player move."""
    global env, game_state
    
    data = request.get_json()
    column = data.get('column')
    
    if env.game_over:
        return jsonify({"error": "Game is over"}), 400
    
    if not env.is_valid_action(column):
        return jsonify({"error": "Invalid move"}), 400
    
    # Make the move
    env.step(column)
    
    # Update game state
    game_state.update({
        "board": env.board.tolist(),
        "current_player": env.current_player,
        "game_over": env.game_over,
        "winner": env.winner,
        "valid_actions": env.get_valid_actions()
    })
    
    return jsonify(game_state)

@app.route('/api/ai_move', methods=['POST'])
def make_ai_move():
    """Make an AI move."""
    global env, game_state, agent
    
    if env.game_over:
        return jsonify({"error": "Game is over"}), 400
    
    valid_actions = env.get_valid_actions()
    
    if len(valid_actions) == 0:
        return jsonify({"error": "No valid moves"}), 400
    
    if agent is None:
        # Use random move if no trained model
        action = np.random.choice(valid_actions)
    else:
        # Use trained AI
        board_encoding = env.get_board_encoding()
        action = agent.get_action(board_encoding, valid_actions, training=False)
    
    # Make the move
    env.step(action)
    
    # Update game state
    game_state.update({
        "board": env.board.tolist(),
        "current_player": env.current_player,
        "game_over": env.game_over,
        "winner": env.winner,
        "valid_actions": env.get_valid_actions(),
        "ai_move": action
    })
    
    return jsonify(game_state)

@app.route('/api/model/load', methods=['POST'])
def load_model_endpoint():
    """Load a different model."""
    data = request.get_json()
    model_path = data.get('model_path', 'models/connect4_model_final.pth')
    
    success = load_model(model_path)
    
    return jsonify({
        "success": success,
        "message": f"Model {'loaded' if success else 'failed to load'} from {model_path}"
    })

if __name__ == '__main__':
    # Try to load default model
    load_model()
    
    # Initialize game
    reset_game()
    
    print("Starting Connect 4 API server...")
    print("Access the game at: http://localhost:5000")
    print("API endpoints:")
    print("  GET  /api/state - Get game state")
    print("  POST /api/new_game - Start new game")
    print("  POST /api/move - Make human move")
    print("  POST /api/ai_move - Make AI move")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
