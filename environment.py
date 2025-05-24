import numpy as np
from typing import List, Tuple, Optional

class Connect4Environment:
    """
    Connect 4 game environment for reinforcement learning.
    """
    
    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.game_over = False
        self.winner = None
        
    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current board state."""
        return self.board.copy()
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid column indices where a piece can be dropped."""
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def is_valid_action(self, action: int) -> bool:
        """Check if action (column) is valid."""
        return 0 <= action < self.cols and self.board[0, action] == 0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action and return (state, reward, done, info).
        
        Args:
            action: Column index to drop piece
            
        Returns:
            new_state: Updated board state
            reward: Reward for the action
            done: Whether game is finished
            info: Additional info dictionary
        """
        if self.game_over:
            return self.get_state(), 0, True, {"error": "Game already over"}
        
        if not self.is_valid_action(action):
            # Invalid move penalty
            return self.get_state(), -10, True, {"error": "Invalid move"}
        
        # Drop piece in the column
        row = self._drop_piece(action, self.current_player)
        
        # Check for win
        if self._check_win(row, action, self.current_player):
            self.game_over = True
            self.winner = self.current_player
            reward = 100  # Win reward
        elif len(self.get_valid_actions()) == 0:
            # Board full - tie
            self.game_over = True
            self.winner = 0
            reward = 50  # Tie reward
        else:
            reward = 1  # Small positive reward for valid move
        
        # Switch players
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        
        info = {
            "winner": self.winner,
            "valid_actions": self.get_valid_actions()
        }
        
        return self.get_state(), reward, self.game_over, info
    
    def _drop_piece(self, col: int, player: int) -> int:
        """Drop a piece in the specified column and return the row it landed in."""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                return row
        return -1  # Column is full (shouldn't happen with valid actions)
    
    def _check_win(self, row: int, col: int, player: int) -> bool:
        """Check if the last move resulted in a win."""
        # Check all four directions: horizontal, vertical, diagonal
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal /
            (1, -1)   # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the piece just placed
            
            # Check positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.rows and 0 <= c < self.cols and 
                   self.board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.rows and 0 <= c < self.cols and 
                   self.board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 4:
                return True
        
        return False
    
    def render(self) -> str:
        """Render the board as a string."""
        result = "\n  " + " ".join([str(i) for i in range(self.cols)]) + "\n"
        for row in self.board:
            result += "| " + " ".join([str(cell) if cell != 0 else "." for cell in row]) + " |\n"
        result += "+" + "-" * (2 * self.cols + 1) + "+\n"
        return result
    
    def get_board_encoding(self) -> np.ndarray:
        """
        Get board encoding for neural network input.
        Returns 3D array: [player1_board, player2_board, valid_moves]
        """
        player1_board = (self.board == 1).astype(np.float32)
        player2_board = (self.board == 2).astype(np.float32)
        
        # Create valid moves layer
        valid_moves = np.zeros((self.rows, self.cols), dtype=np.float32)
        for col in self.get_valid_actions():
            for row in range(self.rows - 1, -1, -1):
                if self.board[row, col] == 0:
                    valid_moves[row, col] = 1.0
                    break
        
        return np.stack([player1_board, player2_board, valid_moves], axis=0)
