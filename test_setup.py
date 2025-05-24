#!/usr/bin/env python3
"""
Simple test script to verify the Connect 4 RL setup is working correctly.
"""

from environment import Connect4Environment
from agent import DQNAgent
import numpy as np

def test_environment():
    """Test the Connect 4 environment."""
    print('🎮 Testing Connect 4 Environment...')
    env = Connect4Environment()
    print(f'✅ Empty board created: {env.board.shape}')
    print(env.render())
    
    # Test basic game mechanics
    state = env.reset()
    valid_actions = env.get_valid_actions()
    print(f'Valid actions: {valid_actions}')
    
    # Make a move
    action = valid_actions[0]
    state, reward, done, info = env.step(action)
    print(f'Made move in column {action}')
    print(f'Reward: {reward}, Done: {done}')
    print(env.render())
    
    return env

def test_agent(env):
    """Test the DQN agent."""
    print('🤖 Testing DQN Agent...')
    agent = DQNAgent()
    print(f'✅ Agent created with device: {agent.device}')
    
    # Test board encoding
    board_encoding = env.get_board_encoding()
    print(f'Board encoding shape: {board_encoding.shape}')
    
    # Test action selection
    valid_actions = env.get_valid_actions()
    action = agent.get_action(board_encoding, valid_actions, training=False)
    print(f'Agent chose action: {action}')
    
    # Test memory storage
    next_state, reward, done, info = env.step(action)
    next_board_encoding = env.get_board_encoding()
    agent.remember(board_encoding, action, reward, next_board_encoding, done)
    print(f'✅ Experience stored in memory: {len(agent.memory)} experiences')
    
    return agent

def test_full_game():
    """Test a complete game."""
    print('🎯 Testing full game simulation...')
    env = Connect4Environment()
    agent = DQNAgent()
    
    move_count = 0
    while not env.game_over and move_count < 42:  # Max possible moves
        valid_actions = env.get_valid_actions()
        if len(valid_actions) == 0:
            break
            
        # Alternate between agent and random moves
        if env.current_player == 1:
            board_encoding = env.get_board_encoding()
            action = agent.get_action(board_encoding, valid_actions, training=False)
            print(f'Agent (Player 1) plays column {action}')
        else:
            action = np.random.choice(valid_actions)
            print(f'Random (Player 2) plays column {action}')
        
        env.step(action)
        move_count += 1
        
        if move_count <= 5:  # Show first few moves
            print(env.render())
    
    print(f'🏁 Game finished after {move_count} moves')
    if env.winner == 1:
        print('🏆 Agent wins!')
    elif env.winner == 2:
        print('🎲 Random wins!')
    else:
        print('🤝 Tie game!')
    
    print(env.render())

def main():
    """Run all tests."""
    print('🚀 Connect 4 RL Setup Test')
    print('=' * 40)
    
    try:
        # Test environment
        env = test_environment()
        print('✅ Environment test passed!\n')
        
        # Test agent
        agent = test_agent(env)
        print('✅ Agent test passed!\n')
        
        # Test full game
        test_full_game()
        print('✅ Full game test passed!\n')
        
        print('🎉 All tests passed! Your setup is ready for training!')
        print('\nNext steps:')
        print('1. Run: python train.py  (to train the model)')
        print('2. Run: python api.py    (to play via web interface)')
        print('3. Run: python evaluate.py models/connect4_model_final.pth play (to play via CLI)')
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 