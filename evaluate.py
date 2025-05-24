import torch
import numpy as np
from environment import Connect4Environment
from agent import DQNAgent, random_agent_action

def play_against_human(model_path: str):
    """
    Interactive game where human plays against trained AI.
    
    Args:
        model_path: Path to trained model
    """
    env = Connect4Environment()
    agent = DQNAgent()
    
    # Load trained model
    agent.load(model_path)
    agent.epsilon = 0  # No exploration during play
    
    print("=== Connect 4 vs AI ===")
    print("You are Player 1 (X), AI is Player 2 (O)")
    print("Enter column number (0-6) to drop your piece")
    print("=" * 40)
    
    while True:
        env.reset()
        print(env.render())
        
        while not env.game_over:
            if env.current_player == 1:
                # Human player's turn
                valid_actions = env.get_valid_actions()
                print(f"Valid columns: {valid_actions}")
                
                try:
                    action = int(input("Your move (column 0-6): "))
                    if action not in valid_actions:
                        print("Invalid move! Try again.")
                        continue
                except ValueError:
                    print("Please enter a valid number!")
                    continue
                
                env.step(action)
                
            else:
                # AI's turn
                board_encoding = env.get_board_encoding()
                valid_actions = env.get_valid_actions()
                
                if len(valid_actions) == 0:
                    break
                
                action = agent.get_action(board_encoding, valid_actions, training=False)
                print(f"AI plays column {action}")
                env.step(action)
            
            print(env.render())
        
        # Game over
        if env.winner == 1:
            print("🎉 You win!")
        elif env.winner == 2:
            print("🤖 AI wins!")
        else:
            print("🤝 It's a tie!")
        
        play_again = input("Play again? (y/n): ").lower()
        if play_again != 'y':
            break
    
    print("Thanks for playing!")

def evaluate_vs_random(model_path: str, games: int = 1000) -> dict:
    """
    Evaluate trained model against random opponent.
    
    Args:
        model_path: Path to trained model
        games: Number of games to play
        
    Returns:
        Dictionary with evaluation results
    """
    env = Connect4Environment()
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0  # No exploration
    
    results = {"wins": 0, "losses": 0, "ties": 0}
    
    print(f"Evaluating against random opponent for {games} games...")
    
    for game in range(games):
        env.reset()
        agent_player = np.random.choice([1, 2])  # Random starting player
        
        while not env.game_over:
            if env.current_player == agent_player:
                # Agent's turn
                board_encoding = env.get_board_encoding()
                valid_actions = env.get_valid_actions()
                
                if len(valid_actions) == 0:
                    break
                
                action = agent.get_action(board_encoding, valid_actions, training=False)
                env.step(action)
            else:
                # Random opponent's turn
                valid_actions = env.get_valid_actions()
                if len(valid_actions) == 0:
                    break
                
                action = random_agent_action(valid_actions)
                env.step(action)
        
        # Record result
        if env.winner == agent_player:
            results["wins"] += 1
        elif env.winner == 0:
            results["ties"] += 1
        else:
            results["losses"] += 1
        
        if (game + 1) % 100 == 0:
            win_rate = (results["wins"] + 0.5 * results["ties"]) / (game + 1)
            print(f"Games {game + 1}/{games}: Win rate = {win_rate:.3f}")
    
    total_games = results["wins"] + results["losses"] + results["ties"]
    win_rate = (results["wins"] + 0.5 * results["ties"]) / total_games
    
    print("\n=== Evaluation Results ===")
    print(f"Total games: {total_games}")
    print(f"Wins: {results['wins']} ({results['wins']/total_games:.1%})")
    print(f"Losses: {results['losses']} ({results['losses']/total_games:.1%})")
    print(f"Ties: {results['ties']} ({results['ties']/total_games:.1%})")
    print(f"Win rate: {win_rate:.3f} ({win_rate:.1%})")
    
    return results

def evaluate_vs_self(model_path: str, games: int = 100) -> dict:
    """
    Evaluate model playing against itself.
    
    Args:
        model_path: Path to trained model
        games: Number of games to play
        
    Returns:
        Dictionary with evaluation results
    """
    env = Connect4Environment()
    agent1 = DQNAgent()
    agent2 = DQNAgent()
    
    agent1.load(model_path)
    agent2.load(model_path)
    
    agent1.epsilon = 0
    agent2.epsilon = 0
    
    results = {"agent1_wins": 0, "agent2_wins": 0, "ties": 0}
    
    print(f"Self-play evaluation for {games} games...")
    
    for game in range(games):
        env.reset()
        agents = [agent1, agent2]
        current_agent_idx = 0
        
        while not env.game_over:
            current_agent = agents[current_agent_idx]
            board_encoding = env.get_board_encoding()
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            action = current_agent.get_action(board_encoding, valid_actions, training=False)
            env.step(action)
            
            # Switch to other agent
            current_agent_idx = 1 - current_agent_idx
        
        # Record result
        if env.winner == 1:
            results["agent1_wins"] += 1
        elif env.winner == 2:
            results["agent2_wins"] += 1
        else:
            results["ties"] += 1
    
    total_games = sum(results.values())
    
    print("\n=== Self-Play Results ===")
    print(f"Total games: {total_games}")
    print(f"Agent 1 wins: {results['agent1_wins']} ({results['agent1_wins']/total_games:.1%})")
    print(f"Agent 2 wins: {results['agent2_wins']} ({results['agent2_wins']/total_games:.1%})")
    print(f"Ties: {results['ties']} ({results['ties']/total_games:.1%})")
    
    return results

def demo_game(model_path: str):
    """
    Show a demo game with AI vs random opponent.
    
    Args:
        model_path: Path to trained model
    """
    env = Connect4Environment()
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0
    
    print("=== Demo Game: AI vs Random ===")
    env.reset()
    
    # AI is player 1, random is player 2
    move_count = 0
    
    while not env.game_over:
        print(f"\nMove {move_count + 1}")
        print(env.render())
        
        if env.current_player == 1:
            # AI's turn
            board_encoding = env.get_board_encoding()
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            action = agent.get_action(board_encoding, valid_actions, training=False)
            print(f"AI (Player 1) plays column {action}")
            env.step(action)
        else:
            # Random opponent's turn
            valid_actions = env.get_valid_actions()
            if len(valid_actions) == 0:
                break
            
            action = random_agent_action(valid_actions)
            print(f"Random (Player 2) plays column {action}")
            env.step(action)
        
        move_count += 1
        input("Press Enter to continue...")
    
    print("\n=== Final Board ===")
    print(env.render())
    
    if env.winner == 1:
        print("🤖 AI wins!")
    elif env.winner == 2:
        print("🎲 Random wins!")
    else:
        print("🤝 It's a tie!")

def benchmark_models(model_paths: list, games_per_model: int = 500):
    """
    Compare multiple trained models.
    
    Args:
        model_paths: List of paths to trained models
        games_per_model: Number of games to evaluate each model
    """
    print("=== Model Benchmark ===")
    
    results = {}
    
    for i, model_path in enumerate(model_paths):
        print(f"\nEvaluating model {i+1}: {model_path}")
        
        try:
            result = evaluate_vs_random(model_path, games_per_model)
            win_rate = (result["wins"] + 0.5 * result["ties"]) / games_per_model
            results[model_path] = win_rate
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
            results[model_path] = 0.0
    
    print("\n=== Benchmark Results ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model_path, win_rate) in enumerate(sorted_results, 1):
        print(f"{rank}. {model_path}: {win_rate:.3f} ({win_rate:.1%})")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluate.py <model_path> [mode]")
        print("  Modes: play, eval, demo, self")
        print("  Default mode: play")
        sys.exit(1)
    
    model_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "play"
    
    if mode == "play":
        play_against_human(model_path)
    elif mode == "eval":
        evaluate_vs_random(model_path, 1000)
    elif mode == "demo":
        demo_game(model_path)
    elif mode == "self":
        evaluate_vs_self(model_path, 100)
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: play, eval, demo, self")
