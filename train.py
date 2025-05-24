import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from environment import Connect4Environment
from agent import DQNAgent, random_agent_action

def train_agent(
    episodes: int = 10000,
    save_every: int = 1000,
    eval_every: int = 500,
    eval_episodes: int = 100,
    model_dir: str = "models",
    plot_dir: str = "plots"
):
    """
    Train the DQN agent to play Connect 4.
    
    Args:
        episodes: Number of training episodes
        save_every: Save model every N episodes
        eval_every: Evaluate model every N episodes
        eval_episodes: Number of episodes for evaluation
        model_dir: Directory to save models
        plot_dir: Directory to save plots
    """
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = Connect4Environment()
    agent = DQNAgent()
    
    # Training metrics
    episode_rewards = []
    win_rates = []
    losses = []
    evaluation_episodes = []
    
    print("Starting training...")
    print(f"Device: {agent.device}")
    print(f"Episodes: {episodes}")
    print(f"Epsilon start: {agent.epsilon}")
    print(f"Epsilon min: {agent.epsilon_min}")
    print("-" * 50)
    
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Randomly decide if agent plays first or second
        agent_player = np.random.choice([1, 2])
        
        while not env.game_over:
            if env.current_player == agent_player:
                # Agent's turn
                board_encoding = env.get_board_encoding()
                valid_actions = env.get_valid_actions()
                
                if len(valid_actions) == 0:
                    break
                
                action = agent.get_action(board_encoding, valid_actions, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Adjust reward based on agent's perspective
                if done:
                    if env.winner == agent_player:
                        reward = 100  # Agent wins
                    elif env.winner == 0:
                        reward = 50   # Tie
                    else:
                        reward = -100  # Agent loses
                else:
                    reward = 1  # Small positive reward for valid moves
                
                # Store experience
                next_board_encoding = env.get_board_encoding()
                agent.remember(board_encoding, action, reward, next_board_encoding, done)
                
                total_reward += reward
                steps += 1
                
            else:
                # Random opponent's turn
                valid_actions = env.get_valid_actions()
                if len(valid_actions) == 0:
                    break
                
                action = random_agent_action(valid_actions)
                env.step(action)
        
        episode_rewards.append(total_reward)
        
        # Train the agent
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        # Evaluation
        if (episode + 1) % eval_every == 0:
            win_rate = evaluate_agent(agent, eval_episodes)
            win_rates.append(win_rate)
            evaluation_episodes.append(episode + 1)
            
            print(f"\nEpisode {episode + 1}")
            print(f"Win rate: {win_rate:.2%}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
        
        # Save model
        if (episode + 1) % save_every == 0:
            model_path = os.path.join(model_dir, f"connect4_model_episode_{episode + 1}.pth")
            agent.save(model_path)
    
    # Save final model
    final_model_path = os.path.join(model_dir, "connect4_model_final.pth")
    agent.save(final_model_path)
    
    # Plot training progress
    plot_training_progress(episode_rewards, win_rates, evaluation_episodes, plot_dir)
    
    print("\nTraining completed!")
    print(f"Final model saved to: {final_model_path}")
    
    return agent

def evaluate_agent(agent: DQNAgent, episodes: int = 100) -> float:
    """
    Evaluate the agent's performance against random opponents.
    
    Args:
        agent: Trained DQN agent
        episodes: Number of evaluation episodes
        
    Returns:
        Win rate as a float between 0 and 1
    """
    env = Connect4Environment()
    wins = 0
    ties = 0
    
    for _ in range(episodes):
        state = env.reset()
        agent_player = np.random.choice([1, 2])
        
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
        
        if env.winner == agent_player:
            wins += 1
        elif env.winner == 0:
            ties += 1
    
    win_rate = (wins + 0.5 * ties) / episodes  # Count ties as half wins
    return win_rate

def plot_training_progress(episode_rewards, win_rates, evaluation_episodes, plot_dir):
    """Plot and save training progress charts."""
    
    # Plot 1: Episode rewards over time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue')
    # Moving average
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = [np.mean(episode_rewards[i:i+window]) for i in range(len(episode_rewards)-window+1)]
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, color='red', linewidth=2)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Win rates over time
    plt.subplot(2, 2, 2)
    plt.plot(evaluation_episodes, win_rates, marker='o', linewidth=2, markersize=4)
    plt.title('Win Rate vs Random Opponent')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Plot 3: Reward distribution
    plt.subplot(2, 2, 3)
    plt.hist(episode_rewards, bins=50, alpha=0.7, color='green')
    plt.title('Reward Distribution')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Learning progress
    plt.subplot(2, 2, 4)
    if len(episode_rewards) >= 1000:
        # Compare early vs late performance
        early_rewards = episode_rewards[:1000]
        late_rewards = episode_rewards[-1000:]
        
        plt.hist(early_rewards, bins=30, alpha=0.5, label='Early (first 1000)', color='red')
        plt.hist(late_rewards, bins=30, alpha=0.5, label='Late (last 1000)', color='blue')
        plt.title('Learning Progress Comparison')
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to file
    metrics = {
        'episode_rewards': episode_rewards,
        'win_rates': win_rates,
        'evaluation_episodes': evaluation_episodes
    }
    np.save(os.path.join(plot_dir, 'training_metrics.npy'), metrics)
    
    print(f"Training plots saved to {plot_dir}/training_progress.png")

def self_play_training(
    episodes: int = 5000,
    pretrained_model: str = None,
    save_every: int = 1000,
    model_dir: str = "models"
):
    """
    Train agent using self-play (agent vs agent).
    
    Args:
        episodes: Number of self-play episodes
        pretrained_model: Path to pretrained model to start from
        save_every: Save model every N episodes
        model_dir: Directory to save models
    """
    
    os.makedirs(model_dir, exist_ok=True)
    
    env = Connect4Environment()
    agent1 = DQNAgent(epsilon=0.3)  # Lower epsilon for self-play
    agent2 = DQNAgent(epsilon=0.3)
    
    # Load pretrained model if provided
    if pretrained_model and os.path.exists(pretrained_model):
        agent1.load(pretrained_model)
        agent2.load(pretrained_model)
        print(f"Loaded pretrained model: {pretrained_model}")
    
    print("Starting self-play training...")
    
    for episode in tqdm(range(episodes), desc="Self-play"):
        state = env.reset()
        
        # Agents take turns
        agents = [agent1, agent2]
        current_agent_idx = 0
        
        while not env.game_over:
            current_agent = agents[current_agent_idx]
            board_encoding = env.get_board_encoding()
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            action = current_agent.get_action(board_encoding, valid_actions, training=True)
            next_state, reward, done, info = env.step(action)
            
            if done:
                # Assign rewards based on game outcome
                if env.winner == current_agent_idx + 1:
                    reward = 100  # Current agent wins
                elif env.winner == 0:
                    reward = 50   # Tie
                else:
                    reward = -100  # Current agent loses
            else:
                reward = 1  # Small positive reward for valid moves
            
            # Store experience
            next_board_encoding = env.get_board_encoding()
            current_agent.remember(board_encoding, action, reward, next_board_encoding, done)
            
            # Train both agents
            if len(current_agent.memory) > current_agent.batch_size:
                current_agent.replay()
            
            # Switch to other agent
            current_agent_idx = 1 - current_agent_idx
        
        # Save models periodically
        if (episode + 1) % save_every == 0:
            agent1.save(os.path.join(model_dir, f"selfplay_agent1_episode_{episode + 1}.pth"))
            agent2.save(os.path.join(model_dir, f"selfplay_agent2_episode_{episode + 1}.pth"))
    
    # Save final models
    agent1.save(os.path.join(model_dir, "selfplay_agent1_final.pth"))
    agent2.save(os.path.join(model_dir, "selfplay_agent2_final.pth"))
    
    print("Self-play training completed!")

if __name__ == "__main__":
    # Train agent
    agent = train_agent(episodes=10000)
    
    # Optional: Continue with self-play training
    # self_play_training(episodes=5000, pretrained_model="models/connect4_model_final.pth")
