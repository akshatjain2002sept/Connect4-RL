# Connect4-RL

A reinforcement learning project that trains an AI agent to play Connect 4 using Deep Q-Learning (DQN). This is an excellent educational project for learning about machine learning, neural networks, and reinforcement learning concepts.

## 🎯 Project Overview

This project implements:
- **Connect 4 Game Environment**: Complete game logic with board state management
- **Deep Q-Network (DQN) Agent**: Neural network that learns to play Connect 4
- **Training Pipeline**: Comprehensive training system with progress tracking
- **Evaluation Tools**: Test the trained agent against various opponents
- **Web Interface**: Play against the AI through a beautiful web interface
- **Command Line Interface**: Interactive terminal-based gameplay

## 🚀 Quick Start

### 1. Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Setup (Optional)

```bash
python test_setup.py
```

This runs a comprehensive test to verify all components are working correctly.

### 4. Train the Model

```bash
python train.py
```

This will:
- Train the AI for 10,000 episodes against random opponents
- Save models every 1,000 episodes in the `models/` directory
- Generate training plots in the `plots/` directory
- Show progress updates every 500 episodes

### 5. Play Against the AI

**Web Interface (Recommended):**
```bash
python api.py
```
Then open http://localhost:5000 in your browser

**Command Line:**
```bash
python evaluate.py models/connect4_model_final.pth play
```

## 📁 Project Structure

```
connect4-rl/
├── environment.py    # Connect 4 game environment
├── agent.py         # DQN agent implementation
├── train.py         # Training pipeline
├── evaluate.py      # Evaluation and testing tools
├── api.py           # Web interface and API
├── requirements.txt # Python dependencies
├── models/          # Trained models (created during training)
├── plots/           # Training progress plots (created during training)
└── README.md        # This file
```

## 🧠 How It Works

### Deep Q-Learning (DQN)

The AI uses a deep neural network to learn the optimal strategy for Connect 4:

1. **State Representation**: The board is encoded as three 6x7 layers:
   - Player 1 pieces
   - Player 2 pieces  
   - Valid move positions

2. **Neural Network Architecture**:
   - 3 convolutional layers (32, 64, 128 filters)
   - 3 fully connected layers (512, 256, 7 neurons)
   - Output: Q-values for each of the 7 columns

3. **Training Process**:
   - **Experience Replay**: Stores game experiences in a buffer
   - **Target Network**: Stabilizes training with a separate target network
   - **Epsilon-Greedy**: Balances exploration vs exploitation
   - **Reward System**: +100 for wins, +50 for ties, -100 for losses, +1 for valid moves

### Key RL Concepts Demonstrated

- **Markov Decision Process**: Game states, actions, rewards, transitions
- **Q-Learning**: Learning action-value functions
- **Function Approximation**: Using neural networks for large state spaces
- **Experience Replay**: Breaking correlation in sequential data
- **Target Networks**: Stabilizing the learning process
- **Epsilon-Greedy Policy**: Exploration vs exploitation trade-off

## 🔧 Detailed Usage

### Training Options

**Basic Training:**
```bash
python train.py
```

**Custom Training (modify in train.py):**
```python
agent = train_agent(
    episodes=20000,        # Number of training episodes
    save_every=2000,       # Save model every N episodes
    eval_every=1000,       # Evaluate every N episodes
    eval_episodes=200      # Episodes per evaluation
)
```

**Self-Play Training:**
```python
# After initial training, improve with self-play
self_play_training(
    episodes=5000,
    pretrained_model="models/connect4_model_final.pth"
)
```

### Evaluation Commands

**Play Against Human:**
```bash
python evaluate.py models/connect4_model_final.pth play
```

**Evaluate Against Random Opponent:**
```bash
python evaluate.py models/connect4_model_final.pth eval
```

**Watch Demo Game:**
```bash
python evaluate.py models/connect4_model_final.pth demo
```

**Self-Play Evaluation:**
```bash
python evaluate.py models/connect4_model_final.pth self
```

### Web Interface Features

- **Interactive Board**: Click columns to make moves
- **Real-time Updates**: See moves as they happen
- **Game Statistics**: Track wins, losses, and ties
- **AI vs Human**: Play against the trained AI
- **Responsive Design**: Works on desktop and mobile

## 📊 Understanding the Results

### Training Metrics

The training process generates several important metrics:

1. **Win Rate**: Percentage of games won against random opponents
2. **Episode Rewards**: Total reward accumulated per episode
3. **Epsilon Decay**: Exploration rate over time
4. **Loss**: Neural network training loss

### Expected Performance

A well-trained model should achieve:
- **70-90% win rate** against random opponents
- **Stable performance** in self-play (around 50% win rate for each side)
- **Strategic play** (blocking opponent wins, creating winning opportunities)

### Training Progress Visualization

Training generates plots showing:
- Episode rewards over time
- Win rate progression
- Reward distribution
- Learning progress comparison

## 🎓 Learning Opportunities

This project demonstrates key ML/RL concepts:

### 1. **Environment Design**
- State representation
- Action spaces
- Reward engineering
- Game termination conditions

### 2. **Neural Network Architecture**
- Convolutional layers for spatial patterns
- Fully connected layers for decision making
- Dropout for regularization

### 3. **Reinforcement Learning**
- Q-learning algorithm
- Experience replay buffer
- Target network updates
- Epsilon-greedy exploration

### 4. **Training Strategies**
- Curriculum learning (random → self-play)
- Hyperparameter tuning
- Model checkpointing
- Performance evaluation

## 🔬 Experimentation Ideas

Try these modifications to learn more:

### 1. **Hyperparameter Tuning**
```python
agent = DQNAgent(
    lr=0.0005,           # Learning rate
    gamma=0.95,          # Discount factor
    epsilon_decay=0.999, # Exploration decay
    batch_size=64        # Training batch size
)
```

### 2. **Network Architecture**
- Add more convolutional layers
- Try different layer sizes
- Experiment with activation functions
- Add batch normalization

### 3. **Reward Engineering**
- Reward center column play
- Penalize random moves
- Reward defensive play
- Add positional bonuses

### 4. **Training Strategies**
- Train against different opponents
- Use curriculum learning
- Implement prioritized experience replay
- Try different exploration strategies

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# In agent.py, force CPU usage
device = torch.device("cpu")
```

**Poor Performance:**
- Train for more episodes
- Adjust learning rate
- Check reward function
- Verify game logic

**Training Too Slow:**
- Reduce network size
- Decrease batch size
- Use GPU if available
- Reduce evaluation frequency

### Performance Tips

1. **Use GPU**: Install CUDA-compatible PyTorch for faster training
2. **Parallel Training**: Train multiple agents simultaneously
3. **Checkpoint Frequently**: Save models regularly to prevent loss
4. **Monitor Progress**: Watch win rates and adjust hyperparameters

## 📚 Further Learning

### Reinforcement Learning Resources
- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [Deep RL Course by Hugging Face](https://huggingface.co/deep-rl-course/unit0/introduction)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

### Advanced Techniques to Explore
- **Double DQN**: Reduce overestimation bias
- **Dueling DQN**: Separate value and advantage estimation  
- **Rainbow DQN**: Combine multiple improvements
- **Policy Gradient Methods**: REINFORCE, A3C, PPO
- **Monte Carlo Tree Search**: AlphaZero-style approaches

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:
- Experiment with different architectures
- Try new training strategies
- Improve the web interface
- Add new evaluation metrics
- Share your results and learnings!

## 📄 License

This project is open source and available for educational use.

---

**Happy Learning! 🚀**

*This project demonstrates the power of reinforcement learning in game environments and provides hands-on experience with deep learning, neural networks, and AI training pipelines.*
