"""
Train a lightweight DQN advisor for the Hybrid strategy.

Usage:
    python -m scripts.train_rl_hybrid --data data/processed/sample.parquet --episodes 10
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import torch

from rl.dqn import DQNAgent
from rl.trading_environment import TradingEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    df = df.sort_values('timestamp') if 'timestamp' in df.columns else df.sort_index()
    required = {'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")
    return df


def train(args):
    df = load_data(args.data)
    env = TradingEnvironment(df, config={'transaction_cost': args.commission, 'slippage': args.slippage})
    agent = DQNAgent(env.state_dim, env.action_dim, config={
        'learning_rate': args.lr,
        'buffer_capacity': args.buffer,
        'batch_size': args.batch,
    })

    epsilon = args.epsilon_start
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action = agent.policy_net.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train_step()
            step += 1
            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        logger.info(f"Episode {episode+1}/{args.episodes} reward={total_reward:.2f} epsilon={epsilon:.3f}")
        agent.update_target_network()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.policy_net.state_dict(), output_dir / args.model_name)
    logger.info(f"Saved RL advisor weights to {output_dir / args.model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL advisor for Hybrid strategy")
    parser.add_argument('--data', required=True, help='Path to OHLCV parquet/csv')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--commission', type=float, default=0.001)
    parser.add_argument('--slippage', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--buffer', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epsilon-start', dest='epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', dest='epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon-decay', dest='epsilon_decay', type=float, default=0.99)
    parser.add_argument('--output-dir', default='models', help='Directory to save model weights')
    parser.add_argument('--model-name', default='dqn_rl_advisor.pth')
    args = parser.parse_args()

    train(args)
