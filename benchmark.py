import time
import torch
import random
from vector2048_env import Vector2048Env
from evaluate import mpc_policy
from dqn2048 import DQN
from ppo2048 import PPOAgent


def main():
    episodes = 5
    rollout_length = 100
    batch_size = 1024

    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== Policy Benchmarks (same params) ==========

    print("\n=== Random Policy Baseline ===")
    rand_env = Vector2048Env(batch_size, 4, device)
    start = time.time()
    for ep in range(episodes):
        state, done = rand_env.reset()
        for _ in range(rollout_length):
            acts = torch.randint(0, 4, (batch_size,), device=rand_env.device)
            state, _, done = rand_env.step(acts)
    elapsed = time.time() - start
    print(f"Random: episodes={episodes}, steps={rollout_length}, batch={batch_size}, time={elapsed:.4f}s")

    print("\n=== MPC Policy Benchmark ===")
    mpc_env = Vector2048Env(1, 4, device)  # MPC runs one environment at a time
    horizon = 3; samples = 512
    start = time.time()
    for ep in range(episodes):
        state, done = mpc_env.reset()
        for _ in range(rollout_length):
            action = mpc_policy(mpc_env, state, horizon, samples, mpc_env.device)
            action_tensor = torch.tensor([action], dtype=torch.int64, device=mpc_env.device)
            state, _, done = mpc_env.step(action_tensor)
    elapsed = time.time() - start
    print(f"MPC (H={horizon}, S={samples}): episodes={episodes}, steps={rollout_length}, batch=1, time={elapsed:.4f}s")

    print("\n=== DQN Policy Inference ===")
    dqn_env = Vector2048Env(batch_size, 4, device)
    dqn_net = DQN(4).to(dqn_env.device)
    dqn_net.load_state_dict(torch.load('dqn2048.pth', map_location=dqn_env.device))
    dqn_net.eval()
    start = time.time()
    for ep in range(episodes):
        state, done = dqn_env.reset()
        for _ in range(rollout_length):
            with torch.no_grad():
                q = dqn_net(state.to(dqn_env.device))
            acts = q.argmax(dim=1)
            state, _, done = dqn_env.step(acts)
    elapsed = time.time() - start
    print(f"DQN: episodes={episodes}, steps={rollout_length}, batch={batch_size}, time={elapsed:.4f}s")

    print("\n=== PPO Policy Inference ===")
    ppo_env = Vector2048Env(batch_size, 4, device)
    ppo_agent = PPOAgent(4, batch_size).to(ppo_env.device)
    ppo_agent.load_state_dict(torch.load('ppo2048.pth', map_location=ppo_env.device))
    ppo_agent.eval()
    start = time.time()
    for ep in range(episodes):
        state, done = ppo_env.reset()
        for _ in range(rollout_length):
            with torch.no_grad():
                logits, _ = ppo_agent(state.to(ppo_env.device))
            acts = logits.argmax(dim=1)
            state, _, done = ppo_env.step(acts)
    elapsed = time.time() - start
    print(f"PPO: episodes={episodes}, steps={rollout_length}, batch={batch_size}, time={elapsed:.4f}s")


if __name__ == '__main__':
    main() 