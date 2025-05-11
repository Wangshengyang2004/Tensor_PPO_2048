import time
import torch
from vector2048_env import Vector2048Env

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_envs = 4096         # batch size of parallel games (max GPU memory permitting)
    episodes = 50
    max_steps = 1000

    # Instantiate vectorized env
    env = Vector2048Env(num_envs, grid_size=4, device=device)

    returns = []
    start_time = time.time()
    for ep in range(1, episodes + 1):
        # reset batch of games
        state, done = env.reset()
        total_reward = torch.zeros(num_envs, device=device)
        for t in range(max_steps):
            # random actions for all envs in batch
            actions = torch.randint(0, 4, (num_envs,), dtype=torch.int64, device=device)
            # step once
            state, reward, done = env.step(actions)
            total_reward += reward
            # stop when all games done
            if done.all():
                break
        avg_ret = total_reward.mean().item()
        returns.append(avg_ret)
        print(f"Episode {ep}/{episodes}: avg_return={avg_ret:.2f}")

    elapsed = time.time() - start_time
    print("\nRandom baseline over batch of", num_envs)
    print(f"Avg return: {sum(returns)/len(returns):.2f}, Time: {elapsed:.2f}s, Steps/sec: {episodes * num_envs / elapsed:.1f}")

if __name__ == '__main__':
    main() 