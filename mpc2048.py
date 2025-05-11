import time
import torch
import random
import matplotlib.pyplot as plt
from collections import Counter
from vector2048_env import Vector2048Env
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import platform

# Default configuration
DEVICE = torch.device('cpu')
EPISODES = 1000
MAX_STEPS = 1000
HORIZON = 8
SAMPLES = 1024
ACTION_BATCH = None  # will be set after DEVICE is known

# Pre-allocated simulation env and step function per process
SIM_ENV = Vector2048Env(4 * SAMPLES, grid_size=4, device=DEVICE)
SIM_STEP = SIM_ENV.step

def mpc_policy(env, state, horizon, samples, device):
    """
    Vectorized MPC policy: simulate rollouts and pick the action with highest average reward.
    """
    grid_size = env.grid_size
    # simulate 4*samples parallel trajectories
    sim = Vector2048Env(4 * samples, grid_size, device)
    # broadcast current state
    sim.state = state.repeat(4 * samples, 1, 1)
    sim.done = torch.zeros(4 * samples, dtype=torch.bool, device=device)
    actions = torch.arange(4, device=device)
    init_act = actions.repeat_interleave(samples)
    _, rew0, _ = sim.step(init_act)
    accum = rew0
    # random rollout for horizon-1 steps
    for _ in range(horizon - 1):
        rand_acts = torch.randint(0, 4, (4 * samples,), device=device)
        _, r_h, _ = sim.step(rand_acts)
        accum = accum + r_h
    # pick best action
    avg_vals = accum.view(4, samples).mean(dim=1)
    best_action = int(avg_vals.argmax().item())
    # ensure valid move
    # Ensure state has batch dimension for _move
    if state.dim() == 2:
        state_batch = state.unsqueeze(0)
    else:
        state_batch = state
    moved, _ = env._move(state_batch, best_action)
    changed = (moved != state_batch).reshape(state_batch.shape[0], -1).any(dim=1)
    if not changed.all():
        for a in range(4):
            mv, _ = env._move(state_batch, a)
            if (mv != state_batch).reshape(state_batch.shape[0], -1).any(dim=1):
                best_action = a
                break
    return best_action

def run_episode(_):
    """
    Run one full episode of MPC and return the final tile.
    Uses the mpc_policy (as in evaluate.py) for each step until terminal.
    """
    env_proc = Vector2048Env(1, grid_size=4, device=DEVICE)
    state_batch, done_batch = env_proc.reset()
    state = state_batch[0]
    done = done_batch[0]
    step = 0
    while not done.item() and step < MAX_STEPS:
        # select action via mpc_policy
        action = mpc_policy(env_proc, state, HORIZON, SAMPLES, DEVICE)
        action_tensor = torch.tensor([action], dtype=torch.int64, device=DEVICE)
        state_batch, _, done_batch = env_proc.step(action_tensor)
        state = state_batch[0]
        done = done_batch[0]
        step += 1
    # return the maximum tile reached
    return int(state.max().item())

def main():
    global ACTIONS
    if ACTION_BATCH is None:
        ACTIONS = torch.arange(4, device=DEVICE)

    final_tiles = []
    start_time = time.time()

    # Run episodes in parallel on CPU, serial with tqdm on GPU
    if DEVICE.type == 'cpu':
        # determine number of worker processes, cap at 60 on Windows
        max_procs = cpu_count()
        if platform.system().lower() == 'windows':
            max_procs = min(max_procs, 60)
        with Pool(processes=max_procs) as pool:
            for tile in tqdm(pool.imap(run_episode, range(EPISODES)), total=EPISODES, desc='MPC Episodes'):
                final_tiles.append(tile)
    else:
        for ep in range(1, EPISODES + 1):
            # reset game
            env_proc = Vector2048Env(1, grid_size=4, device=DEVICE)
            state_batch, done_batch = env_proc.reset()
            state = state_batch[0]
            done = done_batch[0]
            step = 0
            total_reward = 0.0
            # play until game over or max_steps
            while not done.item() and step < MAX_STEPS:
                # compute MPC action
                act = mpc_policy(env_proc, state, HORIZON, SAMPLES, DEVICE)
                act_tensor = torch.tensor([act], dtype=torch.int64, device=DEVICE)
                state_batch, rew_batch, done_batch = env_proc.step(act_tensor)
                reward = rew_batch.item()
                total_reward += reward
                state = state_batch[0]
                done = done_batch[0]
                step += 1
            # record final tile
            final_tile = int(state.max().item())
            final_tiles.append(final_tile)
            print(f"Episode {ep}/{EPISODES}: final_tile={final_tile}")

    elapsed = time.time() - start_time
    # compute distribution and success rate
    counter = Counter(final_tiles)
    tiles = sorted(counter.keys())
    counts = [counter[t] for t in tiles]
    success_count = sum(c for t, c in counter.items() if t >= 2048)
    success_rate = success_count / EPISODES
    print(f"\nMPC final tile distribution over {EPISODES} episodes:")
    for t, c in zip(tiles, counts):
        print(f"  tile {t}: {c}")
    print(f"Success rate (>=2048): {success_rate:.2%}")
    # plot bar chart
    plt.figure()
    plt.bar([str(t) for t in tiles], counts)
    plt.xlabel('Final tile')
    plt.ylabel('Count')
    plt.title(f'MPC Final Tile Distribution over {EPISODES} episodes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('mpc_final_tile_distribution.png')
    print("Saved bar chart to mpc_final_tile_distribution.png")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MPC rollout for 2048')
    parser.add_argument('--episodes', type=int, default=1000, help='number of episodes')
    parser.add_argument('--horizon', type=int, default=8, help='planning horizon')
    parser.add_argument('--samples', type=int, default=1024, help='number of rollouts per action')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu', help='device to run on')
    args = parser.parse_args()
    # Override defaults
    DEVICE = torch.device(args.device)
    EPISODES = args.episodes
    HORIZON = args.horizon
    SAMPLES = args.samples
    ACTIONS = torch.arange(4, device=DEVICE)
    main() 