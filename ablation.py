import time
import torch
import matplotlib.pyplot as plt
from vector2048_env import Vector2048Env

# --- Embedded benchmark utilities ---
def run_episode(step_fn, env, rollout_length):
    # reset environment
    state, done = env.reset()
    batch_size = env.num_envs
    for _ in range(rollout_length):
        actions = torch.randint(0, 4, (batch_size,), dtype=torch.int64, device=env.device)
        state, _, done = step_fn(actions)

def benchmark(env_constructor, name, episodes, rollout_length, compiled=False):
    env = env_constructor()
    # coerce device to torch.device if given as string
    if not isinstance(env.device, torch.device):
        env.device = torch.device(env.device)
    # choose step function
    if compiled:
        try:
            step_fn = torch.compile(env.step)
        except Exception:
            step_fn = env.step
    else:
        step_fn = env.step
    # warm-up
    run_episode(step_fn, env, rollout_length)
    if env.device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for ep in range(episodes):
        start = time.time()
        run_episode(step_fn, env, rollout_length)
        if env.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"[{name}]{' [compiled]' if compiled else ''} Episode {ep+1}/{episodes}: {elapsed:.4f}s")
    avg = sum(times) / len(times)
    print(f"--> {name}{' [compiled]' if compiled else ''} average: {avg:.4f}s over {episodes} episodes")
    print()
    return avg
# --- End embedded utilities ---

def main():
    # Ablation settings: environment‐level optimizations
    episodes = 5
    rollout_length = 100
    batch_size = 1024

    device_types = ['cpu']
    if torch.cuda.is_available():
        device_types.append('cuda')

    # Define conditions: (name, constructor, compiled)
    conditions = []
    # CPU LUT
    conditions.append(("LUT-CPU", lambda: Vector2048Env(batch_size, 4, torch.device('cpu')), False))
    if 'cuda' in device_types:
        # GPU LUT
        conditions.append(("LUT-GPU", lambda: Vector2048Env(batch_size, 4, torch.device('cuda')), False))
        # GPU Triton
        conditions.append(("Triton-GPU", lambda: Vector2048Env(batch_size, 4, torch.device('cuda')), False))
        # compiled variants
        conditions.append(("LUT-GPU-COMP", lambda: Vector2048Env(batch_size, 4, torch.device('cuda')), True))
        conditions.append(("Triton-GPU-COMP", lambda: Vector2048Env(batch_size, 4, torch.device('cuda')), True))

    print("\n=== Environment Ablation Study ===")
    # prepare for plotting
    cond_names = []
    avg_times = []
    for name, constructor, compiled in conditions:
        # extract use_triton flag and device from a quick env instance
        env = constructor()
        use_triton = env.use_triton
        device = env.device
        # build new constructor capturing flags
        def make_env(use_triton=use_triton, device=device):
            e = Vector2048Env(batch_size, 4, device)
            e.use_triton = use_triton
            return e

        print(f"\n-- {name}{' [compiled]' if compiled else ''} --")
        avg = benchmark(make_env, name, episodes, rollout_length, compiled)
        cond_names.append(name + (' [C]' if compiled else ''))
        avg_times.append(avg)

    # Plot average times as horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, len(cond_names)*0.5 + 1))
    ax.barh(cond_names, avg_times, color='skyblue')
    ax.set_xlabel('Avg Time (s)')
    ax.set_title('Environment Ablation: Average Episode Time')
    plt.tight_layout()
    plt.savefig('ablation_env_avg_time.png')
    print("Saved ablation plot to ablation_env_avg_time.png")

if __name__ == '__main__':
    main() 