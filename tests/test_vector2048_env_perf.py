import torch
import sys, os
import time

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector2048_env import Vector2048Env

# Benchmarking env.step performance

def test_step_performance():
    """Benchmark env.step for LUT vs Triton (plain and compiled on CUDA)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid_size = 4
    num_envs = 1024
    reps = 500

    # Prepare a fixed batch of random actions
    actions = torch.randint(0, 4, (num_envs,), device=device, dtype=torch.int64)

    results = {}
    for mode in ['lut', 'triton']:
        # Initialize environment
        env = Vector2048Env(num_envs, grid_size, device)
        if mode == 'triton':
            env.use_triton = True
            env._move = env._move_triton
        else:
            env.use_triton = False
            env._move = env._move_lut

        for compiled in [False, True]:
            key = f"{mode}_{'compiled' if compiled else 'plain'}"
            # Optionally compile env.step for CUDA
            if compiled and device.type == 'cuda':
                env.step = torch.compile(env.step, backend='inductor')

            # Warmup
            for _ in range(10):
                _ = env.step(actions)

            # Timing
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(reps):
                    _ = env.step(actions)
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
            else:
                t0 = time.time()
                for _ in range(reps):
                    _ = env.step(actions)
                elapsed_ms = (time.time() - t0) * 1000

            per_call = elapsed_ms / reps
            results[key] = per_call
            print(f"{key}: {per_call:.3f} ms per step")

    # Report compile speedups and kernel speedup
    for mode in ['lut', 'triton']:
        plain = results[f"{mode}_plain"]
        comp = results[f"{mode}_compiled"]
        speedup = (plain - comp) / plain * 100 if comp < plain else 0.0
        print(f"{mode} compile speedup: {speedup:.1f}%")

    lut_plain = results['lut_plain']
    tri_plain = results['triton_plain']
    speedup = (lut_plain - tri_plain) / lut_plain * 100 if tri_plain < lut_plain else 0.0
    print(f"Triton vs LUT speedup: {speedup:.1f}%")


if __name__ == '__main__':
    test_step_performance() 