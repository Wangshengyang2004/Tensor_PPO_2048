#!/usr/bin/env python3
# perf_test.py
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ppo2048 import PPOAgent

def benchmark(agent_constructor, device, num_envs, grid_size, steps=50):
    """
    Run a simple forward (and backward) benchmark to measure throughput.
    """
    # Instantiate and unwrap DDP if used
    agent = agent_constructor(grid_size=grid_size, num_envs=num_envs).to(device)
    # Determine the worker for custom methods
    worker = agent.module if isinstance(agent, DDP) else agent
    agent.eval()
    # random state inputs
    state = torch.randint(0, 16, (num_envs, grid_size, grid_size), device=device, dtype=torch.int64)
    # dummy optimizer for backward pass
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3)

    # Warm up
    for _ in range(10):
        with torch.no_grad():
            action, logp, entropy, val = worker.get_action_and_value(state)

    # Benchmark forward+backward
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        action, logp, entropy, val = worker.get_action_and_value(state)
        # small dummy loss: mean of value + sum of action log-prob
        loss = val.mean() - logp.mean() * 0.01
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if device.type=='cuda' else None
    elapsed = time.time() - start
    return elapsed

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_envs = 64
    grid_size = 4
    steps = 100

    print(f"Benchmarking on device: {device}, envs={num_envs}, steps={steps}")
    # Baseline
    t0 = benchmark(PPOAgent, device, num_envs, grid_size, steps)
    print(f"Baseline time: {t0:.4f}s, throughput: {steps/t0:.2f} updates/s")

    # Compiled
    print("Compiling agent with TorchInductor...")
    base_agent = PPOAgent(grid_size=grid_size, num_envs=num_envs).to(device)
    compiled_agent = torch.compile(base_agent, backend='inductor', fullgraph=True)
    def compiled_constructor(grid_size=None, num_envs=None):
        return compiled_agent
    t1 = benchmark(compiled_constructor, device, num_envs, grid_size, steps)
    print(f"Compiled time: {t1:.4f}s, throughput: {steps/t1:.2f} updates/s")

    # DDP-wrapped compiled model (single rank)
    print("\nInitializing single-rank DDP for compiled model...")
    # Initialize process group for single-rank DDP
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)
    def ddp_compiled_constructor(grid_size=None, num_envs=None):
        model = PPOAgent(grid_size=grid_size, num_envs=num_envs).to(device)
        compiled = torch.compile(model, backend='inductor', fullgraph=True)
        # Wrap in DDP; for CUDA, specify device_ids
        if device.type == 'cuda':
            return DDP(compiled, device_ids=[0], output_device=0)
        else:
            return DDP(compiled)
    t2 = benchmark(ddp_compiled_constructor, device, num_envs, grid_size, steps)
    print(f"Compiled+DDP time: {t2:.4f}s, throughput: {steps/t2:.2f} updates/s")
    # Clean up process group
    dist.destroy_process_group()

    print("Done.") 