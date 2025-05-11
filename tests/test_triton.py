import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector2048_env import Vector2048Env
from triton_kernels import compute_gae_advantages, compute_shaped_rewards

def test_move_batch(num_envs=8, grid_size=4, device=None):
    # device: CUDA required for Triton
    device = torch.device('cuda' if (device is None and torch.cuda.is_available()) else (device or 'cpu'))
    # generate random exponents 0-11 and map to tile values
    exps = torch.randint(0, 12, (num_envs, grid_size, grid_size), device=device)
    state = torch.where(exps == 0,
                        torch.zeros_like(exps, dtype=torch.int64),
                        (2 ** exps).to(torch.int64))
    env = Vector2048Env(num_envs, grid_size, device)
    # force use Triton
    env.use_triton = True

    print(f"Testing Triton vs LUT moves on device {device}, batch size {num_envs}")
    for action in range(4):
        tr_out, tr_rew = env._move_triton(state, action)
        lut_out, lut_rew = env._move_lut(state, action)
        eq_state = torch.equal(tr_out, lut_out)
        eq_rew = torch.allclose(tr_rew, lut_rew)
        result = "PASS" if eq_state and eq_rew else "FAIL"
        print(f"Action {action}: {result}")
        if not (eq_state and eq_rew):
            print("Triton state:\n", tr_out)
            print("LUT state:\n", lut_out)
            print("Triton reward:\n", tr_rew)
            print("LUT reward:\n", lut_rew)
            break
    # Also test the full env.step behavior
    # Create separate envs for Triton vs LUT full-step
    env_tr = Vector2048Env(num_envs, grid_size, device)
    env_lut = Vector2048Env(num_envs, grid_size, device)
    # Reset to same initial state
    state_tr, done_tr = env_tr.reset()
    state_lut, done_lut = env_lut.reset()
    # Copy initial state to ensure identical starts
    env_lut.state = state_tr.clone()
    env_lut.done = done_tr.clone()
    # Force move implementations
    env_tr.use_triton = True; env_tr._move = env_tr._move_triton
    env_lut.use_triton = False; env_lut._move = env_lut._move_lut
    print("Testing full env.step consistency:")
    for action in range(4):
        act_tensor = torch.tensor([action], device=device, dtype=torch.int64)
        nxt_tr, rew_tr, dn_tr = env_tr.step(act_tensor)
        nxt_l, rew_l, dn_l = env_lut.step(act_tensor)
        eq_state2 = torch.equal(nxt_tr, nxt_l)
        eq_rew2 = torch.allclose(rew_tr, rew_l)
        eq_done2 = torch.equal(dn_tr, dn_l)
        result2 = "PASS" if (eq_state2 and eq_rew2 and eq_done2) else "FAIL"
        print(f"Step Action {action}: {result2}")
        if not (eq_state2 and eq_rew2 and eq_done2):
            print("Triton step state:\n", nxt_tr)
            print("LUT step state:\n", nxt_l)
            print("Triton step reward:\n", rew_tr)
            print("LUT step reward:\n", rew_l)
            print("Triton done:\n", dn_tr)
            print("LUT done:\n", dn_l)
            break
    # return for performance benchmarking
    return env, state


def test_gae_computation():
    # Test parameters
    T, N = 5, 3  # Small test case for verification
    gamma = 0.99
    gae_lambda = 0.95
    
    # Create test data
    rewards = torch.tensor([
        [1.0, 0.5, 0.8],
        [0.0, 1.0, 0.3],
        [0.5, 0.2, 0.0],
        [0.8, 0.4, 1.0],
        [0.3, 0.6, 0.2]
    ], device='cuda')
    
    values = torch.tensor([
        [1.1, 0.6, 0.9],
        [0.1, 1.1, 0.4],
        [0.6, 0.3, 0.1],
        [0.9, 0.5, 1.1],
        [0.4, 0.7, 0.3]
    ], device='cuda')
    
    dones = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ], device='cuda')
    
    next_value = torch.tensor([0.2, 0.8, 0.5], device='cuda')
    
    # Warmup
    for _ in range(10):
        compute_gae_advantages(rewards, values, dones, next_value, gamma, gae_lambda)
        
        advantages_ref = torch.zeros_like(rewards)
        returns_ref = torch.zeros_like(rewards)
        last_gae = torch.zeros(N, device='cuda')
        next_val = next_value
        for t in reversed(range(T)):
            mask_t = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * mask_t - values[t]
            last_gae = delta + gamma * gae_lambda * mask_t * last_gae
            advantages_ref[t] = last_gae
            returns_ref[t] = advantages_ref[t] + values[t]
            next_val = values[t]
    
    # Time Triton implementation
    reps = 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(reps):
        advantages, returns = compute_gae_advantages(
            rewards, values, dones, next_value,
            gamma, gae_lambda
        )
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end)
    
    # Time naive implementation
    start.record()
    for _ in range(reps):
        advantages_ref = torch.zeros_like(rewards)
        returns_ref = torch.zeros_like(rewards)
        last_gae = torch.zeros(N, device='cuda')
        next_val = next_value
        for t in reversed(range(T)):
            mask_t = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * mask_t - values[t]
            last_gae = delta + gamma * gae_lambda * mask_t * last_gae
            advantages_ref[t] = last_gae
            returns_ref[t] = advantages_ref[t] + values[t]
            next_val = values[t]
    end.record()
    torch.cuda.synchronize()
    naive_ms = start.elapsed_time(end)
    
    # Check results
    torch.testing.assert_close(advantages, advantages_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(returns, returns_ref, rtol=1e-5, atol=1e-5)
    print("GAE computation test passed!")
    print(f"Triton GAE average time: {triton_ms/reps:.3f} ms")
    print(f"Naive GAE average time: {naive_ms/reps:.3f} ms")
    print(f"GAE speedup: {(naive_ms - triton_ms) / naive_ms * 100:.1f}%")

def test_reward_shaping():
    # Test parameters
    N, H, W = 3, 4, 4  # Small grid for testing
    
    # Create test data
    state = torch.zeros((N, H, W), device='cuda')
    next_state = torch.zeros((N, H, W), device='cuda')
    
    # Set up some test scenarios
    # Env 0: More empty cells in next state
    state[0] = torch.tensor([
        [2, 4, 0, 0],
        [2, 0, 4, 0],
        [4, 2, 0, 2],
        [2, 4, 2, 4]
    ], device='cuda')
    
    next_state[0] = torch.tensor([
        [2, 4, 0, 0],
        [2, 0, 0, 0],
        [4, 2, 0, 2],
        [2, 4, 2, 4]
    ], device='cuda')
    
    # Env 1: Higher max tile in next state
    state[1] = torch.tensor([
        [2, 4, 8, 4],
        [4, 8, 4, 2],
        [2, 4, 8, 4],
        [4, 2, 4, 8]
    ], device='cuda')
    
    next_state[1] = torch.tensor([
        [2, 4, 16, 4],
        [4, 8, 4, 2],
        [2, 4, 8, 4],
        [4, 2, 4, 8]
    ], device='cuda')
    
    # Env 2: No change (waste move)
    state[2] = torch.tensor([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2]
    ], device='cuda')
    
    next_state[2] = state[2].clone()
    
    raw_rewards = torch.tensor([1.0, 2.0, 0.0], device='cuda')
    
    # Define naive reward shaping implementation
    def naive_reward_shaping(state, next_state, raw_rewards):
        N = state.shape[0]
        shaped_rewards = raw_rewards.clone()
        empty_bonus = torch.zeros_like(raw_rewards)
        max_bonus = torch.zeros_like(raw_rewards)
        waste_penalty = torch.zeros_like(raw_rewards)
        
        for n in range(N):
            # Count empty cells
            empties0 = (state[n] == 0).sum().float()
            empties1 = (next_state[n] == 0).sum().float()
            empty_bonus[n] = 0.01 * (empties1 - empties0)
            
            # Check max tile
            max0 = state[n].max()
            max1 = next_state[n].max()
            max_bonus[n] = 1.0 * (max1 > max0).float()
            
            # Check if state changed
            waste_penalty[n] = -0.1 * (1.0 - (state[n] != next_state[n]).any().float())
            
            shaped_rewards[n] += empty_bonus[n] + max_bonus[n] + waste_penalty[n]
        
        return shaped_rewards, empty_bonus, max_bonus, waste_penalty
    
    # Warmup
    for _ in range(10):
        compute_shaped_rewards(state, next_state, raw_rewards)
        naive_reward_shaping(state, next_state, raw_rewards)
    
    # Time Triton implementation
    reps = 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(reps):
        shaped_rewards, empty_bonus, max_bonus, waste_penalty = compute_shaped_rewards(
            state, next_state, raw_rewards
        )
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end)
    
    # Time naive implementation
    start.record()
    for _ in range(reps):
        shaped_rewards_ref, empty_bonus_ref, max_bonus_ref, waste_penalty_ref = naive_reward_shaping(
            state, next_state, raw_rewards
        )
    end.record()
    torch.cuda.synchronize()
    naive_ms = start.elapsed_time(end)
    
    # Verify results
    # Env 0: Should have positive empty bonus
    assert empty_bonus[0] > 0
    assert max_bonus[0] == 0
    assert waste_penalty[0] == 0
    
    # Env 1: Should have max tile bonus
    assert empty_bonus[1] == 0
    assert max_bonus[1] > 0
    assert waste_penalty[1] == 0
    
    # Env 2: Should have waste penalty
    assert empty_bonus[2] == 0
    assert max_bonus[2] == 0
    assert waste_penalty[2] < 0
    
    print("Reward shaping test passed!")
    print(f"Triton reward shaping average time: {triton_ms/reps:.3f} ms")
    print(f"Naive reward shaping average time: {naive_ms/reps:.3f} ms")
    print(f"Reward shaping speedup: {(naive_ms - triton_ms) / naive_ms * 100:.1f}%")


if __name__ == '__main__':
    env, state = test_move_batch()
    # Performance comparison
    try:
        import time
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            reps = 100
            # warmup
            for _ in range(10):
                for action in range(4):
                    env._move_triton(state, action)
                    env._move_lut(state, action)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # Triton timing
            start.record()
            for _ in range(reps):
                for action in range(4):
                    env._move_triton(state, action)
            end.record()
            torch.cuda.synchronize()
            triton_ms = start.elapsed_time(end)
            # LUT timing
            start.record()
            for _ in range(reps):
                for action in range(4):
                    env._move_lut(state, action)
            end.record()
            torch.cuda.synchronize()
            lut_ms = start.elapsed_time(end)
            print(f"Triton average per-row move time: {triton_ms/(reps*4):.3f} ms")
            print(f"LUT average per-row move time: {lut_ms/(reps*4):.3f} ms")
        else:
            print("Performance benchmark requires CUDA device.")
    except Exception as e:
        print(f"Performance benchmark failed: {e}")

    test_gae_computation()
    test_reward_shaping() 