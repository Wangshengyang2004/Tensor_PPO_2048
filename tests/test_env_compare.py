import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector2048_env import Vector2048Env

def test_env_shapes_and_dtypes():
    """
    Ensure that the vectorized env returns tensors with correct shapes and dtypes on CPU (and GPU if available).
    """
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
    for device in devices:
        env = Vector2048Env(num_envs=8, grid_size=4, device=device)
        state, done = env.reset()
        assert state.shape == (8, 4, 4), f"State shape mismatch on {device}"
        assert done.shape == (8,), f"Done shape mismatch on {device}"
        assert state.dtype == torch.int64, f"State dtype mismatch on {device}"
        assert done.dtype == torch.bool, f"Done dtype mismatch on {device}"
        actions = torch.randint(0, 4, (8,), dtype=torch.int64, device=device)
        next_state, reward, done2 = env.step(actions)
        assert next_state.shape == (8, 4, 4), f"Next state shape mismatch on {device}"
        assert reward.shape == (8,), f"Reward shape mismatch on {device}"
        assert done2.shape == (8,), f"Done2 shape mismatch on {device}"


def test_cpu_gpu_step_equivalence():
    """
    Compare outputs of the same Vector2048Env on CPU vs GPU for identical states and actions.
    """
    if not torch.cuda.is_available():
        return
    cpu = torch.device('cpu')
    gpu = torch.device('cuda')

    # Create a fixed test state with known values (avoiding randomness completely)
    test_state = torch.zeros((8, 4, 4), dtype=torch.int64)
    # Place some tiles in known positions
    test_state[0, 0, 0] = 2
    test_state[0, 0, 3] = 2
    test_state[1, 1, 1] = 4
    test_state[1, 2, 2] = 4
    test_state[2, 3, 0] = 8
    test_state[2, 3, 3] = 8
    test_state[3, 0, 0] = 16
    test_state[3, 0, 1] = 16
    test_state[4, 1, 2] = 32
    test_state[4, 1, 3] = 32
    test_state[5, 2, 0] = 64
    test_state[5, 2, 1] = 64
    test_state[6, 3, 2] = 128
    test_state[6, 3, 3] = 128
    test_state[7, 0, 0] = 256
    test_state[7, 0, 1] = 256

    # instantiate on both devices
    env_cpu = Vector2048Env(num_envs=8, grid_size=4, device=cpu)
    env_gpu = Vector2048Env(num_envs=8, grid_size=4, device=gpu)

    # Set the states directly
    env_cpu.state = test_state.clone()
    env_gpu.state = test_state.clone().to(gpu)

    # Test all actions
    for action in range(4):
        # Use the internal _move method directly to avoid random tile placement
        cpu_state, cpu_reward = env_cpu._move(env_cpu.state, action)
        gpu_state, gpu_reward = env_gpu._move(env_gpu.state, action)
        
        # Convert to CPU and same dtype for comparison
        gpu_state_cpu = gpu_state.cpu().to(cpu_state.dtype)
        gpu_reward_cpu = gpu_reward.cpu().to(cpu_reward.dtype)
        
        # Compare results
        assert torch.equal(cpu_state, gpu_state_cpu), f"State mismatch for action {action}"
        assert torch.allclose(cpu_reward, gpu_reward_cpu), f"Reward mismatch for action {action}"

    # Test done flag calculation
    done_cpu = env_cpu._check_done(env_cpu.state)
    done_gpu = env_gpu._check_done(env_gpu.state)
    assert torch.equal(done_cpu, done_gpu.cpu()), "Done flags mismatch" 