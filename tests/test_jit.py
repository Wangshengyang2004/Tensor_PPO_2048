import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector2048_env import Vector2048Env


def main():
    print("Testing torch.compile of Vector2048Env.step...")
    # instantiate environment
    device = torch.device('cuda')
    env = Vector2048Env(8, 4, device)
    # compile the step function
    compiled_step = torch.compile(env.step)

    # test reset
    print("Calling reset()...")
    state, done = env.reset()
    print(f"state dtype: {state.dtype}, shape: {state.shape}")

    # sample random actions
    actions = torch.randint(0, 4, (8,), dtype=torch.int64, device=device)
    print("Calling compiled step() with random actions...")
    next_state, reward, done2 = compiled_step(actions)
    print(f"next_state dtype: {next_state.dtype}, shape: {next_state.shape}")
    print(f"reward dtype: {reward.dtype}, shape: {reward.shape}")
    print(f"done dtype: {done2.dtype}, shape: {done2.shape}")

    print("torch.compile env reset and step OK.")


if __name__ == '__main__':
    main() 