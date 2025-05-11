# dqn2048.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from vector2048_env import Vector2048Env
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# Neural network for Q-value approximation
class DQN(nn.Module):
    def __init__(self, grid_size, hidden_size=256):
        super(DQN, self).__init__()
        # simple conv + MLP
        # after two 2x2 convs (stride=1), spatial dims = grid_size - 2
        conv_out = (grid_size - 2) * (grid_size - 2) * 64
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
    def forward(self, x):
        # x: (batch, grid, grid)
        x = x.unsqueeze(1).float()          # (batch,1,grid,grid)
        x = torch.log2(x + 1.0) / 11.0     # normalize
        return self.net(x)                # (batch,4)

# Simple replay buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        # store numpy arrays and scalars
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.int64),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.int64),
            torch.tensor(dones, dtype=torch.float32)
        )
    def __len__(self):
        return len(self.buffer)

# Select action with epsilon-greedy
def select_action(q_net, state, eps, device):
    if random.random() < eps:
        return random.randrange(4)
    with torch.no_grad():
        s = torch.tensor(state, device=device).unsqueeze(0)  # (1,grid,grid)
        q = q_net(s)                                       # (1,4)
        return int(q.argmax(dim=1).item())

# Optimize DQN
def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    if len(memory) < batch_size:
        return None
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Q(s,a)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # max_{a'} Q_target(next_s, a')
    next_q = target_net(next_states).max(1)[0].detach()
    # compute expected Q
    expected_q = rewards + gamma * next_q * (1 - dones)
    loss = F.mse_loss(q_values, expected_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Main training loop
def main():
    # Distributed setup
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    if distributed:
        # choose backend and ensure GPUs match process count
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if world_size > n_gpus:
                raise RuntimeError(f"World size ({world_size}) exceeds available GPUs ({n_gpus})")
            torch.cuda.set_device(local_rank)
            backend = 'nccl'
        else:
            backend = 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
    # Hyperparameters
    grid_size = 4
    num_episodes = 500
    max_steps = 500  # shorter horizon to limit stale data
    batch_size = 128
    replay_capacity = 10000
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.1
    eps_decay = 0.995
    target_update = 10
    # select device, respecting distributed local_rank
    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}' if distributed else 'cuda')
    elif torch.backends.mps.is_available() and not distributed:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Environment: batch of parallel boards
    global_num_envs = 1024  # reduced parallel envs to avoid stale transitions
    num_envs = global_num_envs // world_size
    env = Vector2048Env(num_envs, grid_size, device)

    # Networks
    policy_net = DQN(grid_size).to(device)
    if distributed:
        policy_net = DDP(policy_net, device_ids=[local_rank], output_device=local_rank)
    target_net = DQN(grid_size).to(device)
    # load weights from policy, adjusting for DDP if needed
    if distributed:
        target_net.load_state_dict(policy_net.module.state_dict())
    else:
        target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Compile for performance (optional, skip under DDP)
    if not distributed and hasattr(torch, 'compile'):
        try:
            policy_net = torch.compile(policy_net)
            target_net = torch.compile(target_net)
        except Exception:
            pass

    optimizer = optim.Adam(policy_net.parameters(), lr=5e-5)  # lower LR for stability
    memory = ReplayMemory(replay_capacity)
    eps = eps_start

    # setup TensorBoard on rank0
    rank0 = not distributed or local_rank == 0
    if rank0:
        writer = SummaryWriter()
    # track loss for reporting
    loss_sum = 0.0
    loss_count = 0

    for ep in trange(1, num_episodes+1, desc="Episodes"):
        # reset all boards
        state, done = env.reset()  # state: (num_envs, grid, grid)
        # track cumulative rewards per board
        total_reward = torch.zeros(num_envs, device=device)
        for t in range(max_steps):
            # compute Q-values and select actions ε-greedy in batch
            q_vals = policy_net(state.to(device))               # (num_envs,4)
            rand = torch.rand(num_envs, device=device)
            greedy_actions = q_vals.argmax(dim=1)
            random_actions = torch.randint(0, 4, (num_envs,), device=device)
            actions = torch.where(rand < eps, random_actions, greedy_actions)
            # step all boards
            next_states, rewards, dones = env.step(actions)
            # collect transitions per board
            for i in range(num_envs):
                memory.push(
                    state[i].cpu().numpy(),
                    int(actions[i].item()),
                    float(rewards[i].item()),
                    next_states[i].cpu().numpy(),
                    bool(dones[i].item())
                )
            state = next_states
            total_reward += rewards

            loss_val = optimize_model(policy_net, target_net, memory, optimizer,
                           batch_size, gamma, device)
            if loss_val is not None:
                loss_sum += loss_val
                loss_count += 1
            # if all boards done, end this episode early
            if dones.all():
                break

        # decay epsilon
        eps = max(eps_end, eps * eps_decay)
        # average return across batch
        avg_return = total_reward.mean().item()
        # episode logging
        if rank0:
            writer.add_scalar('Performance/avg_return', avg_return, ep)
            writer.add_scalar('Performance/max_tile', state.max().item(), ep)
            writer.add_scalar('Train/epsilon', eps, ep)
            if loss_count > 0:
                writer.add_scalar('Loss/td_loss', loss_sum / loss_count, ep)
        # reset stats
        loss_sum = 0.0
        loss_count = 0

        # update target network
        if ep % target_update == 0:
            # sync weights from policy to target, handle DDP wrapper
            if distributed:
                target_net.load_state_dict(policy_net.module.state_dict())
            else:
                target_net.load_state_dict(policy_net.state_dict())

        if not distributed or local_rank == 0:
            max_tile = state.max().item()
            print(f"Episode {ep}: Avg Total Reward = {avg_return:.2f}, Max Tile = {max_tile}, Epsilon = {eps:.3f}")

    if rank0:
        writer.close()

    if not distributed or local_rank == 0:
        torch.save(policy_net.state_dict(), 'dqn2048.pth')

if __name__ == '__main__':
    main() 