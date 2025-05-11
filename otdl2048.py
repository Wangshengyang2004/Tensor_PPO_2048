import os
import torch
torch._dynamo.config.suppress_errors = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from vector2048_env import Vector2048Env
import random
import numpy as np


class NTupleNetwork:
    """
    N-Tuple network for 2048, with optimistic initialization.
    """
    def __init__(self, grid_size=4, tuple_count=100, tuple_length=4, init_value=1000.0):
        # grid_size: board dimension
        # tuple_count: number of random N-tuples
        # tuple_length: cells per tuple
        self.grid_size = grid_size
        self.tuple_count = tuple_count
        self.tuple_length = tuple_length
        # generate random tuple indices (flattened positions)
        positions = list(range(grid_size * grid_size))
        self.tuple_indices = [random.sample(positions, tuple_length) for _ in range(tuple_count)]
        # precompute bit shifts for packing exponents into index
        self.shifts = [4 * (tuple_length - 1 - i) for i in range(tuple_length)]
        # weight tables: one table per tuple, optimistic init
        size = 1 << (4 * tuple_length)
        self.weights = [torch.full((size,), init_value, dtype=torch.float32) for _ in range(tuple_count)]

    def value(self, state):
        """
        Compute estimated value for given board state(s).
        state: torch.Tensor of shape (batch, grid_size, grid_size) storing tile values
        Returns: torch.Tensor of shape (batch,)
        """
        # move to CPU for indexing into CPU weights
        st = state.detach().cpu().to(torch.int64)
        batch = st.shape[0]
        # compute exponents (0 for empty, else log2)
        exps = torch.where(st == 0,
                           torch.zeros_like(st, dtype=torch.int64),
                           torch.log2(st).to(torch.int64))
        flat = exps.view(batch, -1)  # (batch, grid_size*grid_size)
        # accumulate tuple contributions
        vals = torch.zeros(batch, dtype=torch.float32)
        for t in range(self.tuple_count):
            idxs = torch.zeros(batch, dtype=torch.int64)
            inds = self.tuple_indices[t]
            for j, pos in enumerate(inds):
                idxs |= (flat[:, pos] << self.shifts[j])
            vals += self.weights[t][idxs]
        return vals.to(state.device)

    def update(self, transitions, lr=0.1, gamma=0.99):
        """
        Perform Optimistic TD update over a list of transitions.
        transitions: list of (state, action, reward, next_state, done)
        lr: learning rate
        gamma: discount factor
        """
        for (s, a, r, s1, done) in transitions:
            # s and s1 are CPU numpy arrays of shape (grid_size, grid_size)
            st = torch.tensor(s, dtype=torch.int64).unsqueeze(0)
            s1t = torch.tensor(s1, dtype=torch.int64).unsqueeze(0)
            # current value and next value
            v_s = self.value(st)[0].item()
            v_s1 = 0.0 if done else self.value(s1t)[0].item()
            target = r + gamma * v_s1
            error = target - v_s
            # update each tuple weight
            flat = torch.where(st == 0,
                               torch.zeros_like(st, dtype=torch.int64),
                               torch.log2(st).to(torch.int64)).view(-1)
            for t in range(self.tuple_count):
                idx = 0
                for j, pos in enumerate(self.tuple_indices[t]):
                    idx |= (int(flat[pos].item()) << self.shifts[j])
                # optimistic update
                self.weights[t][idx] += lr * error


def select_action(net, state, env, device):
    """
    Choose action by evaluating afterstates.
    state: torch.Tensor shape (grid_size, grid_size)
    env: Vector2048Env instance (will use _move without spawning tiles)
    """
    best_val = -float('inf')
    best_act = 0
    x = state.unsqueeze(0)  # shape (1,grid,grid)
    for a in range(4):
        # compute afterstate (no spawn) via private _move
        after, _ = env._move(x, a)
        val = net.value(after)[0].item()
        if val > best_val:
            best_val = val
            best_act = a
    return best_act


def main():
    # Distributed setup
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')
    # device selection
    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Hyperparameters
    episodes = 1000
    max_steps = 1000
    lr = 0.1
    gamma = 0.99
    tuple_count = 100
    tuple_length = 4
    init_value = 1000.0

    # Environment
    env = Vector2048Env(1, grid_size=4, device=device)
    # Agent
    net = NTupleNetwork(grid_size=4, tuple_count=tuple_count,
                        tuple_length=tuple_length, init_value=init_value)
    if distributed:
        net = DDP(net)

    rank0 = not distributed or local_rank == 0
    if rank0:
        writer = SummaryWriter(log_dir='runs/otdl')

    for ep in trange(1, episodes+1, desc='OTDL Episodes'):
        state, done = env.reset()
        total_reward = 0.0
        # collect transitions
        transitions = []
        for step in range(max_steps):
            act = select_action(net, state[0], env, device)
            action = torch.tensor([act], dtype=torch.int64, device=device)
            next_state, reward, done = env.step(action)
            transitions.append((state[0].cpu(), act, reward.item(), next_state[0].cpu(), done[0].item()))
            state = next_state
            total_reward += reward.item()
            if done.any():
                break
        # Update network
        net.update(transitions, lr=lr, gamma=gamma)

        # Logging
        if rank0:
            writer.add_scalar('Performance/total_reward', total_reward, ep)
            max_tile = state.max().item()
            writer.add_scalar('Performance/max_tile', max_tile, ep)
            print(f"Episode {ep}: Total Reward={total_reward:.2f}, Max Tile={max_tile}")

    if rank0:
        torch.save(net.state_dict(), 'otdl2048.pth')
        writer.close()


if __name__ == '__main__':
    main() 