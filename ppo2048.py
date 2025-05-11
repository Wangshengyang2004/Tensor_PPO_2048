# ppo2048.py
import argparse
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from vector2048_env import Vector2048Env
from triton_kernels import compute_shaped_rewards, compute_gae_advantages
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.amp import GradScaler
import random
import numpy as np

# Enable CuDNN autotuner and TF32 tensor cores for float32 matmuls
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

"""
PPO training script for 2048. To run standalone with composite reward shaping:
  python ppo2048.py --com_reward
"""

class Critic(nn.Module):
    def __init__(self, grid_size=4, hidden_size=256):
        super().__init__()
        # Input processing and encoding
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(256)
        
        # Calculate flattened size for fully connected layers
        conv_out = grid_size * grid_size * 256
        
        self.linear1 = nn.Linear(conv_out, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        # Ensure proper dimensionality
        if len(x.shape) == 3:  # (batch, grid, grid)
            x = x.unsqueeze(1).float()  # add channel -> (batch, 1, grid, grid)
        
        # Log2 normalization of tile values
        x = torch.log2(x + 1.0) / 11.0
        
        # Convolutional layers
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm2(x)
        
        # Flatten and feed through fully connected layers
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        value = self.value_head(x)
        
        return value.squeeze(-1)


class Actor(nn.Module):
    def __init__(self, grid_size=4, hidden_size=256, num_actions=4):
        super().__init__()
        # Input processing and encoding
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(256)
        
        # Calculate flattened size for fully connected layers
        conv_out = grid_size * grid_size * 256
        
        self.linear1 = nn.Linear(conv_out, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 64)
        self.policy_head = nn.Linear(64, num_actions)

    def forward(self, x, legal_actions=None):
        # Ensure proper dimensionality
        if len(x.shape) == 3:  # (batch, grid, grid)
            x = x.unsqueeze(1).float()  # add channel -> (batch, 1, grid, grid)
        
        # Log2 normalization of tile values
        x = torch.log2(x + 1.0) / 11.0
        
        # Convolutional layers
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm2(x)
        
        # Flatten and feed through fully connected layers
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        logits = self.policy_head(x)
        
        # Apply action masking for illegal moves if provided
        if legal_actions is not None:
            # Mask out illegal actions with -inf (preserve dtype)
            logits = torch.where(legal_actions == 0.0,
                                 torch.full_like(logits, -float('inf')),
                                 logits)
        
        return nn.functional.log_softmax(logits, dim=-1)


class PPOAgent(nn.Module):
    def __init__(self, grid_size=4, num_envs=1, hidden_size=256):
        super(PPOAgent, self).__init__()
        self.grid_size = grid_size
        self.num_actions = 4
        self.num_envs = num_envs
        
        # Separate actor and critic networks
        self.actor = Actor(grid_size, hidden_size, self.num_actions)
        self.critic = Critic(grid_size, hidden_size)
        # Share convolutional backbone between actor and critic to save compute
        self.critic.conv1 = self.actor.conv1
        self.critic.batch_norm1 = self.actor.batch_norm1
        self.critic.conv2 = self.actor.conv2
        self.critic.batch_norm2 = self.actor.batch_norm2
        
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None, legal_actions=None):
        # Get action logits from actor
        log_probs = self.actor(x, legal_actions)
        
        # Convert log_probs to distribution
        policy_dist = Categorical(logits=log_probs)
        
        # Sample action if not provided
        if action is None:
            action = policy_dist.sample()
        
        # Get value estimate from critic
        value = self.critic(x)
        
        return action, policy_dist.log_prob(action), policy_dist.entropy(), value
    
    def forward(self, x):
        # This is for backward compatibility with the original code
        log_probs = self.actor(x)
        policy_dist = Categorical(logits=log_probs)
        values = self.critic(x)
        return log_probs, values


def log_model_info(agent, writer):
    """Log model architecture and parameters to TensorBoard."""
    if writer is not None:
        # Unwrap DistributedDataParallel if needed
        core = agent.module if hasattr(agent, 'module') else agent
        # Log model architecture and parameter counts for actor and critic
        actor_params = sum(p.numel() for p in core.actor.parameters())
        critic_params = sum(p.numel() for p in core.critic.parameters())
        writer.add_text('Model/actor_arch', str(core.actor), 0)
        writer.add_text('Model/critic_arch', str(core.critic), 0)
        writer.add_scalar('Model/actor_param_count', actor_params, 0)
        writer.add_scalar('Model/critic_param_count', critic_params, 0)

def log_training_metrics(writer, ep, rollout_return, max_tile, policy_loss_avg, value_loss_avg, 
                       entropy_avg, clipfrac_avg, avg_kl, agent, optimizer, val_est):
    """Log training metrics to TensorBoard."""
    if writer is not None:
        writer.add_scalar('Performance/rollout_return', rollout_return, ep)
        writer.add_scalar('Performance/max_tile', max_tile, ep)
        writer.add_scalar('Loss/policy', policy_loss_avg, ep)
        writer.add_scalar('Loss/value', value_loss_avg, ep)
        writer.add_scalar('Loss/entropy', entropy_avg, ep)
        writer.add_scalar('Loss/clipfrac', clipfrac_avg, ep)
        writer.add_scalar('Stats/approx_kl', avg_kl, ep)
        
        # Use underlying model if wrapped in DDP
        model_for_norm = agent.module if isinstance(agent, DDP) else agent
        grad_norm = torch.sqrt(sum(p.grad.data.norm()**2 for p in model_for_norm.parameters() if p.grad is not None))
        param_norm = torch.sqrt(sum(p.data.norm()**2 for p in model_for_norm.parameters()))
        writer.add_scalar('Grad/norm', grad_norm.item(), ep)
        writer.add_scalar('Params/norm', param_norm.item(), ep)
        
        # Learning rate
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], ep)
        
        # Weight norms
        writer.add_scalar('Weights/actor_head_weight_norm', model_for_norm.actor.policy_head.weight.data.norm().item(), ep)
        writer.add_scalar('Weights/actor_head_bias_norm', model_for_norm.actor.policy_head.bias.data.norm().item(), ep)
        writer.add_scalar('Weights/critic_head_weight_norm', model_for_norm.critic.value_head.weight.data.norm().item(), ep)
        writer.add_scalar('Weights/critic_head_bias_norm', model_for_norm.critic.value_head.bias.data.norm().item(), ep)
        
        # Value estimates
        writer.add_scalar('Value/mean_estimate', val_est.mean().item(), ep)

def train(agent, use_shaping=False, seed=42, rollout_length=128, global_num_envs=2048, log_interval=10, writer=None, max_tile=None):
    # Distributed setup
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    # Unwrap DDP for method calls
    core_agent = agent.module if hasattr(agent, 'module') else agent
    # only rank 0 shows progress and log to TensorBoard
    rank0 = not distributed or local_rank == 0
    # initialize TensorBoard writer (only on rank0)
    if rank0:
        if writer is None:
            print("Warning: No TensorBoard writer provided, logging will be disabled")

    # Hyperparameters (can be overridden by CLI)
    grid_size = 4
    lr = 3e-5            # lower LR for stability
    epochs = 200000      # more training
    gamma = 0.998        # higher discount to reward long-term gains (increased)
    clip_eps = 0.2       # tighter PPO clipping for stability
    gae_lambda = 0.95    # smoother GAE (adjusted)
    ent_coef = 0.01      # entropy coefficient (to encourage exploration)
    vf_coef = 0.5        # value function coefficient

    # per-process env count and device selection
    num_envs = global_num_envs // world_size
    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}' if distributed else 'cuda')
    elif torch.backends.mps.is_available() and not distributed:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # instantiate the environment directly
    env = Vector2048Env(num_envs, grid_size, device)
    
    if rank0 and writer is not None:
        # Log model architecture and parameter counts for actor and critic
        log_model_info(agent, writer)

    # set up automatic mixed precision scaler
    scaler = GradScaler(device=device)
    optimizer = optim.AdamW(agent.parameters(), lr=lr)  # Switch to AdamW
    # add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Use cosine annealing
    # checkpoint interval: save 10 times throughout training
    checkpoint_interval = max(1, epochs // 10)

    state, done = env.reset()
    # Track best tile seen during training
    historical_max_tile = 0
    
    # progress iterator
    if rank0:
        ep_iter = tqdm(range(epochs), desc="Episodes")
    else:
        ep_iter = range(epochs)

    # Compile actor and critic modules (unwrap DDP if necessary)
    core_agent = agent.module if hasattr(agent, 'module') else agent
    try:
        core_agent.actor = torch.compile(core_agent.actor, backend='inductor', fullgraph=True)
        core_agent.critic = torch.compile(core_agent.critic, backend='inductor', fullgraph=True)
        print("Successfully compiled actor and critic modules")
    except Exception as e:
        print(f"Warning: Failed to compile actor/critic modules: {str(e)}")
    # No need to reassign to agent; DDP wraps core_agent

    # Define the per-episode update step (no compilation to avoid Dynamo conflicts)
    def train_step_fn(agent_model, states_b, actions_b, returns_b, logp_old_b, values_b, optim):
        # zero gradients
        optim.zero_grad()
        # mixed precision forward and loss
        with autocast():
            # Unwrap DDP
            core_model = agent_model.module if hasattr(agent_model, 'module') else agent_model
            action, new_logp, entropy, new_val = core_model.get_action_and_value(states_b, actions_b)
            # surrogate objective
            ratio = torch.exp(new_logp - logp_old_b)
            s1 = ratio * advantages_b
            s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_b
            policy_loss = -torch.min(s1, s2).mean()
            # value clipping
            vpred = new_val.view(-1)
            v_unclipped = (vpred - returns_b).pow(2)
            v_clipped = values_b + torch.clamp(vpred - values_b, -clip_eps, clip_eps)
            v_clipped_loss = (v_clipped - returns_b).pow(2)
            value_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
            # entropy bonus and total loss
            ent_loss = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * ent_loss
        # scale and backward
        scaler.scale(loss).backward()
        # unscale before clipping
        scaler.unscale_(optim)
        # gradient clipping
        core = agent_model.module if hasattr(agent_model, 'module') else agent_model
        torch.nn.utils.clip_grad_norm_(core.parameters(), max_norm=0.5)
        # optimizer step
        scaler.step(optim)
        scaler.update()
        return policy_loss.item(), value_loss.item(), ent_loss.item()

    # Compile the per-episode update step for speed
    try:
        compiled_step = torch.compile(train_step_fn, backend='inductor')
        print("Compiled per-episode train_step_fn for faster updates")
    except Exception:
        compiled_step = train_step_fn  # fallback

    # Compile environment step for speed
    try:
        env.step = torch.compile(env.step, backend='inductor')
        if rank0: print("Compiled env.step for faster transitions")
    except Exception as e:
        if rank0: print(f"Warning: Failed to compile env.step: {str(e)}")

    for ep in ep_iter:
        # Preallocate buffers for rollout to avoid Python lists & torch.stack
        states_buf   = torch.empty(rollout_length, num_envs, grid_size, grid_size, device=device, dtype=env.state.dtype)
        actions_buf  = torch.empty(rollout_length, num_envs, dtype=torch.int64, device=device)
        rewards_buf  = torch.empty(rollout_length, num_envs, device=device)
        dones_buf    = torch.empty(rollout_length, num_envs, device=device)
        values_buf   = torch.empty(rollout_length, num_envs, device=device)
        logp_buf     = torch.empty(rollout_length, num_envs, device=device)
        if use_shaping:
            raw_buf       = torch.empty(rollout_length, num_envs, device=device)
            empty_bon_buf = torch.empty(rollout_length, num_envs, device=device)
            max_bon_buf   = torch.empty(rollout_length, num_envs, device=device)
            waste_pen_buf = torch.empty(rollout_length, num_envs, device=device)
        # collect rollout
        for t in range(rollout_length):
            # Get legal actions (if available in the environment)
            legal_actions = None  # To be implemented if the environment supports it
            
            # Take a step
            with torch.no_grad():
                action, logp, _, val = core_agent.get_action_and_value(state, legal_actions=legal_actions)
            
            next_state, rew, done = env.step(action)
            
            # Mark envs done if they reached the max_tile threshold
            if max_tile is not None:
                # next_state shape: [num_envs, grid, grid]
                max_vals = next_state.view(num_envs, -1).max(dim=1)[0]
                done = done | (max_vals >= max_tile)
            
            # reward shaping
            if use_shaping:
                try:
                    # Use Triton kernel for reward shaping
                    shaped_rew, empty_diff, max_bonus, waste = compute_shaped_rewards(
                        state, next_state, rew
                    )
                except (ImportError, RuntimeError):
                    # Fallback to CPU implementation if Triton not available
                    raw = rew.clone()
                    empties0 = (state == 0).view(num_envs, -1).sum(dim=1).float()
                    empties1 = (next_state == 0).view(num_envs, -1).sum(dim=1).float()
                    empty_diff = 0.01 * (empties1 - empties0)
                    max0 = state.view(num_envs, -1).max(dim=1)[0].float()
                    max1 = next_state.view(num_envs, -1).max(dim=1)[0].float()
                    max_bonus = 1.0 * (max1 > max0).float()
                    changed = (next_state != state).view(num_envs, -1).any(dim=1).float()
                    waste = -0.1 * (1 - changed)
                    shaped_rew = rew + empty_diff + max_bonus + waste
                    # record components into buffers
                    if use_shaping:
                        raw_buf[t]       = raw if 'raw' in locals() else rew
                        empty_bon_buf[t] = empty_diff
                        max_bon_buf[t]   = max_bonus
                        waste_pen_buf[t] = waste
                        rew = shaped_rew

            # append reward
            rewards_buf[t] = rew
            dones_buf[t]   = done.float()
            values_buf[t]  = val
            logp_buf[t]    = logp
            states_buf[t].copy_(state)
            actions_buf[t] = action
            state = next_state
            
            # partial reset: only reinitialize finished envs and preserve others
            if done.any():
                prev_state = state.clone()
                init_state, init_done = env.reset()
                # ensure matching dtypes for assignment
                if init_state.dtype != prev_state.dtype:
                    init_state = init_state.to(prev_state.dtype)
                # reset only those envs that finished
                prev_state[done] = init_state[done]
                state = prev_state
                done = init_done
                # sync internal environment state
                env.state = state.clone()
                env.done = done.clone()

            # Update historical max tile tracking
            curr_max_tile = next_state.max().item()
            historical_max_tile = max(historical_max_tile, curr_max_tile)

        # compute advantages and returns using Triton kernel
        # stack tensors into [T, N]
        rewards_t = rewards_buf
        values_t  = values_buf
        dones_t   = dones_buf
        # get value for final state
        with torch.no_grad():
            next_val = core_agent.get_value(state)
        # Triton-based GAE computation
        advantages_t, returns_t = compute_gae_advantages(
            rewards_t, values_t, dones_t, next_val, gamma, gae_lambda
        )
        # flatten states and actions
        states_b = states_buf.view(-1, grid_size, grid_size)
        actions_b = actions_buf.view(-1)
        # flatten and normalize advantages
        adv_stack = advantages_t.view(-1)
        advantages_b = ((adv_stack - adv_stack.mean()) / (adv_stack.std(unbiased=False) + 1e-8)).detach()
        # flatten returns and detach
        returns_b = returns_t.view(-1).detach()
        # old log prob and values
        logp_old_b = logp_buf.view(-1).detach()
        values_b = values_buf.view(-1).detach()

        # Run training step normally (no CUDA graphs)
        policy_loss_avg, value_loss_avg, entropy_avg = compiled_step(
            agent, states_b, actions_b, returns_b, logp_old_b, values_b, optimizer)
        clipfrac_avg, avg_kl = 0.0, 0.0

        # update tqdm and TensorBoard logs
        if rank0 and ep % log_interval == 0:
            # compute actual rollout return from collected rewards
            stacked_rewards = rewards_buf.sum(dim=0)  # per-env returns
            rollout_returns = stacked_rewards.mean().item()
            # Use current max tile across all envs, but preserve historical best
            current_max_tile = state.max().item()
            # Update logs with both current and historical max
            if writer is not None:
                writer.add_scalar('Performance/current_max_tile', current_max_tile, ep)
                writer.add_scalar('Performance/historical_max_tile', historical_max_tile, ep)
            # tqdm update
            ep_iter.set_postfix({'rollout_return': f"{rollout_returns:.2f}", 'max_tile': historical_max_tile})
            
            # Get current value estimate for logging
            with torch.no_grad():
                val_est = core_agent.get_value(state)
            
            # Log metrics to TensorBoard
            log_training_metrics(
                writer, ep, rollout_returns, historical_max_tile,
                policy_loss_avg, value_loss_avg, entropy_avg,
                clipfrac_avg, avg_kl, agent, optimizer, val_est
            )

        # after all optimization steps for this episode, update scheduler
        # skipped initial scheduler.step() to avoid warning; will step after each episode

        # periodic checkpoint
        if rank0 and ep % checkpoint_interval == 0:
            ckpt_path = f"models/ppo2048_ep{ep}.pth"
            torch.save(agent.state_dict(), ckpt_path)
            if rank0:
                print(f"Saved checkpoint: {ckpt_path}")

    # final save after all episodes
    if not distributed or local_rank == 0:
        torch.save(agent.state_dict(), 'models/ppo2048_final.pth')
        print("Saved final model: models/ppo2048_final.pth")
    # close TensorBoard writer
    if rank0 and writer is not None:
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO for 2048 with optional reward shaping')
    parser.add_argument('--com_reward', action='store_true', help='enable composite reward shaping')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--rollout-length', type=int, default=128, help='rollout length per update')
    parser.add_argument('--global-num-envs', type=int, default=2048, help='total parallel environments')
    parser.add_argument('--log-interval', type=int, default=10, help='episodes between TensorBoard logs')
    parser.add_argument('--max-tile', type=int, default=None, help='Tile value that counts as a win')
    args = parser.parse_args()

    # Set random seeds before compilation
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Set numpy seed if numpy is used
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    # Initialize TensorBoard writer before compilation
    # Only initialize for rank 0 in distributed setting
    writer = None
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not world_size > 1 or local_rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            print("Successfully initialized TensorBoard writer")
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard writer: {str(e)}")
            print("Training will continue without TensorBoard logging")

    # Create and initialize model before compilation
    if torch.cuda.is_available():
        if world_size > 1:
            torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}' if world_size > 1 else 'cuda')
    elif torch.backends.mps.is_available() and not world_size > 1:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Create model
    grid_size = 4
    num_envs = args.global_num_envs // world_size
    agent = PPOAgent(grid_size, num_envs).to(device)
    
    if world_size > 1:
        print("Initializing distributed training...")
        dist.init_process_group(backend='nccl', init_method='env://')
        agent = DDP(agent, device_ids=[local_rank], output_device=local_rank)
        print(f"Process {local_rank} initialized for distributed training")

    # Log model information before training
    if not world_size > 1 or local_rank == 0:
        log_model_info(agent, writer)

    # Invoke (compiled) training
    train(
        agent=agent,
        use_shaping=args.com_reward,
        seed=args.seed,
        rollout_length=args.rollout_length,
        global_num_envs=args.global_num_envs,
        log_interval=args.log_interval,
        writer=writer,
        max_tile=args.max_tile
    ) 