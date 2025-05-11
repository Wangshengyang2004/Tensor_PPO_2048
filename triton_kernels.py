# Core imports and Triton availability check
import torch
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    # Provide dummy Triton stubs so decorators don't break
    class _DummyTriton:
        @staticmethod
        def jit(fn=None, **kwargs):
            if fn is None:
                return lambda x: x
            return fn
        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b
    triton = _DummyTriton()
    class _DummyTL:
        @staticmethod
        def constexpr(x=None, **kwargs):
            return None
    tl = _DummyTL()

# Game Logic Kernels
@triton.jit
def row_move_kernel(
    rows_ptr, lut_rows_ptr, lut_rewards_ptr, out_exps_ptr, out_rews_ptr,
    stride_rows: tl.constexpr, stride_out_exps: tl.constexpr,
    stride_lut_row: tl.constexpr, stride_lut_reward: tl.constexpr,
    TOTAL_ROWS: tl.constexpr, BLOCK_ROWS: tl.constexpr
):
    row_ids = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    mask = row_ids < TOTAL_ROWS
    # compute base offsets
    rbase = row_ids * stride_rows
    # load each exponent component
    ex0 = tl.load(rows_ptr + rbase + 0, mask=mask, other=0)
    ex1 = tl.load(rows_ptr + rbase + 1, mask=mask, other=0)
    ex2 = tl.load(rows_ptr + rbase + 2, mask=mask, other=0)
    ex3 = tl.load(rows_ptr + rbase + 3, mask=mask, other=0)
    # compute LUT index
    idx = (ex0 << 12) | (ex1 << 8) | (ex2 << 4) | ex3
    # lookup new exponents explicitly
    lutbase = idx * stride_lut_row
    new0 = tl.load(lut_rows_ptr + lutbase + 0, mask=mask, other=0)
    new1 = tl.load(lut_rows_ptr + lutbase + 1, mask=mask, other=0)
    new2 = tl.load(lut_rows_ptr + lutbase + 2, mask=mask, other=0)
    new3 = tl.load(lut_rows_ptr + lutbase + 3, mask=mask, other=0)
    # store new exponents
    obase = row_ids * stride_out_exps
    tl.store(out_exps_ptr + obase + 0, new0, mask=mask)
    tl.store(out_exps_ptr + obase + 1, new1, mask=mask)
    tl.store(out_exps_ptr + obase + 2, new2, mask=mask)
    tl.store(out_exps_ptr + obase + 3, new3, mask=mask)
    # lookup and store rewards
    rews = tl.load(lut_rewards_ptr + idx * stride_lut_reward, mask=mask, other=0.0)
    tl.store(out_rews_ptr + row_ids, rews, mask=mask)

# PPO Training Kernels
@triton.jit
def gae_kernel(
    rewards_ptr,      # *Pointer to rewards tensor [T, N]
    values_ptr,       # *Pointer to values tensor [T, N]
    dones_ptr,       # *Pointer to dones tensor [T, N]
    next_value_ptr,  # *Pointer to next_value tensor [N]
    advantages_ptr,   # *Pointer to output advantages tensor [T, N]
    returns_ptr,     # *Pointer to output returns tensor [T, N]
    gamma,           # Discount factor
    gae_lambda,      # GAE lambda parameter
    stride_t,        # Time dimension stride
    stride_n,        # Batch dimension stride
    T,               # Rollout length
    N,               # Batch size
    BLOCK_SIZE: tl.constexpr,  # Number of parallel threads
):
    # Compute linear index and check bounds
    pid = tl.program_id(0)
    n_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = n_idx < N

    # Initialize last GAE value for each environment
    last_gae = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    next_val = tl.load(next_value_ptr + n_idx * stride_n, mask=mask)

    # Iterate backwards through time steps
    for t in range(T-1, -1, -1):
        # Load values for current timestep
        rewards = tl.load(rewards_ptr + t * stride_t + n_idx * stride_n, mask=mask)
        values = tl.load(values_ptr + t * stride_t + n_idx * stride_n, mask=mask)
        dones = tl.load(dones_ptr + t * stride_t + n_idx * stride_n, mask=mask)

        # Compute GAE
        mask_t = 1.0 - dones.to(tl.float32)
        delta = rewards + gamma * next_val * mask_t - values
        last_gae = delta + gamma * gae_lambda * mask_t * last_gae

        # Store advantages and returns
        tl.store(advantages_ptr + t * stride_t + n_idx * stride_n, last_gae, mask=mask)
        tl.store(returns_ptr + t * stride_t + n_idx * stride_n, last_gae + values, mask=mask)

        # Update next value
        next_val = values

@triton.jit
def reward_shaping_kernel(
    state_ptr,        # *Pointer to current state tensor [N, H, W]
    next_state_ptr,   # *Pointer to next state tensor [N, H, W]
    raw_reward_ptr,   # *Pointer to raw reward tensor [N]
    shaped_reward_ptr,# *Pointer to output shaped reward tensor [N]
    empty_bonus_ptr,  # *Pointer to output empty cells bonus tensor [N]
    max_bonus_ptr,    # *Pointer to output max tile bonus tensor [N]
    waste_penalty_ptr,# *Pointer to output waste penalty tensor [N]
    stride_n,         # Batch dimension stride
    stride_h,         # Height dimension stride
    stride_w,         # Width dimension stride
    N,               # Batch size
    H,               # Grid height
    W,               # Grid width
    BLOCK_SIZE: tl.constexpr,  # Number of parallel threads
):
    # Compute linear index and check bounds
    pid = tl.program_id(0)
    n_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = n_idx < N

    # Initialize counters for each environment
    empties0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    empties1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    max0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    max1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    changed = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Count empty cells and find max tiles
    for h in range(H):
        for w in range(W):
            # Load current and next state values
            s0 = tl.load(state_ptr + n_idx * stride_n + h * stride_h + w * stride_w, mask=mask)
            s1 = tl.load(next_state_ptr + n_idx * stride_n + h * stride_h + w * stride_w, mask=mask)
            
            # Count empty cells
            empties0 += (s0 == 0).to(tl.float32)
            empties1 += (s1 == 0).to(tl.float32)
            
            # Track max tiles
            max0 = tl.maximum(max0, s0)
            max1 = tl.maximum(max1, s1)
            
            # Track if state changed
            changed = tl.maximum(changed, (s0 != s1).to(tl.float32))

    # Compute reward components
    empty_bonus = 0.01 * (empties1 - empties0)
    max_bonus = 1.0 * (max1 > max0).to(tl.float32)
    waste_penalty = -0.1 * (1.0 - changed)

    # Load raw reward
    raw_reward = tl.load(raw_reward_ptr + n_idx, mask=mask)

    # Compute final shaped reward
    shaped_reward = raw_reward + empty_bonus + max_bonus + waste_penalty

    # Store all components
    tl.store(shaped_reward_ptr + n_idx, shaped_reward, mask=mask)
    tl.store(empty_bonus_ptr + n_idx, empty_bonus, mask=mask)
    tl.store(max_bonus_ptr + n_idx, max_bonus, mask=mask)
    tl.store(waste_penalty_ptr + n_idx, waste_penalty, mask=mask)

# Wrapper functions
def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns using Triton kernel.
    
    Args:
        rewards: Tensor of shape [T, N] containing rewards
        values: Tensor of shape [T, N] containing value estimates
        dones: Tensor of shape [T, N] containing done flags
        next_value: Tensor of shape [N] containing value estimate for final state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: Tensor of shape [T, N] containing GAE advantages
        returns: Tensor of shape [T, N] containing returns
    """
    device = rewards.device
    T, N = rewards.shape
    
    # Create output tensors
    advantages = torch.empty_like(rewards)
    returns = torch.empty_like(rewards)
    
    # Launch kernel
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gae_kernel[grid](
        rewards.contiguous(),
        values.contiguous(),
        dones.contiguous(),
        next_value.contiguous(),
        advantages,
        returns,
        gamma,
        gae_lambda,
        rewards.stride(0),
        rewards.stride(1),
        T,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return advantages, returns

def compute_shaped_rewards(
    state: torch.Tensor,
    next_state: torch.Tensor,
    raw_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute shaped rewards using Triton kernel.
    
    Args:
        state: Tensor of shape [N, H, W] containing current states
        next_state: Tensor of shape [N, H, W] containing next states
        raw_rewards: Tensor of shape [N] containing raw rewards
    
    Returns:
        shaped_rewards: Tensor of shape [N] containing shaped rewards
        empty_bonus: Tensor of shape [N] containing empty cells bonus
        max_bonus: Tensor of shape [N] containing max tile bonus
        waste_penalty: Tensor of shape [N] containing waste penalty
    """
    device = state.device
    N, H, W = state.shape
    
    # Create output tensors
    shaped_rewards = torch.empty_like(raw_rewards)
    empty_bonus = torch.empty_like(raw_rewards)
    max_bonus = torch.empty_like(raw_rewards)
    waste_penalty = torch.empty_like(raw_rewards)
    
    # Launch kernel
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    reward_shaping_kernel[grid](
        state.contiguous(),
        next_state.contiguous(),
        raw_rewards.contiguous(),
        shaped_rewards,
        empty_bonus,
        max_bonus,
        waste_penalty,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        N,
        H,
        W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return shaped_rewards, empty_bonus, max_bonus, waste_penalty

def compute_row_move(
    rows_exps: torch.Tensor,
    lut_rows: torch.Tensor,
    lut_rewards: torch.Tensor,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute row moves using Triton kernel.
    
    Args:
        rows_exps: Tensor of shape [num_rows, 4] containing row exponents
        lut_rows: Tensor of shape [_ROW_COUNT, 4] containing LUT for row moves
        lut_rewards: Tensor of shape [_ROW_COUNT] containing LUT for rewards
        num_rows: Number of rows to process
    
    Returns:
        out_exps: Tensor of shape [num_rows, 4] containing output exponents
        row_rewards: Tensor of shape [num_rows] containing rewards
    """
    # Prepare output buffers
    out_exps = torch.empty_like(rows_exps)
    row_rewards = torch.empty((num_rows,), device=rows_exps.device)
    
    # Launch kernel
    BLOCK = 128
    grid = ((num_rows + BLOCK - 1) // BLOCK,)
    
    row_move_kernel[grid](
        rows_exps, lut_rows, lut_rewards,
        out_exps, row_rewards,
        rows_exps.stride(0), out_exps.stride(0),
        lut_rows.stride(0), lut_rewards.stride(0),
        num_rows, BLOCK
    )
    
    return out_exps, row_rewards

# CPU fallback implementations if Triton not available
if not _HAS_TRITON:
    def compute_gae_advantages(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Naive CPU GAE fallback"""
        T, N = rewards.shape
        advantages = torch.empty_like(rewards)
        returns = torch.empty_like(rewards)
        last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
        for t in range(T-1, -1, -1):
            mask_t = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask_t - values[t]
            last_gae = delta + gamma * gae_lambda * mask_t * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + values[t]
            next_value = values[t]
        return advantages, returns

    def compute_shaped_rewards(
        state: torch.Tensor,
        next_state: torch.Tensor,
        raw_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Naive CPU reward shaping fallback"""
        N = raw_rewards.size(0)
        empties0 = (state == 0).view(N, -1).sum(dim=1).float()
        empties1 = (next_state == 0).view(N, -1).sum(dim=1).float()
        empty_bonus = 0.01 * (empties1 - empties0)
        max0 = state.view(N, -1).max(dim=1)[0].float()
        max1 = next_state.view(N, -1).max(dim=1)[0].float()
        max_bonus = (max1 > max0).float()
        changed = (next_state != state).view(N, -1).any(dim=1).float()
        waste_penalty = -0.1 * (1 - changed)
        shaped = raw_rewards + empty_bonus + max_bonus + waste_penalty
        return shaped, empty_bonus, max_bonus, waste_penalty

    def compute_row_move(
        rows_exps: torch.Tensor,
        lut_rows: torch.Tensor,
        lut_rewards: torch.Tensor,
        num_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Naive CPU row-move fallback"""
        idx = ((rows_exps[:,0] << 12) |
               (rows_exps[:,1] << 8) |
               (rows_exps[:,2] << 4) |
               rows_exps[:,3]).to(torch.int64)
        out_exps = lut_rows[idx]
        row_rewards = lut_rewards[idx]
        return out_exps, row_rewards 