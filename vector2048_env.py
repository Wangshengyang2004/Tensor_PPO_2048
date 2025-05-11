# vector2048_env.py
import torch
import math
import platform
# TorchScript detection
# from torch.jit import is_scripting

# detect if Triton is available on Linux
try:
    import triton
    import triton.language as tl
    from triton_kernels import compute_row_move
    # Triton only makes sense on CUDA devices
    _HAS_TRITON = platform.system() == 'Linux'  # Will check device type during __init__
except ImportError:
    # no Triton: define dummy triton.jit and dummy tl with constexpr for annotations
    _HAS_TRITON = False
    class _DummyTriton:
        @staticmethod
        def jit(fn):
            return fn
    triton = _DummyTriton()
    class _DummyTL:
        # constexpr used in annotations; provide dummy attribute
        @staticmethod
        def constexpr(x=None, **kwargs):
            return None
    tl = _DummyTL

# Precompute LUT for row moves: 4 cells, 4-bit exponents, total 16-bit index
_ROW_COUNT = 1 << 16
_LUT_ROW = torch.zeros(_ROW_COUNT, 4, dtype=torch.int64)
_LUT_REWARD = torch.zeros(_ROW_COUNT, dtype=torch.float32)
for idx in range(_ROW_COUNT):
    # decode exponents
    e0 = (idx >> 12) & 0xF
    e1 = (idx >> 8) & 0xF
    e2 = (idx >> 4) & 0xF
    e3 = idx & 0xF
    tiles = [0 if e == 0 else (1 << e) for e in (e0, e1, e2, e3)]
    # compress and merge
    merged = []
    reward = 0
    i = 0
    while i < 4:
        if i < 3 and tiles[i] == tiles[i + 1] and tiles[i] != 0:
            mv = tiles[i] * 2
            reward += mv
            merged.append(mv)
            i += 2
        else:
            if tiles[i] != 0:
                merged.append(tiles[i])
            i += 1
    # pad to length 4
    while len(merged) < 4:
        merged.append(0)
    # to exponents
    new_exps = [0 if v == 0 else int(math.log2(v)) for v in merged]
    _LUT_ROW[idx] = torch.tensor(new_exps, dtype=torch.int64)
    _LUT_REWARD[idx] = reward

class Vector2048Env:
    """
    A vectorized 2048 environment supporting batch operations in PyTorch.
    State is represented as an integer tensor of shape (num_envs, grid_size, grid_size).
    """
    def __init__(self, num_envs, grid_size=4, device='cpu'):
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.device = device
        # whether to use Triton kernel (only on CUDA devices)
        self.use_triton = _HAS_TRITON and (str(device).startswith('cuda') or 
                                           (hasattr(device, 'type') and device.type == 'cuda'))
        # bind move function: Triton vs Python LUT
        if self.use_triton:
            self._move = self._move_triton
        else:
            self._move = self._move_lut
        # move LUTs to device
        self._LUT_ROW = _LUT_ROW.to(device)
        self._LUT_REWARD = _LUT_REWARD.to(device)
        self.reset()

    def reset(self):
        """
        Resets all environments by zeroing states and spawning two initial tiles.
        Returns:
            state: LongTensor of shape (num_envs, grid_size, grid_size)
            done: BoolTensor of shape (num_envs,)
        """
        # Set a random seed unique to this device to ensure reproducibility
        # while not causing CPU/GPU to share the exact same sequence (which could mask bugs)
        seed = hash(str(self.device)) % 2**32
        torch.manual_seed(seed)
        
        self.state = torch.zeros(self.num_envs, self.grid_size, self.grid_size,
                                 device=self.device, dtype=torch.int64)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.score = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._add_random_tiles(count=2)
        return self.state.clone(), self.done.clone()

    def step(self, actions):
        """
        Steps the batch of environments with the provided actions.
        Args:
            actions: LongTensor of shape (num_envs,), values in {0,1,2,3} for up,right,down,left.
        Returns:
            next_state: LongTensor (num_envs, grid_size, grid_size)
            reward: FloatTensor (num_envs,)
            done: BoolTensor (num_envs,)
        """
        prev = self.state.clone()
        # vectorized moves for all actions
        moved0, rew0 = self._move(self.state, 0)
        moved1, rew1 = self._move(self.state, 1)
        moved2, rew2 = self._move(self.state, 2)
        moved3, rew3 = self._move(self.state, 3)
        moved_stack = torch.stack([moved0, moved1, moved2, moved3], dim=1)
        rew_stack = torch.stack([rew0, rew1, rew2, rew3], dim=1)
        idx = torch.arange(self.num_envs, device=self.device)
        actions = actions.to(self.device)
        # select next state and rewards according to action
        self.state = moved_stack[idx, actions]
        rewards = rew_stack[idx, actions]
        # Add new tiles where the board changed
        changed = (self.state != prev).view(self.num_envs, -1).any(dim=1)
        if changed.any():
            self._add_random_tiles(count=1)
        # Update cumulative score
        self.score += rewards
        # Compute done flags
        done = self._check_done(self.state)
        self.done = done
        return self.state.clone(), rewards.clone(), done.clone()

    # def _move(self, states, action):
    #     """
    #     Moves the batch of boards according to the action (0=up,1=right,2=down,3=left).
    #     Uses rotate/flip + table lookup (LUT) for left logic.
    #     Returns:
    #         new_states: Tensor like states
    #         rewards: FloatTensor of shape (batch,)
    #     """
    #     # rotate/flip into left-move orientation
    #     if action == 0:   # up
    #         s = states.permute(0,2,1)
    #     elif action == 2: # down
    #         s = states.permute(0,2,1).flip(dims=[2])
    #     elif action == 1: # right
    #         s = states.flip(dims=[2])
    #     else:
    #         s = states
    #     batch = s.shape[0]
    #     # flatten rows: (batch*grid_size, 4)
    #     rows = s.reshape(batch * self.grid_size, self.grid_size)
    #     # extract exponents (0 for empty, else log2)
    #     exps = torch.where(rows == 0,
    #                        torch.zeros_like(rows, dtype=torch.int64),
    #                        torch.log2(rows).to(torch.int64))
    #     # compute LUT index
    #     idx = ((exps[:,0] << 12) |
    #            (exps[:,1] << 8)  |
    #            (exps[:,2] << 4)  |
    #            exps[:,3]).to(torch.int64)
    #     # lookup new exponents and rewards
    #     new_exps = self._LUT_ROW[idx]               # (batch*grid_size,4)
    #     row_rewards = self._LUT_REWARD[idx]        # (batch*grid_size,)
    #     # reconstruct tile values: 2**exp, 0 if exp==0
    #     out_vals = torch.where(new_exps > 0,
    #                             1 << new_exps,
    #                             torch.zeros_like(new_exps))
    #     # reshape to board
    #     out = out_vals.reshape(batch, self.grid_size, self.grid_size)
    #     # reshape rewards per-row and sum per-board
    #     rewards = row_rewards.reshape(batch, self.grid_size).sum(dim=1)
    #     # rotate/flip back
    #     if action == 0:
    #         out = out.permute(0,2,1)
    #     elif action == 2:
    #         out = out.flip(dims=[2]).permute(0,2,1)
    #     elif action == 1:
    #         out = out.flip(dims=[2])
    #     return out, rewards

    def _add_random_tiles(self, count=1):
        """
        Adds 'count' random tiles (2 or 4) to each board where possible.
        Uses a max-random trick to sample uniform empty cells without Python loops over envs.
        """
        for _ in range(count):
            flat = self.state.view(self.num_envs, -1)
            # random scores, masked so filled cells get -1
            rnd = torch.rand(self.num_envs, flat.size(1), device=self.device)
            mask = (flat == 0)
            scores = rnd * mask.to(rnd.dtype) + (~mask).to(rnd.dtype) * -1
            pos = scores.argmax(dim=1)
            row = pos // self.grid_size
            col = pos % self.grid_size
            # generate new tiles matching state dtype
            dtype = self.state.dtype
            tiles = torch.where(
                torch.rand(self.num_envs, device=self.device) < 0.9,
                torch.full((self.num_envs,), 2, device=self.device, dtype=dtype),
                torch.full((self.num_envs,), 4, device=self.device, dtype=dtype)
            )
            self.state[torch.arange(self.num_envs), row, col] = tiles

    def _check_done(self, states):
        """
        Checks if each board is in a terminal (no moves left) state.
        """
        # any empty cell
        empty = (states == 0).view(self.num_envs, -1).any(dim=1)
        # any horizontal merge possible
        h = (states[:, :, :-1] == states[:, :, 1:]).view(self.num_envs, -1).any(dim=1)
        # any vertical merge possible
        v = (states[:, :-1, :] == states[:, 1:, :]).view(self.num_envs, -1).any(dim=1)
        # done if no empty and no merges
        return ~(empty | h | v)

    # alias LUT move for fallback
    # _move_lut = _move

    def _move_triton(self, states, action):
        """
        Triton-based row merge via LUT lookup in GPU.
        """
        # rotate/flip into left-move orientation
        if action == 0:
            s = states.permute(0,2,1)
        elif action == 2:
            s = states.permute(0,2,1).flip(dims=[2])
        elif action == 1:
            s = states.flip(dims=[2])
        else:
            s = states
        batch = s.shape[0]
        # compute exponents
        exps = torch.where(s == 0,
                          torch.zeros_like(s, dtype=torch.int32),
                          (s.float().log2()).to(torch.int32))
        # reshape to rows
        rows_exps = exps.reshape(batch * self.grid_size, self.grid_size).contiguous()
        num_rows = rows_exps.shape[0]
        
        # Use the compute_row_move function from triton_kernels
        out_exps, row_rewards = compute_row_move(
            rows_exps, self._LUT_ROW, self._LUT_REWARD, num_rows
        )
        
        # reconstruct tile values
        out_vals = torch.where(out_exps > 0,
                               1 << out_exps,
                               torch.zeros_like(out_exps))
        out = out_vals.reshape(batch, self.grid_size, self.grid_size)
        rewards = row_rewards.reshape(batch, self.grid_size).sum(dim=1)
        # rotate/flip back
        if action == 0:
            out = out.permute(0,2,1)
        elif action == 2:
            out = out.flip(dims=[2]).permute(0,2,1)
        elif action == 1:
            out = out.flip(dims=[2])
        # Ensure consistent dtypes with the LUT version
        return out.to(self.state.dtype), rewards.to(torch.float32)

    def _move_lut(self, states, action):
        """
        LUT-based row merge via LUT lookup in CPU.
        """
        # rotate/flip into left-move orientation
        if action == 0:   # up
            s = states.permute(0,2,1)
        elif action == 2: # down
            s = states.permute(0,2,1).flip(dims=[2])
        elif action == 1: # right
            s = states.flip(dims=[2])
        else:
            s = states
        batch = s.shape[0]
        # flatten rows: (batch*grid_size, 4)
        rows = s.reshape(batch * self.grid_size, self.grid_size)
        # extract exponents (0 for empty, else log2)
        exps = torch.where(rows == 0,
                           torch.zeros_like(rows, dtype=torch.int64),
                           torch.log2(rows).to(torch.int64))
        # compute LUT index
        idx = ((exps[:,0] << 12) |
               (exps[:,1] << 8)  |
               (exps[:,2] << 4)  |
               exps[:,3]).to(torch.int64)
        # lookup new exponents and rewards
        new_exps = self._LUT_ROW[idx]               # (batch*grid_size,4)
        row_rewards = self._LUT_REWARD[idx]        # (batch*grid_size,)
        # reconstruct tile values: 2**exp, 0 if exp==0
        out_vals = torch.where(new_exps > 0,
                                1 << new_exps,
                                torch.zeros_like(new_exps))
        # reshape to board
        out = out_vals.reshape(batch, self.grid_size, self.grid_size)
        # reshape rewards per-row and sum per-board
        rewards = row_rewards.reshape(batch, self.grid_size).sum(dim=1)
        # rotate/flip back
        if action == 0:
            out = out.permute(0,2,1)
        elif action == 2:
            out = out.flip(dims=[2]).permute(0,2,1)
        elif action == 1:
            out = out.flip(dims=[2])
        return out, rewards

    def _move(self, states, action):
        if not self.use_triton:
            out, rewards = self._move_lut(states, action)
        else:
            out, rewards = self._move_triton(states, action)
        return out.to(self.state.dtype), rewards 