"""
Latent State/Action History Buffer for SpectralDynamicsNetwork.

Maintains sliding windows of latent states and raw actions of length seq_len.
Used in three contexts:
  1. Training unroll loop: accumulate ground-truth states/actions, predict next state
  2. MCTS planning: start from real buffer, branch imagined trajectories
  3. Self-play data collection: persist real states/actions across time steps

SpectralDynamicsNetwork.forward expects:
    states:  (B, L, C, H, W)
    actions: (B, L, 1) for discrete  /  (B, L, action_dim) for continuous
and returns:
    next_state: (B, C, H, W)
"""

import torch
import copy


class LatentStateActionBuffer:
    """
    Sliding window buffer of (latent_state, action) pairs.

    state_buffer:  (seq_len, C, H, W)   -- latent states from representation net
    action_buffer: (seq_len, action_dim) -- raw actions (1 for discrete)
    """

    def __init__(self, seq_len, state_shape, action_dim=1, device='cuda'):
        """
        Args:
            seq_len: int, length of the sliding window (L)
            state_shape: tuple (C, H, W) shape of each latent state
            action_dim: int, 1 for discrete actions, action_space_size for continuous
            device: 'cuda' or 'cpu'
        """
        self.seq_len = seq_len
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device

        # Initialise with zeros
        self.state_buffer = torch.zeros(seq_len, *state_shape, device=device)
        self.action_buffer = torch.zeros(seq_len, action_dim, device=device)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def push(self, state, action):
        """
        Push a new (state, action) pair, sliding the window left by one.

        Args:
            state:  (C, H, W) tensor
            action: (action_dim,) tensor or scalar
        """
        state = state.detach().to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device).float()
        action = action.detach().to(self.device).float()
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.shape[0] != self.action_dim:
            # Discrete action stored as scalar in dim-1 tensor
            action = action[:self.action_dim]

        # Shift left by 1 and append
        self.state_buffer = torch.cat([self.state_buffer[1:], state.unsqueeze(0)], dim=0)
        self.action_buffer = torch.cat([self.action_buffer[1:], action.unsqueeze(0)], dim=0)

    def push_state_only(self, state):
        """
        Push a new latent state without an action (for the very last observed state
        before we know which action will be taken).
        This shifts the state buffer but leaves action buffer unchanged.
        Useful at the boundary: we observe a new state but haven't decided the action yet.
        """
        state = state.detach().to(self.device)
        self.state_buffer = torch.cat([self.state_buffer[1:], state.unsqueeze(0)], dim=0)
        # Also shift action buffer to keep alignment, fill last with zero
        self.action_buffer = torch.cat(
            [self.action_buffer[1:], torch.zeros(1, self.action_dim, device=self.device)],
            dim=0,
        )

    def get_state_sequence(self):
        """Return (seq_len, C, H, W) detached."""
        return self.state_buffer.clone().detach()

    def get_action_sequence(self):
        """Return (seq_len, action_dim) detached."""
        return self.action_buffer.clone().detach()

    def reset(self):
        """Zero out both buffers."""
        self.state_buffer.zero_()
        self.action_buffer.zero_()

    def clone(self):
        """Deep copy for branching during MCTS."""
        new = LatentStateActionBuffer(
            self.seq_len, self.state_shape, self.action_dim, self.device
        )
        new.state_buffer = self.state_buffer.clone()
        new.action_buffer = self.action_buffer.clone()
        return new

    def to(self, device):
        self.device = device
        self.state_buffer = self.state_buffer.to(device)
        self.action_buffer = self.action_buffer.to(device)
        return self


class BatchLatentBuffer:
    """
    Batched version of LatentStateActionBuffer for training.

    state_buffer:  (B, seq_len, C, H, W)
    action_buffer: (B, seq_len, action_dim)
    """

    def __init__(self, batch_size, seq_len, state_shape, action_dim=1, device='cuda'):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device

        self.state_buffer = torch.zeros(batch_size, seq_len, *state_shape, device=device)
        self.action_buffer = torch.zeros(batch_size, seq_len, action_dim, device=device)

    def push(self, states, actions):
        """
        Push a batch of (state, action) pairs.

        Args:
            states:  (B, C, H, W) tensor
            actions: (B, action_dim) or (B, 1) tensor
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        # Detach to avoid graph explosion during unroll
        states = states.detach()
        actions = actions.detach().float()

        self.state_buffer = torch.cat([self.state_buffer[:, 1:], states.unsqueeze(1)], dim=1)
        self.action_buffer = torch.cat([self.action_buffer[:, 1:], actions.unsqueeze(1)], dim=1)

    def get_state_sequence(self):
        """Return (B, seq_len, C, H, W)."""
        return self.state_buffer.clone()

    def get_action_sequence(self):
        """Return (B, seq_len, action_dim)."""
        return self.action_buffer.clone()

    def reset(self):
        self.state_buffer.zero_()
        self.action_buffer.zero_()
