"""
Buffer integration utilities for Gumbel MCTS with MiniSTU World Model.

Provides methods to integrate ground truth and imaginary buffers into MCTS planning.
"""

import torch


def prepare_world_model_input(buffer_managers, batch_size, device='cuda'):
    """
    Prepare state sequences from imaginary buffers for world model inference during MCTS.
    
    Args:
        buffer_managers: List of DualBufferManager instances (one per environment)
        batch_size: Number of environments
        device: Device to move tensors to
    
    Returns:
        state_sequences: Batch of state sequences (batch_size, seq_len, channels, H, W)
                        or None if buffers not ready
    """
    if buffer_managers is None or len(buffer_managers) == 0:
        return None
    
    state_sequences = []
    
    for i in range(min(batch_size, len(buffer_managers))):
        try:
            # Get current imagined state sequence from buffer
            state_seq = buffer_managers[i].get_imagined_sequence()
            state_sequences.append(state_seq)
        except RuntimeError:
            # Buffer not yet filled, return None
            return None
    
    if len(state_sequences) < batch_size:
        return None
    
    # Stack all state sequences
    batch_state_sequences = torch.stack(state_sequences).to(device)
    return batch_state_sequences


def get_buffer_state_for_env(buffer_managers, env_idx):
    """
    Get the current imagined state sequence for a specific environment.
    
    Args:
        buffer_managers: List of DualBufferManager instances
        env_idx: Index of environment
    
    Returns:
        state_sequence: (seq_len, channels, H, W) or None if buffer not ready
    """
    if buffer_managers is None or env_idx >= len(buffer_managers):
        return None
    
    try:
        return buffer_managers[env_idx].get_imagined_sequence()
    except RuntimeError:
        return None


def update_all_buffers_with_state(buffer_managers, predicted_states):
    """
    Update imaginary buffers with newly predicted states from world model.
    
    Args:
        buffer_managers: List of DualBufferManager instances
        predicted_states: Batch of predicted states (batch_size, channels, H, W)
    """
    if buffer_managers is None:
        return
    
    for i, manager in enumerate(buffer_managers):
        if i < len(predicted_states):
            try:
                manager.imagine_step(predicted_states[i].detach())
            except RuntimeError:
                pass


def reset_all_imaginary_buffers(buffer_managers, model):
    """
    Reset all imaginary buffers with current ground truth states.
    
    Args:
        buffer_managers: List of DualBufferManager instances
        model: Model with representation_model attribute
    """
    if buffer_managers is None:
        return
    
    for manager in buffer_managers:
        try:
            gt_states = manager.get_gt_encoded_states(model.representation_model)
            manager.reset_imagination(gt_states)
        except RuntimeError:
            pass


def initialize_imaginary_buffers(buffer_managers, model):
    """
    Initialize imaginary buffers at the start of planning.
    
    Args:
        buffer_managers: List of DualBufferManager instances
        model: Model with representation_model attribute
    """
    if buffer_managers is None:
        return
    
    for manager in buffer_managers:
        try:
            gt_states = manager.get_gt_encoded_states(model.representation_model)
            manager.start_planning(gt_states)
        except RuntimeError:
            pass
