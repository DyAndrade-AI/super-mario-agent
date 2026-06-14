"""
Replay memory for off-policy Q-learning.
"""
from typing import Dict, Tuple

import numpy as np
import torch


def _as_uint8_observation(obs: np.ndarray) -> np.ndarray:
    obs_array = np.asarray(obs)
    if obs_array.dtype == np.uint8:
        return obs_array

    if np.issubdtype(obs_array.dtype, np.floating):
        obs_array = obs_array * 255.0 if obs_array.max(initial=0.0) <= 1.0 else obs_array

    return np.clip(obs_array, 0, 255).astype(np.uint8)


def _uint8_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32).div_(255.0)
    if tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor


class PrioritizedReplayBuffer:
    """
    Proportional prioritized replay buffer.

    Observations are stored as uint8 to keep Atari-style frame stacks practical.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 1_000_000,
        priority_eps: float = 1e-6,
    ):
        if capacity <= 0:
            raise ValueError("capacity debe ser mayor que cero")

        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.device = device
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.priority_eps = float(priority_eps)

        self.observations = np.empty((self.capacity, *self.obs_shape), dtype=np.uint8)
        self.next_observations = np.empty((self.capacity, *self.obs_shape), dtype=np.uint8)
        self.actions = np.empty((self.capacity,), dtype=np.int64)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.discounts = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.bool_)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        discount: float,
    ):
        self.observations[self.pos] = _as_uint8_observation(obs)
        self.next_observations[self.pos] = _as_uint8_observation(next_obs)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.discounts[self.pos] = float(discount)
        self.dones[self.pos] = bool(done)
        self.priorities[self.pos] = self.max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def beta_by_frame(self, frame_idx: int) -> float:
        if self.beta_frames <= 0:
            return 1.0
        progress = min(max(frame_idx, 0) / self.beta_frames, 1.0)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def sample(self, batch_size: int, beta: float) -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("No se puede muestrear de un replay buffer vacio")

        priorities = self.priorities[: self.size]
        if self.alpha > 0.0:
            scaled_priorities = np.power(np.maximum(priorities, self.priority_eps), self.alpha)
            probabilities = scaled_priorities / scaled_priorities.sum()
        else:
            probabilities = np.full((self.size,), 1.0 / self.size, dtype=np.float32)

        indices = np.random.choice(self.size, size=batch_size, replace=True, p=probabilities)
        sample_probabilities = probabilities[indices]
        weights = np.power(self.size * sample_probabilities, -beta)
        weights /= weights.max(initial=1.0)

        return {
            "obs": _uint8_to_tensor(self.observations[indices], self.device),
            "actions": torch.from_numpy(self.actions[indices]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(self.device),
            "next_obs": _uint8_to_tensor(self.next_observations[indices], self.device),
            "dones": torch.from_numpy(self.dones[indices].astype(np.float32)).to(self.device),
            "discounts": torch.from_numpy(self.discounts[indices]).to(self.device),
            "weights": torch.from_numpy(weights.astype(np.float32)).to(self.device),
            "indices": indices,
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.maximum(priorities, self.priority_eps)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max(initial=self.max_priority)))

    def __len__(self) -> int:
        return self.size


def create_replay_buffer(
    capacity: int,
    obs_shape: Tuple[int, ...],
    device: torch.device,
    **kwargs,
) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(capacity=capacity, obs_shape=obs_shape, device=device, **kwargs)
