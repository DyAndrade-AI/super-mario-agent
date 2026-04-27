"""
Memory buffer para recolección de experiencias en PPO
Almacena observaciones, acciones, recompensas y otros datos
"""
import numpy as np
import torch
from typing import Tuple, Optional, List
from collections import deque


class RolloutBuffer:
    """
    Buffer para almacenar rollouts (experiencias consecutivas)
    Optimizado para PPO con Generalized Advantage Estimation (GAE)
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        action_shape: Tuple = (),
        num_envs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = None,
    ):
        """
        Inicializa el buffer de rollout

        Args:
            buffer_size: Tamaño del buffer
            obs_shape: Shape de observaciones
            action_shape: Shape de acciones
            num_envs: Número de entornos paralelos
            gamma: Factor de descuento
            gae_lambda: Parámetro lambda para GAE
            device: Dispositivo (CPU/GPU)
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cpu")
        self.pos = 0
        self.full = False

        # Buffers
        self.observations = np.zeros((buffer_size, num_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs) + action_shape, dtype=np.int64)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)

        # Para LSTM: guardar hidden states
        self.lstm_hiddens = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        done: np.ndarray,
        lstm_hidden: Optional[Tuple] = None,
    ):
        """
        Añade una experiencia al buffer

        Args:
            obs: Observación
            action: Acción
            reward: Recompensa
            value: Value estimado
            log_prob: Log probability de la acción
            done: Señal de término de episodio
            lstm_hidden: Hidden state del LSTM (opcional)
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done

        if lstm_hidden is not None:
            if self.lstm_hiddens is None:
                # Inicializar lista de hidden states
                self.lstm_hiddens = []
            self.lstm_hiddens.append(lstm_hidden)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_advantages_and_returns(
        self,
        last_values: np.ndarray,
        normalize_advantages: bool = True,
    ):
        """
        Calcula ventajas y retornos usando Generalized Advantage Estimation

        Args:
            last_values: Values de los últimos estados
            normalize_advantages: Normalizar las ventajas
        """
        # Inicializar gae y next_value
        gae = np.zeros((self.num_envs,), dtype=np.float32)
        next_value = last_values

        # Iterar hacia atrás
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_nonterminal = 1.0 - self.dones[t]
                next_value = last_values
            else:
                next_nonterminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae

            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

        # Normalizar ventajas
        if normalize_advantages:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batch(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Tuple:
        """
        Obtiene un batch de experiencias

        Args:
            batch_size: Tamaño del batch
            shuffle: Mezclar datos

        Returns:
            Tupla con tensores de batch
        """
        # Preparar índices
        indices = np.arange(self.buffer_size * self.num_envs)

        # Reshape para flat indexing
        obs = self.observations.reshape(-1, *self.obs_shape)
        actions = self.actions.reshape(-1, *self.action_shape)
        rewards = self.rewards.reshape(-1)
        values = self.values.reshape(-1)
        log_probs = self.log_probs.reshape(-1)
        dones = self.dones.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)

        if shuffle:
            np.random.shuffle(indices)

        # Crear batches
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            # Convertir a tensores
            batch_obs = torch.from_numpy(obs[batch_indices]).to(self.device)
            batch_actions = torch.from_numpy(actions[batch_indices]).to(self.device)
            batch_advantages = torch.from_numpy(advantages[batch_indices]).to(self.device)
            batch_returns = torch.from_numpy(returns[batch_indices]).to(self.device)
            batch_values = torch.from_numpy(values[batch_indices]).to(self.device)
            batch_log_probs = torch.from_numpy(log_probs[batch_indices]).to(self.device)

            yield batch_obs, batch_actions, batch_advantages, batch_returns, batch_values, batch_log_probs

    def reset(self):
        """Resetea el buffer"""
        self.pos = 0
        self.full = False
        self.observations.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.values.fill(0)
        self.log_probs.fill(0)
        self.dones.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)
        self.lstm_hiddens = None

    def is_full(self) -> bool:
        """Verifica si el buffer está lleno"""
        return self.pos == 0 and self.full


class ExperienceBuffer:
    """Buffer simple para almacenar experiencias sin computar GAE"""

    def __init__(
        self,
        max_size: int = 10000,
        obs_shape: Tuple = (4, 84, 84),
        device: torch.device = None,
    ):
        """
        Inicializa el buffer de experiencias

        Args:
            max_size: Tamaño máximo
            obs_shape: Shape de observaciones
            device: Dispositivo
        """
        self.max_size = max_size
        self.obs_shape = obs_shape
        self.device = device or torch.device("cpu")

        self.buffer = deque(maxlen=max_size)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Añade una experiencia"""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple:
        """Muestrea un batch aleatorio"""
        indices = np.random.randint(0, len(self.buffer), size=batch_size)

        experiences = [self.buffer[i] for i in indices]

        obs_list = [exp[0] for exp in experiences]
        actions_list = [exp[1] for exp in experiences]
        rewards_list = [exp[2] for exp in experiences]
        next_obs_list = [exp[3] for exp in experiences]
        dones_list = [exp[4] for exp in experiences]

        obs = torch.from_numpy(np.stack(obs_list)).to(self.device)
        actions = torch.from_numpy(np.array(actions_list)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards_list)).to(self.device)
        next_obs = torch.from_numpy(np.stack(next_obs_list)).to(self.device)
        dones = torch.from_numpy(np.array(dones_list)).to(self.device)

        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        """Retorna el tamaño del buffer"""
        return len(self.buffer)

    def reset(self):
        """Limpia el buffer"""
        self.buffer.clear()


def create_rollout_buffer(
    buffer_size: int,
    obs_shape: Tuple,
    num_envs: int = 1,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: torch.device = None,
) -> RolloutBuffer:
    """
    Factory function para crear un rollout buffer

    Args:
        buffer_size: Tamaño del buffer
        obs_shape: Shape de observaciones
        num_envs: Número de entornos
        gamma: Factor de descuento
        gae_lambda: Parámetro lambda para GAE
        device: Dispositivo

    Returns:
        RolloutBuffer
    """
    return RolloutBuffer(
        buffer_size=buffer_size,
        obs_shape=obs_shape,
        num_envs=num_envs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device,
    )
