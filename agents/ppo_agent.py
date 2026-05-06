"""
Agente PPO (Proximal Policy Optimization) completo
Implementación del algoritmo de entrenamiento
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path

from models.actor_critic import create_actor_critic
from agents.memory import create_rollout_buffer
from utils.metrics import RewardNormalizer


class PPOAgent:
    """
    Agente PPO con todas las características avanzadas
    """

    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.1,
        clip_range_vf: float = 0.1,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        batch_size: int = 128,
        num_envs: int = 16,
        rollout_steps: int = 512,
        epochs_per_update: int = 3,
        target_kl: Optional[float] = None,
        normalize_advantage: bool = True,
        normalize_reward: bool = True,
        clip_grad_norm: float = 0.5,
        use_mixed_precision: bool = True,
        device: torch.device = None,
        **actor_critic_kwargs
    ):
        """
        Inicializa el agente PPO

        Args:
            observation_space: Espacio de observación
            action_space: Espacio de acciones
            learning_rate: Learning rate
            gamma: Factor de descuento
            gae_lambda: Parámetro lambda para GAE
            clip_range: Rango de clipping
            clip_range_vf: Rango de clipping para value function
            entropy_coeff: Coeficiente de entropía
            value_loss_coeff: Coeficiente de value loss
            batch_size: Tamaño de batch
            num_envs: Número de entornos paralelos
            rollout_steps: Pasos de rollout
            epochs_per_update: Épocas por actualización
            normalize_advantage: Normalizar ventajas
            normalize_reward: Normalizar recompensas
            clip_grad_norm: Norma de clipping de gradientes
            use_mixed_precision: Usar mixed precision
            device: Dispositivo
            **actor_critic_kwargs: Argumentos para el modelo
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device or torch.device("cpu")

        # Hiperparámetros
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.epochs_per_update = epochs_per_update
        self.target_kl = target_kl
        self.normalize_advantage = normalize_advantage
        self.normalize_reward = normalize_reward
        self.clip_grad_norm = clip_grad_norm

        # Crear modelo
        self.model = create_actor_critic(
            observation_space,
            action_space,
            **actor_critic_kwargs
        ).to(self.device)

        # Optimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)

        # Learning rate scheduler
        self.scheduler = None  # Será configurado en train.py si es necesario

        # Mixed precision
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Buffer de rollout
        self.buffer = create_rollout_buffer(
            buffer_size=rollout_steps,
            obs_shape=observation_space.shape,
            num_envs=num_envs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=self.device,
        )

        # Normalización de recompensas
        if normalize_reward:
            self.reward_normalizers = [
                RewardNormalizer(gamma=gamma) for _ in range(num_envs)
            ]
        else:
            self.reward_normalizers = None

        # Tracking
        self.total_steps = 0
        self.update_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.last_rollout_mean_reward = 0.0
        self.last_rollout_total_reward = 0.0
        self.last_rollout_completed_episodes = 0
        self.last_rollout_episode_returns = []
        self.last_rollout_episode_lengths = []

        # LSTM hidden state
        self.lstm_hidden = None

    def _prepare_obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convierte observaciones NHWC/HWC al formato NCHW de la CNN."""
        obs_tensor = torch.from_numpy(obs).to(self.device)
        if obs_tensor.ndim == 3:
            obs_tensor = obs_tensor.unsqueeze(0)

        if obs_tensor.ndim == 4 and obs_tensor.shape[-1] in [1, 3, 4]:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        return obs_tensor.float()

    def _normalize_rewards(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Normaliza recompensas escalares o vectorizadas."""
        reward_array = np.asarray(rewards, dtype=np.float32)
        if self.reward_normalizers is None:
            return reward_array

        original_shape = reward_array.shape
        flat_rewards = reward_array.reshape(-1)
        flat_dones = np.asarray(dones, dtype=np.bool_).reshape(-1)
        normalized = np.empty_like(flat_rewards, dtype=np.float32)

        for i, reward in enumerate(flat_rewards):
            normalizer = self.reward_normalizers[min(i, len(self.reward_normalizers) - 1)]
            normalized[i] = normalizer.normalize(float(reward))
            if i < len(flat_dones) and flat_dones[i]:
                normalizer.reset()

        return normalized.reshape(original_shape)

    def _reset_lstm_hidden_for_done(self, dones: np.ndarray):
        """Limpia el estado recurrente de los entornos que terminaron."""
        if self.lstm_hidden is None:
            return

        done_mask = torch.as_tensor(dones, dtype=torch.bool, device=self.device).reshape(-1)
        if not torch.any(done_mask):
            return

        def reset_hidden_tensor(hidden_tensor: torch.Tensor) -> torch.Tensor:
            hidden_tensor = hidden_tensor.detach().clone()
            if hidden_tensor.ndim >= 2 and hidden_tensor.shape[1] == done_mask.numel():
                hidden_tensor[:, done_mask, ...] = 0
            return hidden_tensor

        if isinstance(self.lstm_hidden, tuple):
            self.lstm_hidden = tuple(reset_hidden_tensor(tensor) for tensor in self.lstm_hidden)
        else:
            self.lstm_hidden = reset_hidden_tensor(self.lstm_hidden)

    def _freeze_batch_norm_stats(self):
        """Mantiene BatchNorm en eval sin apagar gradientes del resto del modelo."""
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def collect_rollout(self, env, progress_interval: int = 0) -> Tuple[float, int]:
        """
        Recolecta un rollout completo del entorno

        Args:
            env: Entorno
            progress_interval: Cada cuÃ¡ntos pasos imprimir progreso (0 desactiva)

        Returns:
            Tupla (reward promedio, steps totales)
        """
        was_training = self.model.training
        self.model.eval()

        obs, _ = env.reset()
        obs_shape = self.observation_space.shape
        actual_num_envs = obs.shape[0] if obs.ndim == len(obs_shape) + 1 else 1

        if actual_num_envs != self.num_envs:
            raise ValueError(
                f"El agente fue creado con num_envs={self.num_envs}, "
                f"pero el entorno devolvio {actual_num_envs} observaciones"
            )

        episode_rewards = np.zeros(actual_num_envs, dtype=np.float32)
        episode_lengths = np.zeros(actual_num_envs, dtype=np.int32)
        rollout_rewards = []
        rollout_episode_returns = []
        rollout_episode_lengths = []
        completed_episodes = 0

        for step in range(self.rollout_steps):
            # Obtener acción del modelo
            with torch.no_grad():
                obs_tensor = self._prepare_obs_tensor(obs)
                action, log_prob, value, self.lstm_hidden = self.model.get_action_and_value(
                    obs_tensor,
                    self.lstm_hidden
                )

                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()

            # Ejecutar acción en el entorno
            action_array = np.asarray(action, dtype=np.int64).reshape(-1)
            if actual_num_envs == 1:
                action_to_env = int(action_array[0])
            else:
                action_to_env = action_array

            next_obs, reward, done, info = env.step(action_to_env)
            reward_array = np.asarray(reward, dtype=np.float32).reshape(actual_num_envs)
            done_array = np.asarray(done, dtype=np.bool_).reshape(actual_num_envs)
            rollout_rewards.append(reward_array.copy())

            # Normalizar recompensa
            normalized_reward = self._normalize_rewards(reward_array, done_array)

            # Añadir al buffer
            self.buffer.add(
                obs=obs,
                action=action_array,
                reward=normalized_reward,
                value=np.asarray(value, dtype=np.float32).reshape(actual_num_envs),
                log_prob=np.asarray(log_prob, dtype=np.float32).reshape(actual_num_envs),
                done=done_array.astype(np.float32),
            )

            # Tracking
            episode_rewards += reward_array
            episode_lengths += 1
            self.total_steps += actual_num_envs

            # Reset si el episodio terminó
            for env_idx, is_done in enumerate(done_array):
                if is_done:
                    episode_return = float(episode_rewards[env_idx])
                    episode_length = int(episode_lengths[env_idx])
                    self.episode_returns.append(episode_return)
                    self.episode_lengths.append(episode_length)
                    rollout_episode_returns.append(episode_return)
                    rollout_episode_lengths.append(episode_length)
                    episode_rewards[env_idx] = 0.0
                    episode_lengths[env_idx] = 0
                    completed_episodes += 1

            if actual_num_envs == 1 and done_array[0]:
                self.lstm_hidden = None
                obs, _ = env.reset()
            else:
                self._reset_lstm_hidden_for_done(done_array)
                obs = next_obs

            if progress_interval and (
                (step + 1) % progress_interval == 0 or step + 1 == self.rollout_steps
            ):
                collected = (step + 1) * actual_num_envs
                total = self.rollout_steps * actual_num_envs
                print(f"  Rollout: {collected:,} / {total:,} transiciones", flush=True)

        # Calcular value del último estado
        with torch.no_grad():
            obs_tensor = self._prepare_obs_tensor(obs)
            last_values = self.model.get_value(obs_tensor).cpu().numpy()

        # Computar ventajas y retornos
        self.buffer.compute_advantages_and_returns(
            last_values,
            normalize_advantages=self.normalize_advantage
        )

        rollout_rewards = np.concatenate(rollout_rewards) if rollout_rewards else np.array([0.0])
        self.last_rollout_mean_reward = float(np.mean(rollout_rewards))
        self.last_rollout_total_reward = float(np.sum(rollout_rewards))
        self.last_rollout_completed_episodes = completed_episodes
        self.last_rollout_episode_returns = rollout_episode_returns
        self.last_rollout_episode_lengths = rollout_episode_lengths

        # Retornar statistics
        if completed_episodes > 0 and self.episode_returns:
            avg_reward = np.mean(self.episode_returns[-100:])
        else:
            avg_reward = self.last_rollout_mean_reward

        if was_training:
            self.model.train()

        return avg_reward, self.total_steps

    def update(self) -> Dict[str, float]:
        """
        Actualiza el modelo usando el rollout buffer

        Returns:
            Diccionario con métricas de entrenamiento
        """
        self.model.train()
        self._freeze_batch_norm_stats()

        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        kl_early_stop = False
        epochs_ran = 0
        optimizer_steps = 0

        # Múltiples épocas de entrenamiento
        for epoch in range(self.epochs_per_update):
            epochs_ran += 1
            # Iterar sobre batches
            for batch_data in self.buffer.get_batch(self.batch_size, shuffle=True):
                obs, actions, advantages, returns, values, old_log_probs = batch_data

                # Transponer observaciones de (batch, height, width, channels) a (batch, channels, height, width)
                if obs.ndim == 4 and obs.shape[-1] in [1, 3, 4]:
                    obs = obs.permute(0, 3, 1, 2)

                # Forward pass
                if self.use_mixed_precision:
                    with autocast(dtype=torch.float16):
                        new_log_probs, new_values, entropy, _ = self.model.evaluate_actions(obs, actions)
                        policy_loss = self._compute_policy_loss(
                            new_log_probs, old_log_probs, advantages
                        )
                        value_loss = self._compute_value_loss(
                            new_values, returns, values
                        )
                        loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy.mean()

                    approx_kl, clip_fraction = self._compute_policy_stats(
                        new_log_probs,
                        old_log_probs,
                    )
                    if self._should_stop_for_kl(approx_kl):
                        kl_early_stop = True
                        metrics["policy_loss"].append(policy_loss.item())
                        metrics["value_loss"].append(value_loss.item())
                        metrics["entropy"].append(entropy.mean().item())
                        metrics["approx_kl"].append(approx_kl)
                        metrics["clip_fraction"].append(clip_fraction)
                        break

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    new_log_probs, new_values, entropy, _ = self.model.evaluate_actions(obs, actions)
                    policy_loss = self._compute_policy_loss(
                        new_log_probs, old_log_probs, advantages
                    )
                    value_loss = self._compute_value_loss(
                        new_values, returns, values
                    )
                    loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy.mean()

                    approx_kl, clip_fraction = self._compute_policy_stats(
                        new_log_probs,
                        old_log_probs,
                    )
                    if self._should_stop_for_kl(approx_kl):
                        kl_early_stop = True
                        metrics["policy_loss"].append(policy_loss.item())
                        metrics["value_loss"].append(value_loss.item())
                        metrics["entropy"].append(entropy.mean().item())
                        metrics["approx_kl"].append(approx_kl)
                        metrics["clip_fraction"].append(clip_fraction)
                        break

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                optimizer_steps += 1

                # Rastrear métricas
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_fraction"].append(clip_fraction)

            if kl_early_stop:
                break

        # Resetear buffer
        self.buffer.reset()
        self.update_count += 1

        # Promediar métricas
        avg_metrics = {
            k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()
        }
        avg_metrics["kl_early_stop"] = int(kl_early_stop)
        avg_metrics["update_epochs_ran"] = epochs_ran
        avg_metrics["optimizer_steps"] = optimizer_steps
        avg_metrics["max_approx_kl"] = max(metrics["approx_kl"]) if metrics["approx_kl"] else 0.0
        avg_metrics["max_clip_fraction"] = (
            max(metrics["clip_fraction"]) if metrics["clip_fraction"] else 0.0
        )

        return avg_metrics

    def _compute_policy_stats(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> Tuple[float, float]:
        """Calcula KL aproximado y fraccion de acciones clippeadas."""
        with torch.no_grad():
            log_ratio = (new_log_probs - old_log_probs).float()
            ratio = torch.exp(log_ratio)
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()

        return float(approx_kl.item()), float(clip_fraction.item())

    def _should_stop_for_kl(self, approx_kl: float) -> bool:
        """Detiene el update cuando PPO ya cambio demasiado la politica."""
        return (
            self.target_kl is not None
            and self.target_kl > 0.0
            and approx_kl > self.target_kl
        )

    def _compute_policy_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula la pérdida de política con clipping PPO"""
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages

        loss = -torch.min(surr1, surr2).mean()
        return loss

    def _compute_value_loss(
        self,
        new_values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula la pérdida de value con clipping"""
        # Clipping de value function
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values,
            -self.clip_range_vf,
            self.clip_range_vf,
        )

        # MSE loss
        loss1 = (new_values - returns).pow(2)
        loss2 = (value_pred_clipped - returns).pow(2)

        loss = torch.max(loss1, loss2).mean()
        return loss

    def get_model(self) -> nn.Module:
        """Retorna el modelo"""
        return self.model

    def get_optimizer(self) -> optim.Optimizer:
        """Retorna el optimizador"""
        return self.optimizer

    def set_learning_rate(self, lr: float):
        """Ajusta el learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save(self, path: Path):
        """Guarda el agente"""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
        }
        torch.save(checkpoint, path)

    def load(self, path: Path):
        """Carga el agente"""
        checkpoint = torch.load(path, map_location=self.device)
        model_state = checkpoint.get("model_state", checkpoint.get("model"))
        optimizer_state = checkpoint.get("optimizer_state", checkpoint.get("optimizer"))
        if model_state is None:
            raise KeyError("Checkpoint sin 'model_state' ni 'model'")

        self.model.load_state_dict(model_state)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        training_state = checkpoint.get("training_state", {})
        self.total_steps = checkpoint.get("total_steps", training_state.get("total_steps", 0))
        self.update_count = checkpoint.get("update_count", training_state.get("update_count", 0))


def create_ppo_agent(
    observation_space,
    action_space,
    device: torch.device = None,
    **kwargs
) -> PPOAgent:
    """
    Factory function para crear un agente PPO

    Args:
        observation_space: Espacio de observación
        action_space: Espacio de acciones
        device: Dispositivo
        **kwargs: Argumentos adicionales

    Returns:
        PPOAgent
    """
    return PPOAgent(
        observation_space,
        action_space,
        device=device,
        **kwargs
    )
