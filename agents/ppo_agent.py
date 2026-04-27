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
from utils.metrics import RunningMeanStd


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
            self.reward_normalizer = RunningMeanStd()
        else:
            self.reward_normalizer = None

        # Tracking
        self.total_steps = 0
        self.update_count = 0
        self.episode_returns = []
        self.episode_lengths = []

        # LSTM hidden state
        self.lstm_hidden = None

    def collect_rollout(self, env) -> Tuple[float, int]:
        """
        Recolecta un rollout completo del entorno

        Args:
            env: Entorno

        Returns:
            Tupla (reward promedio, steps totales)
        """
        obs, _ = env.reset()

        episode_rewards = []
        episode_length = 0

        for step in range(self.rollout_steps):
            # Obtener acción del modelo
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(self.device)
                if obs_tensor.ndim == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)  # Agregar batch dim

                action, log_prob, value, self.lstm_hidden = self.model.actor_critic.get_action_and_value(
                    obs_tensor,
                    self.lstm_hidden
                )

                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()

            # Ejecutar acción en el entorno
            next_obs, reward, done, info = env.step(action[0] if isinstance(action, np.ndarray) and action.ndim > 0 else action)

            # Normalizar recompensa
            if self.reward_normalizer is not None:
                reward = self.reward_normalizer.normalize(float(reward))

            # Añadir al buffer
            self.buffer.add(
                obs=obs,
                action=action[0] if isinstance(action, np.ndarray) and action.ndim > 0 else action,
                reward=float(reward),
                value=value[0] if isinstance(value, np.ndarray) and value.ndim > 0 else value,
                log_prob=log_prob[0] if isinstance(log_prob, np.ndarray) and log_prob.ndim > 0 else log_prob,
                done=float(done),
            )

            # Tracking
            episode_rewards.append(reward)
            episode_length += 1
            self.total_steps += 1

            # Reset si el episodio terminó
            if done:
                self.episode_returns.append(sum(episode_rewards))
                self.episode_lengths.append(episode_length)
                episode_rewards = []
                episode_length = 0
                self.lstm_hidden = None
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Calcular value del último estado
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(self.device)
            if obs_tensor.ndim == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            last_values = self.model.get_value(obs_tensor).cpu().numpy()

        # Computar ventajas y retornos
        self.buffer.compute_advantages_and_returns(
            last_values,
            normalize_advantages=self.normalize_advantage
        )

        # Retornar statistics
        if self.episode_returns:
            avg_reward = np.mean(self.episode_returns[-100:])
        else:
            avg_reward = 0.0

        return avg_reward, self.total_steps

    def update(self) -> Dict[str, float]:
        """
        Actualiza el modelo usando el rollout buffer

        Returns:
            Diccionario con métricas de entrenamiento
        """
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }

        # Múltiples épocas de entrenamiento
        for epoch in range(self.epochs_per_update):
            # Iterar sobre batches
            for batch_data in self.buffer.get_batch(self.batch_size, shuffle=True):
                obs, actions, advantages, returns, values, old_log_probs = batch_data

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

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()

                # Rastrear métricas
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())

                # Aproximar KL divergence
                approx_kl = (old_log_probs - new_log_probs).mean().item()
                metrics["approx_kl"].append(approx_kl)

        # Resetear buffer
        self.buffer.reset()
        self.update_count += 1

        # Promediar métricas
        avg_metrics = {
            k: float(np.mean(v)) for k, v in metrics.items()
        }

        return avg_metrics

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
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.update_count = checkpoint.get("update_count", 0)


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
