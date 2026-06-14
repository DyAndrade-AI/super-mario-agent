"""
Utilidades para rastrear y calcular métricas de entrenamiento
"""
from collections import deque
from typing import Dict, Optional
import numpy as np


class RunningMeanStd:
    """Calcula media y desviación estándar corriente"""

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """
        Inicializa running stats

        Args:
            epsilon: Pequeño valor para evitar división por cero
            shape: Shape de los datos
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """
        Actualiza las estadísticas

        Args:
            x: Batch de datos
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ):
        """
        Actualiza desde momentos (Welford's algorithm)
        """
        delta = batch_mean - self.mean

        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count


class RewardNormalizer:
    """Normaliza recompensas durante entrenamiento"""

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        """
        Inicializa el normalizador de recompensas

        Args:
            gamma: Factor de descuento
            epsilon: Pequeño valor para estabilidad
        """
        self.return_rms = RunningMeanStd(shape=())
        self.returns = 0.0
        self.gamma = gamma
        self.epsilon = epsilon

    def normalize(self, reward: float) -> float:
        """
        Normaliza una recompensa

        Args:
            reward: Recompensa a normalizar

        Returns:
            Recompensa normalizada
        """
        self.returns = reward + self.gamma * self.returns
        self.return_rms.update(np.array([self.returns]))

        normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return normalized_reward

    def reset(self):
        """Resetea el estado interno"""
        self.returns = 0.0


class MetricsTracker:
    """Rastreador de métricas de entrenamiento"""

    def __init__(self, window_size: int = 100):
        """
        Inicializa el rastreador

        Args:
            window_size: Tamaño de la ventana para promedios móviles
        """
        self.window_size = window_size
        self.metrics = {
            "episode_reward": deque(maxlen=window_size),
            "episode_length": deque(maxlen=window_size),
            "q_loss": deque(maxlen=window_size),
            "td_error": deque(maxlen=window_size),
            "q_value": deque(maxlen=window_size),
            "epsilon": deque(maxlen=window_size),
            "learning_rate": deque(maxlen=window_size),
        }

    def add_episode_metric(self, reward: float, length: int):
        """
        Añade métrica de episodio

        Args:
            reward: Recompensa total del episodio
            length: Duración del episodio
        """
        self.metrics["episode_reward"].append(reward)
        self.metrics["episode_length"].append(length)

    def add_training_metric(
        self,
        q_loss: float,
        td_error: float,
        q_value: float,
        epsilon: float,
        learning_rate: float,
    ):
        """
        Añade métrica de entrenamiento

        Args:
            q_loss: Pérdida temporal-difference
            td_error: Error TD absoluto medio
            q_value: Q-value medio del batch
            epsilon: Exploración epsilon-greedy actual
            learning_rate: Learning rate actual
        """
        self.metrics["q_loss"].append(q_loss)
        self.metrics["td_error"].append(td_error)
        self.metrics["q_value"].append(q_value)
        self.metrics["epsilon"].append(epsilon)
        self.metrics["learning_rate"].append(learning_rate)

    def get_mean_metrics(self) -> Dict[str, float]:
        """
        Obtiene promedios de métricas

        Returns:
            Diccionario con promedios
        """
        return {
            "mean_episode_reward": float(np.mean(self.metrics["episode_reward"]))
            if self.metrics["episode_reward"] else 0.0,
            "mean_episode_length": float(np.mean(self.metrics["episode_length"]))
            if self.metrics["episode_length"] else 0.0,
            "mean_q_loss": float(np.mean(self.metrics["q_loss"]))
            if self.metrics["q_loss"] else 0.0,
            "mean_td_error": float(np.mean(self.metrics["td_error"]))
            if self.metrics["td_error"] else 0.0,
            "mean_q_value": float(np.mean(self.metrics["q_value"]))
            if self.metrics["q_value"] else 0.0,
            "mean_epsilon": float(np.mean(self.metrics["epsilon"]))
            if self.metrics["epsilon"] else 0.0,
            "last_learning_rate": float(self.metrics["learning_rate"][-1])
            if self.metrics["learning_rate"] else 0.0,
        }

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Obtiene estadísticas completas

        Returns:
            Diccionario con media, std, min, max de cada métrica
        """
        stats = {}
        for key, values in self.metrics.items():
            if values:
                values_array = np.array(list(values))
                stats[key] = {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                }
            else:
                stats[key] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return stats

    def reset(self):
        """Resetea todas las métricas"""
        for key in self.metrics:
            self.metrics[key].clear()


def create_metrics_tracker(window_size: int = 100) -> MetricsTracker:
    """
    Factory function para crear un rastreador de métricas

    Args:
        window_size: Tamaño de ventana

    Returns:
        MetricsTracker
    """
    return MetricsTracker(window_size=window_size)
