"""
Distribuciones de probabilidad para política y value
Abstracciones para facilitar el muestreo y cálculo de probabilidades
"""
import torch
import torch.nn as nn
import torch.distributions as td
from typing import Tuple, Optional


class CategoricalDistribution:
    """Distribución categórica para acciones discretas"""

    def __init__(self, logits: torch.Tensor):
        """
        Inicializa distribución categórica

        Args:
            logits: Logits de acciones (batch, num_actions)
        """
        self.distribution = td.Categorical(logits=logits)

    def sample(self) -> torch.Tensor:
        """Muestrea una acción"""
        return self.distribution.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Calcula log probability"""
        return self.distribution.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """Calcula entropía"""
        return self.distribution.entropy()

    def get_probs(self) -> torch.Tensor:
        """Obtiene probabilidades"""
        return self.distribution.probs


class NormalDistribution:
    """Distribución normal para acciones continuas"""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        """
        Inicializa distribución normal

        Args:
            mean: Media (batch, action_dim)
            std: Desviación estándar (batch, action_dim)
        """
        self.distribution = td.Normal(mean, std)

    def sample(self) -> torch.Tensor:
        """Muestrea una acción"""
        return self.distribution.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Calcula log probability"""
        log_prob = self.distribution.log_prob(action)
        # Sumar sobre dimensión de acción
        return log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Calcula entropía"""
        entropy = self.distribution.entropy()
        return entropy.sum(dim=-1)

    def get_mean(self) -> torch.Tensor:
        """Obtiene la media"""
        return self.distribution.mean

    def get_std(self) -> torch.Tensor:
        """Obtiene la desviación estándar"""
        return self.distribution.stddev


class BernoulliDistribution:
    """Distribución Bernoulli para acciones binarias"""

    def __init__(self, logits: torch.Tensor):
        """
        Inicializa distribución Bernoulli

        Args:
            logits: Logits de acciones (batch, num_actions)
        """
        self.distribution = td.Bernoulli(logits=logits)

    def sample(self) -> torch.Tensor:
        """Muestrea una acción"""
        return self.distribution.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Calcula log probability"""
        log_prob = self.distribution.log_prob(action)
        return log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Calcula entropía"""
        entropy = self.distribution.entropy()
        return entropy.sum(dim=-1)


class DistributionFactory:
    """Factory para crear distribuciones"""

    @staticmethod
    def create_categorical(logits: torch.Tensor) -> CategoricalDistribution:
        """Crea distribución categórica"""
        return CategoricalDistribution(logits)

    @staticmethod
    def create_normal(
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> NormalDistribution:
        """Crea distribución normal"""
        # Asegurar que std sea positivo
        std = torch.clamp(std, min=1e-4)
        return NormalDistribution(mean, std)

    @staticmethod
    def create_bernoulli(logits: torch.Tensor) -> BernoulliDistribution:
        """Crea distribución Bernoulli"""
        return BernoulliDistribution(logits)


def log_prob_of_action(
    distribution,
    action: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula log probability de una acción en una distribución

    Args:
        distribution: Distribución (CategoricalDistribution, etc.)
        action: Acción

    Returns:
        Log probability
    """
    return distribution.log_prob(action)


def entropy_of_distribution(distribution) -> torch.Tensor:
    """
    Calcula entropía de una distribución

    Args:
        distribution: Distribución

    Returns:
        Entropía
    """
    return distribution.entropy()
