"""
Modelo Actor-Critic con CNN + LSTM para PPO
Combina la política (Actor) y value function (Critic)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .cnn_backbone import create_cnn_backbone
from .recurrent_module import create_recurrent_module
from .distributions import CategoricalDistribution, DistributionFactory


class ActorCriticNetwork(nn.Module):
    """
    Red Actor-Critic con arquitectura CNN + LSTM
    Procesa frames con CNN, memoria con LSTM, produce policy y value
    """

    def __init__(
        self,
        observation_space,
        action_space,
        cnn_features: list = None,
        lstm_hidden_dim: int = 512,
        policy_hidden_dim: int = 512,
        value_hidden_dim: int = 512,
        use_lstm: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
        init_scale: float = 0.5,
    ):
        """
        Inicializa la red Actor-Critic

        Args:
            observation_space: Espacio de observación
            action_space: Espacio de acciones
            cnn_features: Features para CNN
            lstm_hidden_dim: Dimensión hidden del LSTM
            policy_hidden_dim: Dimensión hidden de la política
            value_hidden_dim: Dimensión hidden del value
            use_lstm: Usar LSTM
            use_batch_norm: Usar batch normalization
            use_layer_norm: Usar layer normalization
            init_scale: Escala de inicialización de pesos
        """
        super().__init__()

        if cnn_features is None:
            cnn_features = [32, 64, 64]

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_actions = action_space.n
        self.use_lstm = use_lstm
        self.init_scale = init_scale

        # CNN Backbone
        self.cnn = create_cnn_backbone(
            input_channels=observation_space.shape[-1],  # Frame stack
            feature_dims=cnn_features,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # Flatten para conectar con LSTM/MLP
        cnn_output_dim = self.cnn.output_dim

        # LSTM si está habilitado
        if use_lstm:
            self.lstm = create_recurrent_module(
                input_dim=cnn_output_dim,
                hidden_dim=lstm_hidden_dim,
                rnn_type="lstm",
                use_layer_norm=use_layer_norm,
            )
            self.lstm_hidden_dim = lstm_hidden_dim
            feature_dim = lstm_hidden_dim
        else:
            self.lstm = None
            # MLP simple si no hay LSTM
            self.feature_fc = nn.Sequential(
                nn.Linear(cnn_output_dim, lstm_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(lstm_hidden_dim) if use_layer_norm else nn.Identity(),
            )
            feature_dim = lstm_hidden_dim

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(policy_hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(policy_hidden_dim, self.num_actions),
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, value_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(value_hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(value_hidden_dim, 1),
        )

        # Inicializar pesos
        self._init_weights(init_scale)

    def _init_weights(self, scale: float = 0.5):
        """Inicializa los pesos"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=scale)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=scale)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        obs: torch.Tensor,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass

        Args:
            obs: Observación (batch, channels, height, width)
            lstm_hidden: Hidden state del LSTM

        Returns:
            Tupla (policy_logits, value, lstm_hidden_new)
        """
        # CNN
        cnn_out = self.cnn(obs)  # (batch, features, h, w)
        batch_size = cnn_out.shape[0]

        # Flatten
        cnn_out = cnn_out.reshape(batch_size, -1)  # (batch, flattened_features)

        # LSTM o MLP
        if self.use_lstm:
            # Reshape para LSTM (seq_len = 1)
            cnn_out = cnn_out.unsqueeze(1)  # (batch, 1, features)
            lstm_out, lstm_hidden = self.lstm(cnn_out, lstm_hidden)
            features = lstm_out.squeeze(1)  # (batch, hidden_dim)
        else:
            features = self.feature_fc(cnn_out)
            lstm_hidden = None

        # Policy y Value
        policy_logits = self.policy_head(features)  # (batch, num_actions)
        value = self.value_head(features)  # (batch, 1)

        return policy_logits, value, lstm_hidden

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Obtiene acción y value (para entrenamiento)

        Args:
            obs: Observación
            lstm_hidden: Hidden state del LSTM

        Returns:
            Tupla (action, log_prob, value, lstm_hidden)
        """
        policy_logits, value, new_lstm_hidden = self(obs, lstm_hidden)

        # Crear distribución y muestrear
        distribution = DistributionFactory.create_categorical(policy_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action, log_prob, value.squeeze(-1), new_lstm_hidden

    def get_action_deterministic(
        self,
        obs: torch.Tensor,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Obtiene acción determinística (para evaluación)

        Args:
            obs: Observación
            lstm_hidden: Hidden state del LSTM

        Returns:
            Tupla (action, lstm_hidden)
        """
        policy_logits, _, new_lstm_hidden = self(obs, lstm_hidden)
        action = policy_logits.argmax(dim=-1)

        return action, new_lstm_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Evalúa acciones (para PPO update)

        Args:
            obs: Observación
            actions: Acciones
            lstm_hidden: Hidden state del LSTM

        Returns:
            Tupla (log_prob, value, entropy, lstm_hidden)
        """
        policy_logits, value, new_lstm_hidden = self(obs, lstm_hidden)

        # Distribución
        distribution = DistributionFactory.create_categorical(policy_logits)

        # Log probability de las acciones
        log_prob = distribution.log_prob(actions)

        # Entropía
        entropy = distribution.entropy()

        return log_prob, value.squeeze(-1), entropy, new_lstm_hidden

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Obtiene el value de una observación"""
        _, value, _ = self(obs)
        return value.squeeze(-1)


class PPOActorCritic(nn.Module):
    """
    Wrapper para el modelo Actor-Critic optimizado para PPO
    Incluye métodos para forward, valor, y evaluación
    """

    def __init__(self, actor_critic: ActorCriticNetwork):
        """
        Inicializa el wrapper PPO

        Args:
            actor_critic: Red Actor-Critic
        """
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs: torch.Tensor, lstm_hidden=None):
        """Forward pass"""
        return self.actor_critic(obs, lstm_hidden)

    def get_action_and_value(self, obs: torch.Tensor, lstm_hidden=None):
        """Obtiene acción y value"""
        return self.actor_critic.get_action_and_value(obs, lstm_hidden)

    def get_action_deterministic(self, obs: torch.Tensor, lstm_hidden=None):
        """Obtiene acción determinística"""
        return self.actor_critic.get_action_deterministic(obs, lstm_hidden)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, lstm_hidden=None):
        """Evalúa acciones"""
        return self.actor_critic.evaluate_actions(obs, actions, lstm_hidden)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Obtiene value"""
        return self.actor_critic.get_value(obs)


def create_actor_critic(
    observation_space,
    action_space,
    **kwargs
) -> ActorCriticNetwork:
    """
    Factory function para crear modelo Actor-Critic

    Args:
        observation_space: Espacio de observación
        action_space: Espacio de acciones
        **kwargs: Argumentos adicionales

    Returns:
        ActorCriticNetwork
    """
    return ActorCriticNetwork(
        observation_space,
        action_space,
        **kwargs
    )
