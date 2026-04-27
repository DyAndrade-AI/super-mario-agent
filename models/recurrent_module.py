"""
Módulo recurrente LSTM para memoria temporal
Permite al agente recordar estados pasados
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMMemory(nn.Module):
    """
    Módulo LSTM para proporcionar memoria al agente
    Procesa secuencias de características CNN
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        """
        Inicializa el módulo LSTM

        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión del hidden state
            num_layers: Número de capas LSTM
            dropout: Dropout entre capas
            use_layer_norm: Usar layer normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        # Inicializar pesos
        self._init_weights()

    def _init_weights(self):
        """Inicializa pesos de las capas LSTM"""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                # Input weights: uniformemente distribuidos
                nn.init.uniform_(param, -0.5, 0.5)
            elif "weight_hh" in name:
                # Hidden weights: ortogonales
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                # Bias: ceros
                nn.init.constant_(param, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Entrada (batch, seq_len, input_dim)
            hidden: Hidden state (h, c) previo

        Returns:
            Tupla (output, (h, c))
        """
        lstm_out, (h, c) = self.lstm(x, hidden)

        # Layer normalization
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)

        return lstm_out, (h, c)

    def get_initial_hidden(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene el hidden state inicial

        Args:
            batch_size: Tamaño de batch
            device: Dispositivo

        Returns:
            Tupla (h, c) de tamaño (num_layers, batch_size, hidden_dim)
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        if device is not None:
            h = h.to(device)
            c = c.to(device)

        return h, c


class GRUMemory(nn.Module):
    """
    Módulo GRU como alternativa a LSTM (más rápido, menos parámetros)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        """
        Inicializa el módulo GRU

        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión del hidden state
            num_layers: Número de capas
            dropout: Dropout
            use_layer_norm: Usar layer normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Inicializa pesos"""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.uniform_(param, -0.5, 0.5)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Entrada (batch, seq_len, input_dim)
            hidden: Hidden state previo (num_layers, batch, hidden_dim)

        Returns:
            Tupla (output, h)
        """
        gru_out, h = self.gru(x, hidden)

        if self.use_layer_norm:
            gru_out = self.layer_norm(gru_out)

        return gru_out, h

    def get_initial_hidden(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Obtiene el hidden state inicial

        Args:
            batch_size: Tamaño de batch
            device: Dispositivo

        Returns:
            Hidden state de tamaño (num_layers, batch_size, hidden_dim)
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        if device is not None:
            h = h.to(device)

        return h


class RecurrentModule(nn.Module):
    """
    Wrapper que proporciona flexibilidad para elegir entre LSTM o GRU
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        **kwargs
    ):
        """
        Inicializa el módulo recurrente

        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión del hidden state
            num_layers: Número de capas
            rnn_type: Tipo de RNN ("lstm" o "gru")
            **kwargs: Argumentos adicionales
        """
        super().__init__()

        self.rnn_type = rnn_type.lower()

        if self.rnn_type == "lstm":
            self.rnn = LSTMMemory(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                **kwargs
            )
        elif self.rnn_type == "gru":
            self.rnn = GRUMemory(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                **kwargs
            )
        else:
            raise ValueError(f"RNN type desconocido: {rnn_type}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, hidden=None):
        """Forward pass"""
        return self.rnn(x, hidden)

    def get_initial_hidden(self, batch_size: int, device: torch.device = None):
        """Obtiene hidden state inicial"""
        return self.rnn.get_initial_hidden(batch_size, device)


def create_recurrent_module(
    input_dim: int,
    hidden_dim: int = 512,
    num_layers: int = 1,
    rnn_type: str = "lstm",
    use_layer_norm: bool = True,
) -> RecurrentModule:
    """
    Factory function para crear un módulo recurrente

    Args:
        input_dim: Dimensión de entrada
        hidden_dim: Dimensión del hidden state
        num_layers: Número de capas
        rnn_type: Tipo de RNN
        use_layer_norm: Usar layer normalization

    Returns:
        RecurrentModule
    """
    return RecurrentModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type=rnn_type,
        use_layer_norm=use_layer_norm,
    )
