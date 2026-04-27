"""
Backbone CNN Residual para procesamiento de frames
Arquitectura profunda con bloques residuales para extracción de características
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResidualBlock(nn.Module):
    """Bloque residual convolucional"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
    ):
        """
        Inicializa un bloque residual

        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            stride: Stride de las convoluciones
            use_batch_norm: Usar batch normalization
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.stride = stride

        # Path principal
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_batch_norm,
        )

        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_batch_norm,
        )

        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)

        # Connection residual (si es necesario cambiar dimensiones)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=not use_batch_norm,
                ),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        identity = x

        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)

        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class CNNBackbone(nn.Module):
    """
    CNN Backbone residual para procesar frames de Super Mario Bros
    Basado en arquitectura DQN mejorada con bloques residuales
    """

    def __init__(
        self,
        input_channels: int = 4,
        feature_dims: List[int] = None,
        kernel_sizes: List[int] = None,
        strides: List[int] = None,
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Inicializa el backbone CNN

        Args:
            input_channels: Canales de entrada (default 4 para frame stack)
            feature_dims: Dimensiones de features [32, 64, 64]
            kernel_sizes: Tamaños de kernel [8, 4, 3]
            strides: Strides [4, 2, 1]
            use_batch_norm: Usar batch normalization
            use_layer_norm: Usar layer normalization al final
            activation: Tipo de activación
        """
        super().__init__()

        if feature_dims is None:
            feature_dims = [32, 64, 64]
        if kernel_sizes is None:
            kernel_sizes = [8, 4, 3]
        if strides is None:
            strides = [4, 2, 1]

        self.input_channels = input_channels
        self.feature_dims = feature_dims
        self.activation = activation
        self.use_layer_norm = use_layer_norm

        # Construir capas convolucionales
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_channels = input_channels
        for out_channels, kernel_size, stride in zip(feature_dims, kernel_sizes, strides):
            # Capa convolucional
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    bias=not use_batch_norm,
                )
            )

            # Normalización
            if use_batch_norm:
                self.norm_layers.append(nn.BatchNorm2d(out_channels))
            else:
                self.norm_layers.append(nn.Identity())

            in_channels = out_channels

        # Bloques residuales adicionales
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(feature_dims[-1], feature_dims[-1], stride=1, use_batch_norm=use_batch_norm),
            ResidualBlock(feature_dims[-1], feature_dims[-1], stride=1, use_batch_norm=use_batch_norm),
        ])

        # Layer normalization final
        if use_layer_norm:
            self.final_layer_norm = nn.LayerNorm(feature_dims[-1])

        # Calcular número de features de salida
        self.output_dim = self._calculate_output_dim()

    def _calculate_output_dim(self) -> int:
        """
        Calcula la dimensión de salida pasando un tensor de prueba

        Returns:
            Número de features de salida
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, 84, 84)
            output = self.forward(dummy_input)
            return output.shape[1] * output.shape[2] * output.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Tensor de entrada (batch, channels, height, width)

        Returns:
            Características extraídas
        """
        # Pasar por capas convolucionales
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = conv(x)
            x = norm(x)
            x = F.relu(x) if self.activation == "relu" else F.elu(x)

        # Pasar por bloques residuales
        for block in self.residual_blocks:
            x = block(x)

        # Layer normalization
        if self.use_layer_norm:
            batch_size, channels, height, width = x.shape
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = self.final_layer_norm(x)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        return x

    def get_output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calcula la forma de salida dada una forma de entrada

        Args:
            input_shape: Tupla (height, width)

        Returns:
            Tupla (height, width) de salida
        """
        h, w = input_shape
        for kernel_size, stride in zip(
            [8, 4, 3],
            [4, 2, 1],
        ):
            h = (h - kernel_size) // stride + 1
            w = (w - kernel_size) // stride + 1
        return h, w


class DuolingCNNBackbone(nn.Module):
    """CNN Backbone con arquitectura dueling (separación en value y advantage)"""

    def __init__(
        self,
        input_channels: int = 4,
        feature_dims: List[int] = None,
        **kwargs
    ):
        """
        Inicializa el backbone dueling

        Args:
            input_channels: Canales de entrada
            feature_dims: Dimensiones de features
            **kwargs: Argumentos para CNNBackbone
        """
        super().__init__()

        self.backbone = CNNBackbone(input_channels, feature_dims, **kwargs)
        self.output_dim = self.backbone.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)


def create_cnn_backbone(
    input_channels: int = 4,
    feature_dims: List[int] = None,
    kernel_sizes: List[int] = None,
    strides: List[int] = None,
    use_batch_norm: bool = True,
    use_layer_norm: bool = True,
) -> CNNBackbone:
    """
    Factory function para crear un backbone CNN

    Args:
        input_channels: Canales de entrada
        feature_dims: Dimensiones de features
        kernel_sizes: Tamaños de kernel
        strides: Strides
        use_batch_norm: Usar batch normalization
        use_layer_norm: Usar layer normalization

    Returns:
        CNNBackbone
    """
    return CNNBackbone(
        input_channels=input_channels,
        feature_dims=feature_dims,
        kernel_sizes=kernel_sizes,
        strides=strides,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
    )
