"""
Sistema de logging para entrenamiento
Integración con TensorBoard y Weights & Biases
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


class MetricsLogger:
    """Logger centralizado para métricas de entrenamiento"""

    def __init__(
        self,
        log_dir: Path,
        experiment_name: str = "super_mario_rl",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
    ):
        """
        Inicializa el logger de métricas

        Args:
            log_dir: Directorio para guardar logs
            experiment_name: Nombre del experimento
            use_wandb: Usar Weights & Biases
            use_tensorboard: Usar TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("TensorBoard no instalado. Deshabilitando TensorBoard.")
                self.use_tensorboard = False

        # Setup Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project="super_mario_rl", name=experiment_name)
                self.wandb = wandb
            except ImportError:
                print("Weights & Biases no instalado. Deshabilitando W&B.")
                self.use_wandb = False

        # Métricas locales
        self.metrics = {}
        self.step = 0

        # Setup logging
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup del logger de Python"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_dir / "training.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Registra métricas

        Args:
            metrics: Diccionario de métricas
            step: Paso de entrenamiento (si es None, usa contador interno)
        """
        if step is None:
            step = self.step
        else:
            self.step = step

        # Guardar localmente
        self.metrics.update(metrics)

        # TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, step)
            self.writer.flush()

        # Weights & Biases
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def log_histogram(self, name: str, values, step: Optional[int] = None):
        """
        Registra histogramas (solo TensorBoard)

        Args:
            name: Nombre del histograma
            values: Valores
            step: Paso de entrenamiento
        """
        if step is None:
            step = self.step

        if self.use_tensorboard:
            self.writer.add_histogram(name, values, step)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """
        Registra un escalar

        Args:
            name: Nombre del escalar
            value: Valor
            step: Paso de entrenamiento
        """
        if step is None:
            step = self.step

        if self.use_tensorboard:
            self.writer.add_scalar(name, value, step)

        if self.use_wandb:
            self.wandb.log({name: value}, step=step)

    def log_text(self, name: str, text: str):
        """
        Registra texto

        Args:
            name: Nombre
            text: Texto a registrar
        """
        self.logger.info(f"{name}: {text}")

        if self.use_wandb:
            self.wandb.log({name: text})

    def log_config(self, config: Dict[str, Any]):
        """
        Registra configuración del experimento

        Args:
            config: Diccionario de configuración
        """
        # Guardar a JSON
        with open(self.log_dir / "config.json", "w") as f:
            # Convertir valores no serializables a strings
            config_serializable = {
                k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
                for k, v in config.items()
            }
            json.dump(config_serializable, f, indent=2)

        self.logger.info(f"Configuración guardada en config.json")

        # Weights & Biases
        if self.use_wandb:
            self.wandb.config.update(config)

    def save_metrics(self):
        """Guarda métricas a JSON"""
        with open(self.log_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

    def close(self):
        """Cierra los loggers"""
        if self.use_tensorboard:
            self.writer.close()

        if self.use_wandb:
            self.wandb.finish()

        self.logger.info("Logger cerrado")


def create_logger(
    log_dir: Path,
    experiment_name: str = "super_mario_rl",
    use_wandb: bool = False,
    use_tensorboard: bool = True,
) -> MetricsLogger:
    """
    Factory function para crear un logger

    Args:
        log_dir: Directorio de logs
        experiment_name: Nombre del experimento
        use_wandb: Usar W&B
        use_tensorboard: Usar TensorBoard

    Returns:
        MetricsLogger configurado
    """
    return MetricsLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
    )
