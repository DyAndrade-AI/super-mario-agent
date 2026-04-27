"""
Configuración centralizada del proyecto
Maneja paths, logging y configuración general
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectPaths:
    """Rutas del proyecto"""
    PROJECT_ROOT: Path
    CHECKPOINTS_DIR: Path
    VIDEOS_DIR: Path
    LOGS_DIR: Path
    PLOTS_DIR: Path

    def __post_init__(self):
        """Crear directorios si no existen"""
        for path in [self.CHECKPOINTS_DIR, self.VIDEOS_DIR, self.LOGS_DIR, self.PLOTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


def get_project_paths(project_root: Optional[str] = None) -> ProjectPaths:
    """
    Obtiene las rutas del proyecto

    Args:
        project_root: Raíz del proyecto (por defecto, directorio de este script)

    Returns:
        ProjectPaths con todas las rutas configuradas
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(project_root)

    return ProjectPaths(
        PROJECT_ROOT=project_root,
        CHECKPOINTS_DIR=project_root / "outputs" / "checkpoints",
        VIDEOS_DIR=project_root / "outputs" / "videos",
        LOGS_DIR=project_root / "outputs" / "logs",
        PLOTS_DIR=project_root / "outputs" / "plots",
    )


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento centralizada"""
    # Paths
    project_paths: ProjectPaths

    # Ambiente
    world: int = 1
    stage: int = 1

    # Training
    total_timesteps: int = 50_000_000
    eval_freq: int = 50_000
    checkpoint_freq: int = 500_000

    # Hardware
    use_gpu: bool = True
    device_id: int = 0

    # Logging
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_dir: Optional[Path] = None

    def __post_init__(self):
        if self.log_dir is None:
            self.log_dir = self.project_paths.LOGS_DIR


def create_training_config(
    world: int = 1,
    stage: int = 1,
    total_timesteps: int = 50_000_000,
    use_gpu: bool = True,
    use_wandb: bool = False,
    project_root: Optional[str] = None,
) -> TrainingConfig:
    """
    Crea una configuración de entrenamiento

    Args:
        world: Mundo de Super Mario
        stage: Etapa del mundo
        total_timesteps: Pasos totales de entrenamiento
        use_gpu: Usar GPU
        use_wandb: Usar Weights & Biases
        project_root: Raíz del proyecto

    Returns:
        TrainingConfig configurado
    """
    paths = get_project_paths(project_root)

    return TrainingConfig(
        project_paths=paths,
        world=world,
        stage=stage,
        total_timesteps=total_timesteps,
        use_gpu=use_gpu,
        use_wandb=use_wandb,
    )
