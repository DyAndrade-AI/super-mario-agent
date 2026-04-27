"""
Utilidades para guardado y carga de checkpoints
Permite reanudar entrenamiento desde puntos de control
"""
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
from datetime import datetime


class CheckpointManager:
    """Gestor de checkpoints para entrenamiento"""

    def __init__(self, checkpoint_dir: Path):
        """
        Inicializa el gestor de checkpoints

        Args:
            checkpoint_dir: Directorio para guardar checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        training_state: Dict[str, Any],
        metrics: Dict[str, float] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Guarda un checkpoint

        Args:
            step: Paso de entrenamiento
            model_state: Estado del modelo (model.state_dict())
            optimizer_state: Estado del optimizador (optimizer.state_dict())
            training_state: Estado del entrenamiento (dicts con info)
            metrics: Métricas actuales
            is_best: Si es el mejor modelo hasta ahora

        Returns:
            Path al checkpoint guardado
        """
        checkpoint = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "training_state": training_state,
            "metrics": metrics or {},
        }

        # Guardar checkpoint regular
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step:010d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Guardar como best si es necesario
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        # Guardar siempre un último checkpoint
        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Tuple[Dict, Dict, Dict, Dict, int]:
        """
        Carga un checkpoint

        Args:
            checkpoint_path: Path al checkpoint

        Returns:
            Tupla (model_state, optimizer_state, training_state, metrics, step)
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        return (
            checkpoint["model_state"],
            checkpoint["optimizer_state"],
            checkpoint["training_state"],
            checkpoint.get("metrics", {}),
            checkpoint["step"],
        )

    def load_latest_checkpoint(self) -> Optional[Tuple[Dict, Dict, Dict, Dict, int]]:
        """
        Carga el último checkpoint

        Returns:
            Tupla (model_state, optimizer_state, training_state, metrics, step)
            o None si no hay checkpoint
        """
        last_path = self.checkpoint_dir / "last_checkpoint.pt"

        if not last_path.exists():
            return None

        return self.load_checkpoint(last_path)

    def load_best_checkpoint(self) -> Optional[Tuple[Dict, Dict, Dict, Dict, int]]:
        """
        Carga el mejor checkpoint

        Returns:
            Tupla (model_state, optimizer_state, training_state, metrics, step)
            o None si no hay checkpoint
        """
        best_path = self.checkpoint_dir / "best_model.pt"

        if not best_path.exists():
            return None

        return self.load_checkpoint(best_path)

    def list_checkpoints(self) -> list:
        """
        Lista todos los checkpoints disponibles

        Returns:
            Lista de paths a checkpoints ordenados por step
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1])
        )
        return checkpoints

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Limpia checkpoints antiguos, manteniendo los últimos N

        Args:
            keep_last_n: Número de checkpoints a mantener
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
                print(f"Checkpoint antiguo eliminado: {checkpoint.name}")

    def save_training_metadata(self, metadata: Dict[str, Any]):
        """
        Guarda metadata del entrenamiento

        Args:
            metadata: Diccionario con metadata
        """
        metadata_path = self.checkpoint_dir / "training_metadata.json"

        # Serializable conversion
        serializable = {}
        for k, v in metadata.items():
            if isinstance(v, (int, float, str, bool, list, dict)):
                serializable[k] = v
            else:
                serializable[k] = str(v)

        with open(metadata_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load_training_metadata(self) -> Dict[str, Any]:
        """
        Carga metadata del entrenamiento

        Returns:
            Diccionario con metadata
        """
        metadata_path = self.checkpoint_dir / "training_metadata.json"

        if not metadata_path.exists():
            return {}

        with open(metadata_path, "r") as f:
            return json.load(f)


def create_checkpoint_manager(checkpoint_dir: Path) -> CheckpointManager:
    """
    Factory function para crear un gestor de checkpoints

    Args:
        checkpoint_dir: Directorio de checkpoints

    Returns:
        CheckpointManager
    """
    return CheckpointManager(checkpoint_dir)
