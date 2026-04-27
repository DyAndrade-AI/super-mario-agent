"""
Script de entrenamiento por etapas usando curriculum learning
Entrena primero en niveles fáciles, luego en niveles difíciles
"""

from train import train
from pathlib import Path

# Configuración del curriculum
CURRICULUM = [
    {"world": 1, "stage": 1, "steps": 10_000_000, "name": "World 1-1"},
    {"world": 1, "stage": 2, "steps": 10_000_000, "name": "World 1-2"},
    {"world": 1, "stage": 3, "steps": 10_000_000, "name": "World 1-3"},
    {"world": 1, "stage": 4, "steps": 20_000_000, "name": "World 1-4"},
]


def train_curriculum():
    """Entrena usando curriculum learning"""

    print("\n" + "="*80)
    print("ENTRENAMIENTO CON CURRICULUM LEARNING")
    print("="*80 + "\n")

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = None

    for i, stage_config in enumerate(CURRICULUM):
        world = stage_config["world"]
        stage = stage_config["stage"]
        steps = stage_config["steps"]
        name = stage_config["name"]

        print(f"\n{'='*80}")
        print(f"ETAPA {i+1}/{len(CURRICULUM)}: {name}")
        print(f"{'='*80}\n")

        # Entrenar esta etapa
        train(
            world=world,
            stage=stage,
            total_timesteps=steps,
            checkpoint_dir=str(checkpoint_dir),
            resume_from_checkpoint=resume_checkpoint,
            use_wandb=False,
            use_tensorboard=True,
            seed=42,
        )

        # Para la siguiente etapa, reanudar desde el último checkpoint
        resume_checkpoint = checkpoint_dir / "last_checkpoint.pt"

        print(f"\nEtapa {name} completada.")


if __name__ == "__main__":
    train_curriculum()
