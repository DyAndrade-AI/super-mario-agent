"""
Entry point principal del proyecto
Orquesta el training, evaluación y visualización
"""
import argparse
import sys
from pathlib import Path

from train import train, evaluate
from play_agent import play


def main():
    """
    Función principal con interfaz por línea de comandos
    """
    parser = argparse.ArgumentParser(
        description="Sistema completo de entrenamiento para agente PPO en Super Mario Bros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenar desde cero
  python main.py train --world 1 --stage 1 --total-timesteps 50000000

  # Reanudar entrenamiento
  python main.py train --resume outputs/checkpoints/best_model.pt

  # Evaluar modelo
  python main.py evaluate outputs/checkpoints/best_model.pt --episodes 10

  # Ver al agente jugando
  python main.py play outputs/checkpoints/best_model.pt --episodes 3 --save-video
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # Comando: train
    train_parser = subparsers.add_parser("train", help="Entrenar el agente")
    train_parser.add_argument("--world", type=int, default=1, help="Mundo de Super Mario (1-8)")
    train_parser.add_argument("--stage", type=int, default=1, help="Etapa del mundo (1-4)")
    train_parser.add_argument("--total-timesteps", type=int, default=50_000_000,
                              help="Pasos totales de entrenamiento")
    train_parser.add_argument("--checkpoint-dir", type=str, default=None,
                              help="Directorio para guardar checkpoints")
    train_parser.add_argument("--resume", type=str, default=None,
                              help="Reanudar desde checkpoint")
    train_parser.add_argument("--use-wandb", action="store_true",
                              help="Usar Weights & Biases para logging")
    train_parser.add_argument("--no-tensorboard", action="store_true",
                              help="Desabilitar TensorBoard")
    train_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed")

    # Comando: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluar el agente")
    eval_parser.add_argument("checkpoint", type=str, help="Path al checkpoint del modelo")
    eval_parser.add_argument("--world", type=int, default=1, help="Mundo")
    eval_parser.add_argument("--stage", type=int, default=1, help="Etapa")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Número de episodios")
    eval_parser.add_argument("--no-render", action="store_true", help="No renderizar")
    eval_parser.add_argument("--no-videos", action="store_true", help="No guardar videos")

    # Comando: play
    play_parser = subparsers.add_parser("play", help="Ver al agente jugando")
    play_parser.add_argument("checkpoint", type=str, help="Path al checkpoint del modelo")
    play_parser.add_argument("--world", type=int, default=1, help="Mundo")
    play_parser.add_argument("--stage", type=int, default=1, help="Etapa")
    play_parser.add_argument("--episodes", type=int, default=1, help="Número de episodios")
    play_parser.add_argument("--mode", type=str, default="rgb_array",
                             choices=["human", "rgb_array", "none"],
                             help="Modo de renderizado")
    play_parser.add_argument("--save-video", action="store_true", help="Guardar video")
    play_parser.add_argument("--fps", type=int, default=30, help="FPS para video")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Ejecutar comando
    if args.command == "train":
        print("\n" + "="*80)
        print("INICIANDO ENTRENAMIENTO DEL AGENTE PPO")
        print("="*80)

        train(
            world=args.world,
            stage=args.stage,
            total_timesteps=args.total_timesteps,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume,
            use_wandb=args.use_wandb,
            use_tensorboard=not args.no_tensorboard,
            seed=args.seed,
        )

    elif args.command == "evaluate":
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint no encontrado: {args.checkpoint}")
            sys.exit(1)

        print("\n" + "="*80)
        print("EVALUANDO AGENTE")
        print("="*80)

        evaluate(
            checkpoint_path=Path(args.checkpoint),
            world=args.world,
            stage=args.stage,
            num_episodes=args.episodes,
            render=not args.no_render,
            save_videos=not args.no_videos,
        )

    elif args.command == "play":
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint no encontrado: {args.checkpoint}")
            sys.exit(1)

        print("\n" + "="*80)
        print("VISUALIZANDO AGENTE EN TIEMPO REAL")
        print("="*80)

        play(
            checkpoint_path=Path(args.checkpoint),
            world=args.world,
            stage=args.stage,
            num_episodes=args.episodes,
            render_mode=args.mode,
            save_video=args.save_video,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
