"""
Script de entrenamiento principal para el agente PPO
Orquesta la recolección de experiencias, actualización del modelo y evaluación
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import numpy as np
from pathlib import Path
from typing import Optional
import time
from datetime import datetime

from config.hyperparameters import *
from config.config import get_project_paths, create_training_config
from utils.device import DeviceManager
from utils.logger import MetricsLogger
from utils.checkpoint import CheckpointManager
from utils.metrics import MetricsTracker
from utils.video_recorder import VideoRecorder
from env.mario_env import create_environment
from agents.ppo_agent import create_ppo_agent


def train(
    world: int = WORLD,
    stage: int = STAGE,
    total_timesteps: int = TOTAL_TIMESTEPS,
    checkpoint_dir: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
    use_wandb: bool = USE_WANDB,
    use_tensorboard: bool = USE_TENSORBOARD,
    seed: int = SEED,
    render_eval: bool = False,
):
    """
    Entrena el agente PPO en Super Mario Bros

    Args:
        world: Mundo de Super Mario
        stage: Etapa del mundo
        total_timesteps: Pasos totales de entrenamiento
        checkpoint_dir: Directorio para checkpoints
        resume_from_checkpoint: Cargar desde checkpoint previo
        use_wandb: Usar Weights & Biases
        use_tensorboard: Usar TensorBoard
        seed: Random seed
        render_eval: Renderizar durante evaluación
    """

    # ========== SETUP ==========
    print("\n" + "="*80)
    print("ENTRENAMIENTO DE AGENTE PPO PARA SUPER MARIO BROS")
    print("="*80 + "\n")

    # Paths
    project_paths = get_project_paths()
    if checkpoint_dir is None:
        checkpoint_dir = project_paths.CHECKPOINTS_DIR

    # Device
    device_manager = DeviceManager(use_gpu=USE_GPU, device_id=DEVICE_ID)
    device = device_manager.get_device()

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Logger
    experiment_name = f"mario_world{world}_stage{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = MetricsLogger(
        log_dir=project_paths.LOGS_DIR,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
    )

    # Log configuración
    config = {
        "world": world,
        "stage": stage,
        "total_timesteps": total_timesteps,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": CLIP_RANGE,
        "entropy_coeff": ENTROPY_COEFF,
        "batch_size": BATCH_SIZE,
        "num_envs": NUM_ENVS_PARALLEL,
        "rollout_steps": ROLLOUT_STEPS,
        "epochs_per_update": EPOCHS_PER_UPDATE,
        "use_lstm": USE_LSTM,
        "use_mixed_precision": USE_MIXED_PRECISION,
        "device": str(device),
    }
    logger.log_config(config)

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(Path(checkpoint_dir))

    # Metrics tracker
    metrics_tracker = MetricsTracker(window_size=100)

    # Video recorder
    video_recorder = VideoRecorder(project_paths.VIDEOS_DIR)

    print(f"Experimento: {experiment_name}")
    print(f"Logs guardados en: {project_paths.LOGS_DIR}")
    print(f"Checkpoints guardados en: {checkpoint_dir}")

    # ========== CREAR ENTORNO ==========
    print(f"\nCreando entorno: SuperMarioBros-{world}-{stage}...")

    env = create_environment(
        world=world,
        stage=stage,
        num_envs=NUM_ENVS_PARALLEL,
        parallel=NUM_ENVS_PARALLEL > 1,
        frame_size=FRAME_WIDTH,
        num_frames=FRAME_STACK,
        skip=FRAME_SKIP,
        use_reward_shaping=True,
    )

    observation_space = env.observation_space
    action_space = env.action_space

    print(f"Observation space: {observation_space}")
    print(f"Action space: {action_space}")

    # ========== CREAR AGENTE ==========
    print("\nCreando agente PPO...")

    agent = create_ppo_agent(
        observation_space=observation_space,
        action_space=action_space,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        clip_range_vf=CLIP_RANGE_VF,
        entropy_coeff=ENTROPY_COEFF,
        value_loss_coeff=VALUE_LOSS_COEFF,
        batch_size=BATCH_SIZE,
        num_envs=NUM_ENVS_PARALLEL,
        rollout_steps=ROLLOUT_STEPS,
        epochs_per_update=EPOCHS_PER_UPDATE,
        normalize_advantage=NORMALIZE_ADVANTAGE,
        normalize_reward=NORMALIZE_REWARD,
        clip_grad_norm=CLIP_GRAD_NORM,
        use_mixed_precision=USE_MIXED_PRECISION,
        device=device,
        cnn_features=CNN_FEATURES,
        lstm_hidden_dim=LSTM_HIDDEN_DIM if USE_LSTM else 512,
        policy_hidden_dim=POLICY_HIDDEN_DIM,
        value_hidden_dim=VALUE_HIDDEN_DIM,
        use_lstm=USE_LSTM,
        use_batch_norm=CNN_USE_BATCH_NORM,
        use_layer_norm=USE_LAYER_NORM,
    )

    print(f"Modelo creado:")
    print(f"  Parámetros totales: {sum(p.numel() for p in agent.model.parameters()):,}")
    print(f"  Parámetros entrenables: {sum(p.numel() for p in agent.model.parameters() if p.requires_grad):,}")

    # ========== REANUDAR DESDE CHECKPOINT ==========
    start_step = 0
    if resume_from_checkpoint is not None and Path(resume_from_checkpoint).exists():
        print(f"\nReanudando desde checkpoint: {resume_from_checkpoint}")
        agent.load(Path(resume_from_checkpoint))
        start_step = agent.total_steps
        print(f"Reanudando desde paso: {start_step}")

    # ========== SETUP LEARNING RATE SCHEDULER ==========
    if LEARNING_RATE_DECAY:
        def lr_lambda(step):
            progress = min(step / LEARNING_RATE_DECAY_STEPS, 1.0)
            return 1.0 - progress * (1.0 - LR_FINAL_FACTOR)

        from torch.optim.lr_scheduler import LambdaLR
        agent.scheduler = LambdaLR(agent.optimizer, lr_lambda)

    # ========== LOOP DE ENTRENAMIENTO ==========
    print("\n" + "="*80)
    print("INICIANDO ENTRENAMIENTO")
    print("="*80 + "\n")

    training_start_time = time.time()
    best_eval_reward = -float("inf")
    best_model_score = -float("inf")

    best_checkpoint = checkpoint_manager.load_best_checkpoint()
    if best_checkpoint is not None:
        best_model_state, _, _, best_metrics, _ = best_checkpoint
        current_model_state = agent.model.state_dict()
        compatible_best = best_model_state is not None and all(
            key in current_model_state and current_model_state[key].shape == value.shape
            for key, value in best_model_state.items()
        )
        if compatible_best:
            best_model_score = float(
                best_metrics.get(
                    "best_score",
                    best_metrics.get(
                        "eval_reward",
                        best_metrics.get("mean_episode_reward", -float("inf")),
                    ),
                )
            )
            best_eval_reward = float(best_metrics.get("eval_reward", best_eval_reward))
        else:
            print("best_model.pt existente incompatible con el modelo actual; se reemplazara al mejorar.")

    def next_interval_after(step: int, interval: int) -> int:
        return ((step // interval) + 1) * interval

    next_checkpoint_step = next_interval_after(agent.total_steps, CHECKPOINT_FREQ)
    next_eval_step = next_interval_after(agent.total_steps, EVAL_FREQ)
    next_video_step = next_interval_after(agent.total_steps, SAVE_VIDEO_FREQ)

    try:
        while agent.total_steps < total_timesteps:
            # Recolectar rollout
            print(f"\nRecolectando rollout... (Steps: {agent.total_steps:,} / {total_timesteps:,})")
            rollout_start = time.time()

            avg_reward, _ = agent.collect_rollout(env, progress_interval=LOG_FREQ)

            rollout_time = time.time() - rollout_start
            rollout_transitions = agent.rollout_steps * agent.num_envs
            rollout_steps_per_sec = rollout_transitions / max(rollout_time, 1e-8)
            print(f"Rollout completado en {rollout_time:.2f}s")

            # Actualizar modelo
            print("Actualizando modelo...")
            update_start = time.time()

            train_metrics = agent.update()

            update_time = time.time() - update_start
            print(f"Update completado en {update_time:.2f}s")

            # Añadir métricas
            metrics_tracker.add_training_metric(
                value_loss=train_metrics["value_loss"],
                policy_loss=train_metrics["policy_loss"],
                entropy=train_metrics["entropy"],
                learning_rate=agent.optimizer.param_groups[0]["lr"],
            )
            for episode_reward, episode_length in zip(
                agent.last_rollout_episode_returns,
                agent.last_rollout_episode_lengths,
            ):
                metrics_tracker.add_episode_metric(episode_reward, episode_length)

            # Log
            mean_metrics = metrics_tracker.get_mean_metrics()
            mean_metrics.update(train_metrics)
            mean_metrics["steps"] = agent.total_steps
            mean_metrics["rollout_mean_reward"] = agent.last_rollout_mean_reward
            mean_metrics["rollout_total_reward"] = agent.last_rollout_total_reward
            mean_metrics["rollout_completed_episodes"] = agent.last_rollout_completed_episodes
            mean_metrics["rollout_steps_per_sec"] = rollout_steps_per_sec

            logger.log_metrics(mean_metrics, step=agent.total_steps)

            # Print stats
            print(f"\nEstadísticas:")
            for key, value in mean_metrics.items():
                if isinstance(value, (int, np.integer)):
                    print(f"  {key}: {value}")
                elif isinstance(value, (float, np.floating)):
                    print(f"  {key}: {value:.6f}")

            if agent.last_rollout_completed_episodes > 0:
                current_score = float(mean_metrics["mean_episode_reward"])
                score_source = "mean_episode_reward"
            else:
                current_score = float(agent.last_rollout_mean_reward)
                score_source = "rollout_mean_reward"

            if current_score > best_model_score:
                best_model_score = current_score
                best_metrics = dict(mean_metrics)
                best_metrics["best_score"] = best_model_score
                best_metrics["best_score_source"] = score_source
                best_path = checkpoint_manager.save_best_model(
                    step=agent.total_steps,
                    model_state=agent.model.state_dict(),
                    optimizer_state=agent.optimizer.state_dict(),
                    training_state={
                        "total_steps": agent.total_steps,
                        "update_count": agent.update_count,
                        "best_score": best_model_score,
                        "best_score_source": score_source,
                    },
                    metrics=best_metrics,
                    filename="best_model.pt",
                )
                print(
                    f"Nuevo mejor modelo registrado ({score_source}: "
                    f"{best_model_score:.6f}). Reemplazado: {best_path}"
                )

            # Update LR scheduler
            if agent.scheduler is not None:
                agent.scheduler.step()

            # Guardarprogreso periódicamente
            if agent.total_steps >= next_checkpoint_step:
                print(f"\nGuardando checkpoint en step {agent.total_steps}...")
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    step=agent.total_steps,
                    model_state=agent.model.state_dict(),
                    optimizer_state=agent.optimizer.state_dict(),
                    training_state={
                        "total_steps": agent.total_steps,
                        "update_count": agent.update_count,
                    },
                    metrics=mean_metrics,
                    is_best=False,
                )
                print(f"Checkpoint guardado: {checkpoint_path}")
                checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)
                while next_checkpoint_step <= agent.total_steps:
                    next_checkpoint_step += CHECKPOINT_FREQ

            # Evaluación
            if agent.total_steps >= next_eval_step:
                print(f"\nEvaluando en step {agent.total_steps}...")
                save_video_this_eval = agent.total_steps >= next_video_step
                eval_reward = evaluate(
                    agent=agent,
                    world=world,
                    stage=stage,
                    num_episodes=EVAL_EPISODES,
                    video_recorder=video_recorder if save_video_this_eval else None,
                    device=device,
                    render=render_eval,
                )

                logger.log_scalar("eval/reward", eval_reward, step=agent.total_steps)
                print(f"Reward de evaluación: {eval_reward:.2f}")

                # Guardar mejor modelo
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    print(f"Nuevo mejor modelo por evaluaciÃ³n! Reward: {eval_reward:.2f}")
                    eval_metrics = dict(mean_metrics)
                    eval_metrics["eval_reward"] = eval_reward
                    eval_metrics["best_score"] = eval_reward
                    eval_metrics["best_score_source"] = "eval_reward"
                    checkpoint_manager.save_best_model(
                        step=agent.total_steps,
                        model_state=agent.model.state_dict(),
                        optimizer_state=agent.optimizer.state_dict(),
                        training_state={
                            "total_steps": agent.total_steps,
                            "update_count": agent.update_count,
                            "best_score": eval_reward,
                            "best_score_source": "eval_reward",
                        },
                        metrics=eval_metrics,
                        filename="best_eval_model.pt",
                    )
                    if eval_reward > best_model_score:
                        best_model_score = eval_reward
                        best_path = checkpoint_manager.save_best_model(
                            step=agent.total_steps,
                            model_state=agent.model.state_dict(),
                            optimizer_state=agent.optimizer.state_dict(),
                            training_state={
                                "total_steps": agent.total_steps,
                                "update_count": agent.update_count,
                                "best_score": eval_reward,
                                "best_score_source": "eval_reward",
                            },
                            metrics=eval_metrics,
                            filename="best_model.pt",
                        )
                        print(f"Mejor modelo global reemplazado: {best_path}")

                while next_eval_step <= agent.total_steps:
                    next_eval_step += EVAL_FREQ
                if save_video_this_eval:
                    while next_video_step <= agent.total_steps:
                        next_video_step += SAVE_VIDEO_FREQ

    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")

    # ========== CLEANUP ==========
    elapsed_time = time.time() - training_start_time
    print(f"\nTiempo total de entrenamiento: {elapsed_time / 3600:.2f} horas")

    env.close()
    logger.save_metrics()
    logger.close()

    print("\nEntrenamiento completado.")
    print(f"Checkpoints guardados en: {checkpoint_dir}")
    print(f"Logs guardados en: {project_paths.LOGS_DIR}")


def evaluate(
    agent,
    world: int = 1,
    stage: int = 1,
    num_episodes: int = 5,
    video_recorder: Optional[VideoRecorder] = None,
    device: torch.device = None,
    render: bool = False,
) -> float:
    """
    Evalúa el agente

    Args:
        agent: Agente a evaluar
        world: Mundo
        stage: Etapa
        num_episodes: Número de episodios
        video_recorder: Grabador de video (opcional)
        device: Dispositivo
        render: Renderizar

    Returns:
        Recompensa promedio
    """
    if device is None:
        device = torch.device("cpu")

    # Crear entorno de evaluación
    eval_env = create_environment(
        world=world,
        stage=stage,
        num_envs=1,
        parallel=False,
        render_mode="rgb_array" if render else None,
    )

    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0.0
        done = False
        lstm_hidden = None

        if video_recorder is not None:
            video_recorder.start_recording(episode)

        while not done:
            # Obtener acción determinística
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)
                if obs_tensor.ndim == 4 and obs_tensor.shape[-1] in [1, 3, 4]:
                    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                action, lstm_hidden = agent.model.get_action_deterministic(obs_tensor, lstm_hidden)
                action = action.cpu().item()

            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward

            if video_recorder is not None and hasattr(eval_env, 'render'):
                frame = eval_env.render()
                if frame is not None:
                    video_recorder.add_frame(frame)

        episode_rewards.append(episode_reward)

        if video_recorder is not None:
            video_recorder.stop_recording(
                reward=episode_reward,
                episode_length=info.get("episode_length", 0),
            )

    eval_env.close()

    avg_reward = float(np.mean(episode_rewards))
    return avg_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrena el agente PPO en Super Mario Bros")
    parser.add_argument("--world", type=int, default=WORLD, help="Mundo de Super Mario")
    parser.add_argument("--stage", type=int, default=STAGE, help="Etapa del mundo")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS, help="Pasos totales")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directorio de checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Reanudar desde checkpoint")
    parser.add_argument("--use-wandb", action="store_true", help="Usar Weights & Biases")
    parser.add_argument("--no-tensorboard", action="store_true", help="Desabilitar TensorBoard")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")

    args = parser.parse_args()

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
