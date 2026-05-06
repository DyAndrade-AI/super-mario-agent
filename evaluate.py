"""
Script de evaluación del agente entrenado
Evalúa sin entrenar y genera estadísticas
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import numpy as np
from pathlib import Path
from typing import Optional
import time

from config.config import get_project_paths
from config.hyperparameters import DEVICE_ID, FRAME_WIDTH, FRAME_STACK, FRAME_SKIP
from utils.device import DeviceManager
from utils.video_recorder import VideoRecorder
from env.mario_env import create_environment
from agents.ppo_agent import create_ppo_agent
from models.actor_critic import create_actor_critic


def evaluate(
    checkpoint_path: Path,
    world: int = 1,
    stage: int = 1,
    num_episodes: int = 10,
    render: bool = True,
    save_videos: bool = True,
    device: Optional[torch.device] = None,
):
    """
    Evalúa un modelo entrenado

    Args:
        checkpoint_path: Path al checkpoint
        world: Mundo de Super Mario
        stage: Etapa del mundo
        num_episodes: Número de episodios
        render: Renderizar
        save_videos: Guardar videos
        device: Dispositivo
    """

    print("="*80)
    print("EVALUACIÓN DEL AGENTE PPO")
    print("="*80 + "\n")

    # Device
    if device is None:
        device_manager = DeviceManager(use_gpu=True, device_id=DEVICE_ID)
        device = device_manager.get_device()

    # Paths
    project_paths = get_project_paths()

    # Crear entorno
    print(f"Creando entorno: SuperMarioBros-{world}-{stage}...")
    eval_env = create_environment(
        world=world,
        stage=stage,
        num_envs=1,
        parallel=False,
        frame_size=FRAME_WIDTH,
        num_frames=FRAME_STACK,
        skip=FRAME_SKIP,
        use_reward_shaping=False,
        render_mode="rgb_array" if render else None,
    )

    observation_space = eval_env.observation_space
    action_space = eval_env.action_space

    # Crear modelo
    print("Cargando modelo...")
    model = create_actor_critic(
        observation_space,
        action_space,
        cnn_features=[32, 64, 64],
        lstm_hidden_dim=512,
    ).to(device)

    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state", checkpoint.get("model"))
    if model_state is None:
        raise KeyError("Checkpoint sin 'model_state' ni 'model'")
    model.load_state_dict(model_state)
    model.eval()

    print(f"Modelo cargado desde: {checkpoint_path}")

    # Video recorder
    if save_videos:
        video_recorder = VideoRecorder(project_paths.VIDEOS_DIR)
    else:
        video_recorder = None

    # Evaluación
    episode_rewards = []
    episode_lengths = []
    episode_progresses = []

    print(f"\nEvaluando en {num_episodes} episodios...\n")

    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        max_x_pos = 0
        done = False
        lstm_hidden = None

        if video_recorder is not None:
            video_recorder.start_recording(episode)

        while not done:
            # Obtener acción
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)
                if obs_tensor.ndim == 4 and obs_tensor.shape[-1] in [1, 3, 4]:
                    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                action, lstm_hidden = model.get_action_deterministic(obs_tensor, lstm_hidden)
                action = action.cpu().item()

            obs, reward, done, info = eval_env.step(action)

            episode_reward += reward
            episode_length += 1
            max_x_pos = max(max_x_pos, info.get("x_pos", 0))

            # Renderizar
            if render:
                frame = eval_env.render()
                if video_recorder is not None and frame is not None:
                    video_recorder.add_frame(frame)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_progresses.append(max_x_pos)

        if video_recorder is not None:
            video_recorder.stop_recording(
                reward=episode_reward,
                episode_length=episode_length,
            )

        print(f"Episodio {episode+1:2d}: Reward={episode_reward:7.2f}, Length={episode_length:5d}, Progress={max_x_pos:5.1f}")

    # Estadísticas
    print("\n" + "="*80)
    print("ESTADÍSTICAS DE EVALUACIÓN")
    print("="*80)
    print(f"Recompensa Media: {np.mean(episode_rewards):.2f} (±{np.std(episode_rewards):.2f})")
    print(f"Recompensa Min/Max: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print(f"Duración Media: {np.mean(episode_lengths):.1f} (±{np.std(episode_lengths):.1f})")
    print(f"Progreso Medio (x_pos): {np.mean(episode_progresses):.1f} (±{np.std(episode_progresses):.1f})")

    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evalúa un agente PPO entrenado")
    parser.add_argument("checkpoint", type=str, help="Path al checkpoint del modelo")
    parser.add_argument("--world", type=int, default=1, help="Mundo de Super Mario")
    parser.add_argument("--stage", type=int, default=1, help="Etapa del mundo")
    parser.add_argument("--episodes", type=int, default=10, help="Número de episodios")
    parser.add_argument("--no-render", action="store_true", help="No renderizar")
    parser.add_argument("--no-videos", action="store_true", help="No guardar videos")

    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint),
        world=args.world,
        stage=args.stage,
        num_episodes=args.episodes,
        render=not args.no_render,
        save_videos=not args.no_videos,
    )
