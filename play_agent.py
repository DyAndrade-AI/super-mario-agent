"""
Script para ver al agente jugando Super Mario Bros en tiempo real
Visualización interactiva del modelo entrenado
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
from models.actor_critic import create_actor_critic


def play(
    checkpoint_path: Path,
    world: int = 1,
    stage: int = 1,
    num_episodes: int = 1,
    render_mode: str = "human",
    save_video: bool = False,
    fps: int = 30,
    device: Optional[torch.device] = None,
):
    """
    Ejecuta episodios del agente entrenado en tiempo real

    Args:
        checkpoint_path: Path al checkpoint
        world: Mundo
        stage: Etapa
        num_episodes: Número de episodios
        render_mode: Modo de renderizado ("human", "rgb_array", etc.)
        save_video: Guardar video
        fps: Frames por segundo
        device: Dispositivo
    """

    print("="*80)
    print("VISUALIZACIÓN DEL AGENTE EN SUPER MARIO BROS")
    print("="*80 + "\n")

    # Device
    if device is None:
        device_manager = DeviceManager(use_gpu=True, device_id=DEVICE_ID)
        device = device_manager.get_device()

    # Paths
    project_paths = get_project_paths()

    # Crear entorno
    print(f"Creando entorno: SuperMarioBros-{world}-{stage}...")

    # Determinar render mode
    if render_mode == "human":
        # Nota: requiere gymrender u otro backend gráfico
        env_render_mode = "human"
    else:
        env_render_mode = "rgb_array"

    try:
        env = create_environment(
            world=world,
            stage=stage,
            num_envs=1,
            parallel=False,
            frame_size=FRAME_WIDTH,
            num_frames=FRAME_STACK,
            skip=FRAME_SKIP,
            use_reward_shaping=False,
            render_mode=env_render_mode,
        )
    except Exception as e:
        print(f"Error creando entorno con render_mode '{env_render_mode}': {e}")
        print("Usando modo sin renderizado gráfico...")
        env = create_environment(
            world=world,
            stage=stage,
            num_envs=1,
            parallel=False,
            render_mode=None,
        )
        render_mode = "none"

    observation_space = env.observation_space
    action_space = env.action_space

    # Crear modelo
    print("Cargando modelo...")
    model = create_actor_critic(
        observation_space,
        action_space,
        cnn_features=[32, 64, 64],
        lstm_hidden_dim=512,
    ).to(device)
    model.eval()

    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state", checkpoint.get("model"))
    if model_state is None:
        raise KeyError("Checkpoint sin 'model_state' ni 'model'")
    model.load_state_dict(model_state)

    print(f"Modelo cargado desde: {checkpoint_path}")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}\n")

    # Video recorder
    if save_video:
        video_recorder = VideoRecorder(project_paths.VIDEOS_DIR, fps=fps)
    else:
        video_recorder = None

    # Jugar
    episode_rewards = []

    print(f"Ejecutando {num_episodes} episodios...\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        max_x_pos = 0
        done = False
        lstm_hidden = None
        start_time = time.time()

        if video_recorder is not None:
            video_recorder.start_recording(episode)

        print(f"Episodio {episode + 1}/{num_episodes}")
        print("  ", end="", flush=True)

        while not done:
            # Obtener acción
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)
                if obs_tensor.ndim == 4 and obs_tensor.shape[-1] in [1, 3, 4]:
                    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                action, lstm_hidden = model.get_action_deterministic(obs_tensor, lstm_hidden)
                action = action.cpu().item()

            obs, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            max_x_pos = max(max_x_pos, info.get("x_pos", 0))

            # Renderizar
            if render_mode == "rgb_array" and video_recorder is not None:
                try:
                    frame = env.render()
                    if frame is not None:
                        video_recorder.add_frame(frame)
                except:
                    pass
            elif render_mode == "human":
                try:
                    env.render()
                except:
                    pass

            # Mostrar progreso
            if episode_length % 100 == 0:
                print(".", end="", flush=True)

        elapsed = time.time() - start_time

        episode_rewards.append(episode_reward)

        if video_recorder is not None:
            video_path = video_recorder.stop_recording(
                reward=episode_reward,
                episode_length=episode_length,
            )
            if video_path:
                print(f"\n  Video guardado: {video_path}")

        print(f"\n  Duración: {elapsed:.1f}s")
        print(f"  Pasos: {episode_length}")
        print(f"  Recompensa: {episode_reward:.2f}")
        print(f"  Progreso (x_pos): {max_x_pos:.1f}\n")

    # Resumen
    print("="*80)
    print("RESUMEN")
    print("="*80)
    if episode_rewards:
        print(f"Recompensa Promedio: {np.mean(episode_rewards):.2f}")
        print(f"Recompensa Min/Max: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")

    env.close()
    print("\nVisualización completada.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualiza un agente PPO jugando Super Mario Bros")
    parser.add_argument("checkpoint", type=str, help="Path al checkpoint del modelo")
    parser.add_argument("--world", type=int, default=1, help="Mundo (1-8)")
    parser.add_argument("--stage", type=int, default=1, help="Etapa (1-4)")
    parser.add_argument("--episodes", type=int, default=1, help="Número de episodios")
    parser.add_argument("--mode", type=str, default="rgb_array", choices=["human", "rgb_array", "none"],
                        help="Modo de renderizado")
    parser.add_argument("--save-video", action="store_true", help="Guardar video")
    parser.add_argument("--fps", type=int, default=30, help="FPS para video")

    args = parser.parse_args()

    play(
        checkpoint_path=Path(args.checkpoint),
        world=args.world,
        stage=args.stage,
        num_episodes=args.episodes,
        render_mode=args.mode,
        save_video=args.save_video,
        fps=args.fps,
    )
