"""
Visualizacion de un agente Dueling DQN jugando Super Mario Bros.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from pathlib import Path
from typing import Optional
import time

import numpy as np
import torch

from config.config import get_project_paths
from config.hyperparameters import (
    CNN_FEATURES,
    CNN_USE_BATCH_NORM,
    DEVICE_ID,
    DQN_DUELING_HIDDEN_DIM,
    DQN_HIDDEN_DIM,
    FRAME_SKIP,
    FRAME_STACK,
    FRAME_WIDTH,
    NOISY_STD_INIT,
    USE_LAYER_NORM,
    USE_NOISY_NETS,
)
from utils.device import DeviceManager
from utils.video_recorder import VideoRecorder
from models.dueling_dqn import create_dueling_dqn


def _prepare_obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    obs_tensor = torch.from_numpy(np.asarray(obs)).to(device=device, dtype=torch.float32).unsqueeze(0)
    if obs_tensor.ndim == 4 and obs_tensor.shape[-1] in (1, 3, 4):
        obs_tensor = obs_tensor.permute(0, 3, 1, 2).contiguous()
    if obs_tensor.max() > 1.0:
        obs_tensor = obs_tensor / 255.0
    return obs_tensor


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
    """Ejecuta episodios con la politica greedy del Q-network."""

    print("=" * 80)
    print("VISUALIZACION DEL AGENTE DUELING DQN")
    print("=" * 80 + "\n")

    if device is None:
        device_manager = DeviceManager(use_gpu=True, device_id=DEVICE_ID)
        device = device_manager.get_device()

    project_paths = get_project_paths()
    env_render_mode = "human" if render_mode == "human" else "rgb_array"

    print(f"Creando entorno: SuperMarioBros-{world}-{stage}...")
    from env.mario_env import create_environment

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
    except Exception as exc:
        print(f"Error creando entorno con render_mode '{env_render_mode}': {exc}")
        print("Usando modo sin renderizado grafico...")
        env = create_environment(
            world=world,
            stage=stage,
            num_envs=1,
            parallel=False,
            frame_size=FRAME_WIDTH,
            num_frames=FRAME_STACK,
            skip=FRAME_SKIP,
            use_reward_shaping=False,
            render_mode=None,
        )
        render_mode = "none"

    print("Cargando Q-network...")
    model = create_dueling_dqn(
        env.observation_space,
        env.action_space,
        cnn_features=CNN_FEATURES,
        hidden_dim=DQN_HIDDEN_DIM,
        dueling_hidden_dim=DQN_DUELING_HIDDEN_DIM,
        use_noisy_nets=USE_NOISY_NETS,
        noisy_std_init=NOISY_STD_INIT,
        use_batch_norm=CNN_USE_BATCH_NORM,
        use_layer_norm=USE_LAYER_NORM,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state", checkpoint.get("model"))
    if model_state is None:
        raise KeyError("Checkpoint sin 'model_state' ni 'model'")
    model.load_state_dict(model_state)
    model.eval()

    print(f"Modelo cargado desde: {checkpoint_path}")
    print(f"Parametros: {sum(p.numel() for p in model.parameters()):,}\n")

    video_recorder = VideoRecorder(project_paths.VIDEOS_DIR, fps=fps) if save_video else None
    episode_rewards = []

    print(f"Ejecutando {num_episodes} episodios...\n")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        max_x_pos = 0
        done = False
        start_time = time.time()

        if video_recorder is not None:
            video_recorder.start_recording(episode)

        print(f"Episodio {episode + 1}/{num_episodes}")
        print("  ", end="", flush=True)

        while not done:
            with torch.no_grad():
                q_values = model(_prepare_obs_tensor(obs, device))
                action = int(q_values.argmax(dim=1).item())

            obs, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            max_x_pos = max(max_x_pos, info.get("x_pos", 0))

            if render_mode == "rgb_array" and video_recorder is not None:
                try:
                    frame = env.render()
                    if frame is not None:
                        video_recorder.add_frame(frame)
                except Exception:
                    pass
            elif render_mode == "human":
                try:
                    env.render()
                except Exception:
                    pass

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

        print(f"\n  Duracion: {elapsed:.1f}s")
        print(f"  Pasos: {episode_length}")
        print(f"  Recompensa: {episode_reward:.2f}")
        print(f"  Progreso (x_pos): {max_x_pos:.1f}\n")

    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)
    if episode_rewards:
        print(f"Recompensa Promedio: {np.mean(episode_rewards):.2f}")
        print(f"Recompensa Min/Max: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")

    env.close()
    print("\nVisualizacion completada.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualiza un agente Dueling DQN jugando Super Mario Bros")
    parser.add_argument("checkpoint", type=str, help="Path al checkpoint del modelo")
    parser.add_argument("--world", type=int, default=1, help="Mundo (1-8)")
    parser.add_argument("--stage", type=int, default=1, help="Etapa (1-4)")
    parser.add_argument("--episodes", type=int, default=1, help="Numero de episodios")
    parser.add_argument(
        "--mode",
        type=str,
        default="rgb_array",
        choices=["human", "rgb_array", "none"],
        help="Modo de renderizado",
    )
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
