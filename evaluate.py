"""
Evaluacion de checkpoints Dueling DQN.
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


def evaluate(
    checkpoint_path: Path,
    world: int = 1,
    stage: int = 1,
    num_episodes: int = 10,
    render: bool = True,
    save_videos: bool = True,
    device: Optional[torch.device] = None,
):
    """Evalua un checkpoint Dueling DQN sin exploracion."""

    print("=" * 80)
    print("EVALUACION DEL AGENTE DUELING DQN")
    print("=" * 80 + "\n")

    if device is None:
        device_manager = DeviceManager(use_gpu=True, device_id=DEVICE_ID)
        device = device_manager.get_device()

    project_paths = get_project_paths()

    print(f"Creando entorno: SuperMarioBros-{world}-{stage}...")
    from env.mario_env import create_environment

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

    print("Cargando Q-network...")
    model = create_dueling_dqn(
        eval_env.observation_space,
        eval_env.action_space,
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

    video_recorder = VideoRecorder(project_paths.VIDEOS_DIR) if save_videos else None
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

        if video_recorder is not None:
            video_recorder.start_recording(episode)

        while not done:
            with torch.no_grad():
                q_values = model(_prepare_obs_tensor(obs, device))
                action = int(q_values.argmax(dim=1).item())

            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            max_x_pos = max(max_x_pos, info.get("x_pos", 0))

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

        print(
            f"Episodio {episode + 1:2d}: "
            f"Reward={episode_reward:7.2f}, "
            f"Length={episode_length:5d}, "
            f"Progress={max_x_pos:5.1f}"
        )

    print("\n" + "=" * 80)
    print("ESTADISTICAS DE EVALUACION")
    print("=" * 80)
    print(f"Recompensa Media: {np.mean(episode_rewards):.2f} (+/- {np.std(episode_rewards):.2f})")
    print(f"Recompensa Min/Max: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print(f"Duracion Media: {np.mean(episode_lengths):.1f} (+/- {np.std(episode_lengths):.1f})")
    print(f"Progreso Medio (x_pos): {np.mean(episode_progresses):.1f} (+/- {np.std(episode_progresses):.1f})")

    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evalua un agente Dueling DQN entrenado")
    parser.add_argument("checkpoint", type=str, help="Path al checkpoint del modelo")
    parser.add_argument("--world", type=int, default=1, help="Mundo de Super Mario")
    parser.add_argument("--stage", type=int, default=1, help="Etapa del mundo")
    parser.add_argument("--episodes", type=int, default=10, help="Numero de episodios")
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
