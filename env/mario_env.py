"""
Wrapper principal del entorno de Super Mario Bros
Integra wrappers y reward shaping
"""
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
try:
    import gym
except ImportError:
    import gymnasium as gym

from .wrappers import create_mario_env, create_vectorized_env, InfoWrapper, unpack_step_result
from .reward_shaping import MarioRewardShaper, create_reward_shaper


class MarioEnvironment:
    """Wrapper principal del entorno con gestión completa"""

    def __init__(
        self,
        world: int = 1,
        stage: int = 1,
        frame_size: int = 84,
        num_frames: int = 4,
        skip: int = 4,
        use_reward_shaping: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Inicializa el entorno

        Args:
            world: Mundo de Super Mario
            stage: Etapa del mundo
            frame_size: Tamaño de frames
            num_frames: Frames a apilar
            skip: Frame skip
            use_reward_shaping: Usar reward shaping
            render_mode: Modo de renderizado
        """
        self.world = world
        self.stage = stage
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.skip = skip
        self.use_reward_shaping = use_reward_shaping

        # Crear entorno
        self.env = create_mario_env(
            world=world,
            stage=stage,
            frame_size=frame_size,
            num_frames=num_frames,
            skip=skip,
            apply_wrappers=True,
            render_mode=render_mode,
        )

        # Reward shaper
        if use_reward_shaping:
            self.reward_shaper = create_reward_shaper()
        else:
            self.reward_shaper = None

        # Tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.max_x_pos = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resetea el entorno

        Args:
            seed: Seed para reproducibilidad

        Returns:
            Tupla (observación, info)
        """
        obs, info = self.env.reset(seed=seed)

        # Resetear tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.max_x_pos = 0

        if self.reward_shaper:
            self.reward_shaper.reset()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Ejecuta un paso en el entorno

        Args:
            action: Acción a ejecutar

        Returns:
            Tupla (observación, recompensa, done, info)
        """
        obs, reward, terminated, truncated, info = unpack_step_result(self.env.step(action))
        done = terminated or truncated

        # Aplicar reward shaping
        if self.reward_shaper:
            reward = self.reward_shaper.shape_reward(reward, done, info)

        # Tracking
        self.episode_reward += reward
        self.episode_length += 1

        # Actualizar x_pos máximo
        x_pos = info.get("x_pos", 0)
        if x_pos > self.max_x_pos:
            self.max_x_pos = x_pos

        # Información adicional
        info["episode_reward"] = self.episode_reward
        info["episode_length"] = self.episode_length
        info["max_x_pos"] = self.max_x_pos

        return obs, reward, done, info

    def render(self):
        """Renderiza el entorno"""
        return self.env.render()

    def close(self):
        """Cierra el entorno"""
        self.env.close()

    @property
    def observation_space(self) -> gym.spaces.Space:
        """Retorna el observation space"""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """Retorna el action space"""
        return self.env.action_space

    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Retorna resumen del episodio

        Returns:
            Diccionario con resumen
        """
        summary = {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "max_x_pos": self.max_x_pos,
        }

        if self.reward_shaper:
            summary.update(self.reward_shaper.get_progress_metrics())

        return summary


class ParallelMarioEnvironment:
    """Entorno paralelo para entrenamiento distribuido"""

    def __init__(
        self,
        num_envs: int = 4,
        world: int = 1,
        stage: int = 1,
        frame_size: int = 84,
        num_frames: int = 4,
        skip: int = 4,
        use_reward_shaping: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Inicializa entornos paralelos

        Args:
            num_envs: Número de entornos
            world: Mundo
            stage: Etapa
            frame_size: Tamaño de frames
            num_frames: Frames a apilar
            skip: Frame skip
            use_reward_shaping: Usar reward shaping
            render_mode: Modo de renderizado
        """
        self.num_envs = num_envs
        self.world = world
        self.stage = stage
        self.use_reward_shaping = use_reward_shaping

        # Crear entornos vectorizados
        self.env = create_vectorized_env(
            num_envs=num_envs,
            world=world,
            stage=stage,
            frame_size=frame_size,
            num_frames=num_frames,
            skip=skip,
            render_mode=render_mode,
        )

        # Reward shapers para cada entorno
        if use_reward_shaping:
            self.reward_shapers = [create_reward_shaper() for _ in range(num_envs)]
        else:
            self.reward_shapers = None

        # Tracking
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs)

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Resetea todos los entornos"""
        if isinstance(self.env, list):
            # Fallback: múltiples entornos normales
            obs_list = []
            info_list = []
            for env in self.env:
                obs, info = env.reset()
                obs_list.append(obs)
                info_list.append(info)

            obs = np.stack(obs_list)
            return obs, {"all": info_list}
        else:
            # VectorEnv
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = {}

            return obs, info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Ejecuta un paso en todos los entornos

        Args:
            actions: Acciones para cada entorno

        Returns:
            Tupla (observaciones, recompensas, dones, info)
        """
        if isinstance(self.env, list):
            # Fallback: múltiples entornos normales
            obs_list = []
            rewards_list = []
            dones_list = []
            info_list = []

            for i, (env, action) in enumerate(zip(self.env, actions)):
                obs, reward, terminated, truncated, info = unpack_step_result(env.step(action))
                done = terminated or truncated

                if self.reward_shapers:
                    reward = self.reward_shapers[i].shape_reward(reward, done, info)

                # Tracking
                self.episode_rewards[i] += reward
                self.episode_lengths[i] += 1

                # Reset si terminó
                if done:
                    info = dict(info)
                    info["terminal_observation"] = obs
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        obs = reset_result[0]
                    else:
                        obs = reset_result
                    if self.reward_shapers:
                        self.reward_shapers[i].reset()
                    self.episode_rewards[i] = 0
                    self.episode_lengths[i] = 0

                obs_list.append(obs)
                rewards_list.append(reward)
                dones_list.append(done)
                info_list.append(info)

            obs = np.stack(obs_list)
            rewards = np.array(rewards_list)
            dones = np.array(dones_list)

            return obs, rewards, dones, {"all": info_list}
        else:
            # VectorEnv
            result = self.env.step(actions)

            if len(result) == 5:
                obs, rewards, terminated, truncated, info = result
                dones = terminated | truncated
            else:
                obs, rewards, dones, info = result

            if isinstance(info, dict) and isinstance(info.get("all"), list):
                info_list = info["all"]
            else:
                info_list = [{} for _ in range(self.num_envs)]

            # Reward shaping y tracking
            if self.reward_shapers:
                for i, (reward, done) in enumerate(zip(rewards, dones)):
                    if i < len(self.reward_shapers):
                        rewards[i] = self.reward_shapers[i].shape_reward(
                            float(reward), bool(done), info_list[i] if i < len(info_list) else {}
                        )
                        if done:
                            self.reward_shapers[i].reset()

            for i, (reward, done) in enumerate(zip(rewards, dones)):
                self.episode_rewards[i] += float(reward)
                self.episode_lengths[i] += 1
                if done:
                    self.episode_rewards[i] = 0
                    self.episode_lengths[i] = 0

            return obs, rewards, dones, info

    def close(self):
        """Cierra los entornos"""
        if isinstance(self.env, list):
            for env in self.env:
                env.close()
        else:
            self.env.close()

    @property
    def observation_space(self):
        """Retorna observation space"""
        if isinstance(self.env, list):
            return self.env[0].observation_space
        return self.env.single_observation_space

    @property
    def action_space(self):
        """Retorna action space"""
        if isinstance(self.env, list):
            return self.env[0].action_space
        return self.env.single_action_space


def create_environment(
    world: int = 1,
    stage: int = 1,
    num_envs: int = 1,
    parallel: bool = False,
    **kwargs
) -> MarioEnvironment | ParallelMarioEnvironment:
    """
    Factory function para crear entorno

    Args:
        world: Mundo
        stage: Etapa
        num_envs: Número de entornos
        parallel: Usar entornos paralelos
        **kwargs: Argumentos adicionales

    Returns:
        Entorno creado
    """
    if parallel and num_envs > 1:
        return ParallelMarioEnvironment(
            num_envs=num_envs,
            world=world,
            stage=stage,
            **kwargs
        )
    else:
        return MarioEnvironment(
            world=world,
            stage=stage,
            **kwargs
        )
