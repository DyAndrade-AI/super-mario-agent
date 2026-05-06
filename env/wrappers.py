"""
Wrappers del entorno para Super Mario Bros
Procesamiento de frames, normalización, frame stacking, etc.
"""
try:
    # Intentar usar gym primero (gym-super-mario-bros)
    import gym
    from gym import spaces
except ImportError:
    # Fallback a gymnasium
    import gymnasium as gym
    from gymnasium import spaces

import numpy as np
from collections import deque
import multiprocessing as mp
import traceback
from typing import Optional


def unpack_step_result(step_result):
    """Normaliza resultados step() de Gym viejo o Gymnasium a 5 valores."""
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
    elif len(step_result) == 4:
        obs, reward, done, info = step_result
        terminated = done
        truncated = False
    else:
        raise ValueError(f"Formato de step() no soportado: {len(step_result)} valores")

    if info is None:
        info = {}

    return obs, reward, terminated, truncated, info


def reset_env(env, **kwargs):
    """Resetea entornos Gym/Gymnasium tolerando kwargs no soportados."""
    clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    seed = clean_kwargs.pop("seed", None)

    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)

    try:
        return env.reset(**clean_kwargs)
    except TypeError as exc:
        if clean_kwargs and "unexpected keyword" in str(exc):
            return env.reset()
        raise


class GrayscaleFrame(gym.ObservationWrapper):
    """Convierte frames a escala de grises"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(env.observation_space.shape[0], env.observation_space.shape[1]),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        """Resetea y aplica transformación"""
        result = reset_env(self.env, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            obs, info = result[0], result[1]
        else:
            obs = result
            info = {}
        return self.observation(obs), info

    def observation(self, obs):
        """Convierte a escala de grises"""
        if obs.shape[-1] == 3:
            # RGB a grayscale usando pesos estándar
            gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            return gray
        return obs


class ResizeFrame(gym.ObservationWrapper):
    """Redimensiona los frames"""

    def __init__(self, env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size, size),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        """Resetea y aplica transformación"""
        result = reset_env(self.env, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            obs, info = result[0], result[1]
        else:
            obs = result
            info = {}
        return self.observation(obs), info

    def observation(self, obs):
        """Redimensiona el frame"""
        import cv2
        resized = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return resized


class FrameStack(gym.ObservationWrapper):
    """Apila los últimos N frames"""

    def __init__(self, env, num_frames: int = 4):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

        # Actualizar observation space
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], num_frames),
            dtype=np.uint8,
        )

    def observation(self, obs):
        """Apila frames"""
        self.frames.append(obs)

        # Rellenar con frames iguales al inicio
        while len(self.frames) < self.num_frames:
            self.frames.append(obs)

        # Stack frames en último eje
        stacked = np.stack(list(self.frames), axis=-1)
        return stacked

    def reset(self, **kwargs):
        """Resetea los frames"""
        self.frames.clear()
        result = reset_env(self.env, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            obs, info = result[0], result[1]
        else:
            obs = result
            info = {}
        obs = self.observation(obs)
        return obs, info


class FrameSkip(gym.Wrapper):
    """Salta frames (acción repetida)"""

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self.total_reward = 0.0

    def step(self, action):
        """Ejecuta acción por skip frames"""
        total_reward = 0.0
        done = False
        terminated = False
        truncated = False
        info = {}
        use_new_step_api = True

        for _ in range(self.skip):
            step_result = self.env.step(action)
            use_new_step_api = len(step_result) == 5
            obs, reward, terminated, truncated, info = unpack_step_result(step_result)
            total_reward += reward
            done = terminated or truncated

            if done:
                break

        if use_new_step_api:
            return obs, total_reward, terminated, truncated, info
        return obs, total_reward, done, info


class NormalizeObservation(gym.ObservationWrapper):
    """Normaliza observaciones a [0, 1]"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        """Resetea y aplica transformación"""
        result = reset_env(self.env, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            obs, info = result[0], result[1]
        else:
            obs = result
            info = {}
        return self.observation(obs), info

    def observation(self, obs):
        """Normaliza el frame"""
        return obs.astype(np.float32) / 255.0


class MarioObservationWrapper(gym.Wrapper):
    """Wrapper combinado para Super Mario Bros"""

    def __init__(self, env, frame_size: int = 84, num_frames: int = 4, skip: int = 4):
        """
        Inicializa el wrapper

        Args:
            env: Entorno base
            frame_size: Tamaño de los frames
            num_frames: Número de frames a apilar
            skip: Número de frames a saltar
        """
        # Aplicar wrappers en orden
        env = FrameSkip(env, skip=skip)
        env = GrayscaleFrame(env)
        env = ResizeFrame(env, size=frame_size)
        env = FrameStack(env, num_frames=num_frames)
        env = NormalizeObservation(env)

        super().__init__(env)
        self.frame_size = frame_size
        self.num_frames = num_frames

    def reset(self, **kwargs):
        """Resetea el entorno"""
        return reset_env(self.env, **kwargs)

    def step(self, action):
        """Ejecuta un paso"""
        return self.env.step(action)


class InfoWrapper(gym.Wrapper):
    """Wrapper que proporciona información mejorada del estado"""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        """Resetea el entorno"""
        result = reset_env(self.env, **kwargs)

        # Manejar diferentes formatos de retorno
        if isinstance(result, tuple):
            if len(result) == 2:
                obs, info = result
            elif len(result) == 1:
                obs = result[0]
                info = {}
            else:
                # Más de 2 valores, tomar los dos primeros
                obs, info = result[0], result[1]
        else:
            obs = result
            info = {}

        return obs, info

    def step(self, action):
        """Ejecuta paso y retorna info mejorada"""
        step_result = self.env.step(action)
        use_new_step_api = len(step_result) == 5
        obs, reward, terminated, truncated, info = unpack_step_result(step_result)

        # Garantizar que info tiene los campos necesarios
        if "x_pos" not in info:
            info["x_pos"] = 0
        if "coins" not in info:
            info["coins"] = 0
        if "enemies_defeated" not in info:
            info["enemies_defeated"] = 0
        if "dead" not in info:
            info["dead"] = terminated

        if use_new_step_api:
            return obs, reward, terminated, truncated, info
        return obs, reward, terminated or truncated, info


class RewardClipping(gym.RewardWrapper):
    """Clipea recompensas a rango [-1, 1]"""

    def reward(self, reward):
        """Clipea recompensa"""
        return np.sign(reward)


def _format_reset_result(reset_result):
    if isinstance(reset_result, tuple) and len(reset_result) >= 2:
        return reset_result[0], reset_result[1]
    return reset_result, {}


def _mario_worker(remote, env_kwargs):
    env = None
    try:
        env = create_mario_env(**env_kwargs)

        while True:
            command, data = remote.recv()

            if command == "reset":
                obs, info = _format_reset_result(reset_env(env, **data))
                remote.send(("ok", (obs, info)))
            elif command == "step":
                obs, reward, terminated, truncated, info = unpack_step_result(env.step(int(data)))
                done = terminated or truncated

                if done:
                    info = dict(info)
                    info["terminal_observation"] = obs
                    reset_obs, reset_info = _format_reset_result(reset_env(env))
                    info["reset_info"] = reset_info
                    obs = reset_obs

                remote.send(("ok", (obs, reward, done, info)))
            elif command == "close":
                if env is not None:
                    env.close()
                remote.close()
                break
            else:
                raise RuntimeError(f"Comando desconocido: {command}")
    except EOFError:
        pass
    except Exception:
        try:
            remote.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


class SubprocMarioVectorEnv:
    """Vectorizador simple con un proceso por entorno Mario."""

    def __init__(
        self,
        num_envs: int,
        world: int = 1,
        stage: int = 1,
        frame_size: int = 84,
        num_frames: int = 4,
        skip: int = 4,
        render_mode: Optional[str] = None,
    ):
        self.num_envs = num_envs
        self.closed = False
        self.waiting = False
        self.env_kwargs = {
            "world": world,
            "stage": stage,
            "frame_size": frame_size,
            "num_frames": num_frames,
            "skip": skip,
            "apply_wrappers": True,
            "render_mode": render_mode,
        }

        sample_env = create_mario_env(**self.env_kwargs)
        self.single_observation_space = sample_env.observation_space
        self.single_action_space = sample_env.action_space
        sample_env.close()

        ctx = mp.get_context("spawn")
        self.remotes, worker_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.processes = []

        for worker_remote in worker_remotes:
            process = ctx.Process(
                target=_mario_worker,
                args=(worker_remote, self.env_kwargs),
                daemon=True,
            )
            process.start()
            worker_remote.close()
            self.processes.append(process)

    def _recv(self, remote):
        status, payload = remote.recv()
        if status == "error":
            self.close()
            raise RuntimeError(f"Error en worker de Mario:\n{payload}")
        return payload

    def reset(self, seed: Optional[int] = None):
        for i, remote in enumerate(self.remotes):
            kwargs = {}
            if seed is not None:
                kwargs["seed"] = seed + i
            remote.send(("reset", kwargs))

        results = [self._recv(remote) for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), {"all": list(infos)}

    def step(self, actions):
        actions = np.asarray(actions).reshape(-1)
        if len(actions) != self.num_envs:
            raise ValueError(f"Se esperaban {self.num_envs} acciones, llegaron {len(actions)}")

        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))

        results = [self._recv(remote) for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            {"all": list(infos)},
        )

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass

        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

        for remote in self.remotes:
            try:
                remote.close()
            except Exception:
                pass

        self.closed = True

    def __del__(self):
        self.close()

    @property
    def observation_space(self):
        return self.single_observation_space

    @property
    def action_space(self):
        return self.single_action_space


def create_mario_env(
    world: int = 1,
    stage: int = 1,
    frame_size: int = 84,
    num_frames: int = 4,
    skip: int = 4,
    apply_wrappers: bool = True,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """
    Crea un entorno de Super Mario Bros con wrappers

    Args:
        world: Mundo (1-8)
        stage: Etapa (1-4)
        frame_size: Tamaño de frames
        num_frames: Frames a apilar
        skip: Frame skip
        apply_wrappers: Aplicar wrappers
        render_mode: Modo de renderizado

    Returns:
        Entorno configurado
    """
    try:
        # Intentar usar gym-super-mario-bros
        import gym as gym_import
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
        import gym_super_mario_bros

        env_id = f"SuperMarioBros-{world}-{stage}-v3"
        env = gym_import.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        print(f"[OK] Entorno real creado: {env_id} ({len(SIMPLE_MOVEMENT)} acciones)")

    except Exception as e:
        # Fallback: usar simulador
        print(f"[FALLBACK] Error con el entorno real: {type(e).__name__}")
        print(f"[FALLBACK] Usando simulador de Super Mario Bros en su lugar")

        try:
            from .mario_simulator import create_mario_simulator
            env = create_mario_simulator(world=world, stage=stage, render_mode=render_mode)
            print(f"[OK] Simulador creado como fallback")
        except Exception as e2:
            raise RuntimeError(
                f"Error al crear entorno: {e}\n"
                f"Y error al crear simulador: {e2}"
            )

    if apply_wrappers:
        # Aplicar wrappers
        env = InfoWrapper(env)
        env = MarioObservationWrapper(
            env,
            frame_size=frame_size,
            num_frames=num_frames,
            skip=skip,
        )

    return env


def create_vectorized_env(
    num_envs: int = 4,
    world: int = 1,
    stage: int = 1,
    frame_size: int = 84,
    num_frames: int = 4,
    skip: int = 4,
    render_mode: Optional[str] = None,
):
    """
    Crea entornos vectorizados para entrenamiento paralelo

    Args:
        num_envs: Número de entornos paralelos
        world: Mundo
        stage: Etapa
        frame_size: Tamaño de frames
        num_frames: Frames a apilar
        skip: Frame skip
        render_mode: Modo de renderizado

    Returns:
        Entorno vectorizado
    """
    try:
        env = SubprocMarioVectorEnv(
            num_envs=num_envs,
            world=world,
            stage=stage,
            frame_size=frame_size,
            num_frames=num_frames,
            skip=skip,
            render_mode=render_mode,
        )
        print(f"[OK] VectorEnv paralelo creado con {num_envs} procesos")
        return env
    except Exception as exc:
        print(f"VectorEnv paralelo no disponible ({type(exc).__name__}: {exc})")
        print("Usando entornos secuenciales como fallback")
        return [
            create_mario_env(
                world=world,
                stage=stage,
                frame_size=frame_size,
                num_frames=num_frames,
                skip=skip,
                apply_wrappers=True,
                render_mode=render_mode,
            )
            for _ in range(num_envs)
        ]
