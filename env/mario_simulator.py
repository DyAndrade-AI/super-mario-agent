"""
Simulador de Super Mario Bros para evitar problemas con ROM
Proporciona un entorno compatible que simula el juego de Super Mario
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple


class SimpleMarioSimulator(gym.Env):
    """
    Simulador simple de Super Mario Bros
    Simula el comportamiento básico del juego para propósitos de entrenamiento
    """

    def __init__(self, world: int = 1, stage: int = 1, render_mode: Optional[str] = None):
        """
        Inicializa el simulador

        Args:
            world: Mundo (1-8)
            stage: Etapa (1-4)
            render_mode: Modo de renderizado
        """
        super().__init__()
        self.world = world
        self.stage = stage
        self.render_mode = render_mode

        # Espacio de observación: frames 84x84 RGB apilados 4 veces
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 4),
            dtype=np.uint8
        )

        # Espacio de acciones: 6 acciones básicas de Mario
        # 0: NOOP
        # 1: Right
        # 2: Right + Jump
        # 3: Left
        # 4: Jump
        # 5: Left + Jump
        self.action_space = spaces.Discrete(6)

        # Estado del juego
        self.x_pos = 0
        self.time_left = 400
        self.coins = 0
        self.enemies_defeated = 0
        self.level_complete = False
        self.done = False
        self.step_count = 0
        self.max_steps = 5000

        # Buffer de frames para stacking
        self.frame_buffer = None

    def _get_observation(self) -> np.ndarray:
        """Retorna la observación actual"""
        if self.frame_buffer is None:
            self.frame_buffer = np.zeros((84, 84, 4), dtype=np.uint8)

        # Crear un frame simple basado en la posición
        frame = np.ones((84, 84), dtype=np.uint8) * 50  # Fondo

        # Dibujar el piso
        frame[70:, :] = 100

        # Dibujar a Mario (como un pequeño rectángulo)
        mario_x = int((self.x_pos % 256) * 84 / 256)
        frame[60:70, mario_x:mario_x+6] = 200

        # Actualizar buffer
        self.frame_buffer = np.roll(self.frame_buffer, 1, axis=-1)
        self.frame_buffer[:, :, 0] = frame

        return self.frame_buffer.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Ejecuta un paso en el entorno

        Args:
            action: Acción a ejecutar

        Returns:
            Tupla (observación, recompensa, terminado, truncado, info)
        """
        self.step_count += 1
        reward = 0.0
        truncated = self.step_count >= self.max_steps

        # Procesar acción
        if action == 1 or action == 2:  # Right
            self.x_pos += 2
            reward += 0.01
        elif action == 3 or action == 5:  # Left
            self.x_pos = max(0, self.x_pos - 1)

        if action == 2 or action == 4 or action == 5:  # Jump
            reward += 0.05

        # Tiempo
        self.time_left -= 1
        if self.time_left <= 0:
            self.done = True

        # Simulación de progreso en el nivel
        if self.x_pos > 250:
            self.level_complete = True
            reward += 100
            self.done = True

        # Simulación de enemigos
        if self.step_count % 50 == 0 and np.random.random() < 0.1:
            self.enemies_defeated += 1
            reward += 10

        # Pequeña probabilidad de encontrar monedas
        if np.random.random() < 0.02:
            self.coins += 1
            reward += 5

        obs = self._get_observation()
        info = {
            "x_pos": self.x_pos,
            "coins": self.coins,
            "enemies_defeated": self.enemies_defeated,
            "time_left": self.time_left,
            "level_complete": self.level_complete,
            "world": self.world,
            "stage": self.stage,
        }

        return obs, reward, self.done, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Resetea el entorno

        Args:
            seed: Seed para reproducibilidad
            options: Opciones adicionales

        Returns:
            Tupla (observación, info)
        """
        super().reset(seed=seed)

        self.x_pos = 0
        self.time_left = 400
        self.coins = 0
        self.enemies_defeated = 0
        self.level_complete = False
        self.done = False
        self.step_count = 0
        self.frame_buffer = None

        obs = self._get_observation()
        info = {
            "x_pos": self.x_pos,
            "coins": self.coins,
            "enemies_defeated": self.enemies_defeated,
            "time_left": self.time_left,
            "world": self.world,
            "stage": self.stage,
        }

        return obs, info

    def render(self) -> Optional[np.ndarray]:
        """Renderiza el entorno (opcional)"""
        if self.render_mode == "rgb_array":
            return self._get_observation()
        return None

    def close(self):
        """Cierra el entorno"""
        pass


def create_mario_simulator(
    world: int = 1,
    stage: int = 1,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """
    Crea un simulador de Super Mario Bros

    Args:
        world: Mundo
        stage: Etapa
        render_mode: Modo de renderizado

    Returns:
        Entorno de Super Mario simulado
    """
    return SimpleMarioSimulator(world=world, stage=stage, render_mode=render_mode)
