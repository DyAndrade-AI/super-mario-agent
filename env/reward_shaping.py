"""
Diseño de recompensas personalizado para Super Mario Bros
Shapes la señal de recompensa para facilitar el aprendizaje
"""
from typing import Tuple, Optional
import numpy as np


class MarioRewardShaper:
    """
    Formador de recompensas para Super Mario Bros
    Combina señales de progresión, supervivencia y actividad
    """

    def __init__(
        self,
        scale_position: float = 1.0,
        scale_coin: float = 50.0,
        scale_kill: float = 100.0,
        penalty_death: float = -50.0,
        penalty_idle: float = -0.01,
        penalty_stuck: float = -0.05,
    ):
        """
        Inicializa el formador de recompensas

        Args:
            scale_position: Escala para recompensa de avance
            scale_coin: Escala para recoger monedas
            scale_kill: Escala para matar enemigos
            penalty_death: Penalización por muerte
            penalty_idle: Penalización por quedarse quieto
            penalty_stuck: Penalización por atascarse
        """
        self.scale_position = scale_position
        self.scale_coin = scale_coin
        self.scale_kill = scale_kill
        self.penalty_death = penalty_death
        self.penalty_idle = penalty_idle
        self.penalty_stuck = penalty_stuck

        # State tracking
        self.last_x_pos = 0
        self.last_coins = 0
        self.last_enemies = 0
        self.idle_steps = 0
        self.max_x_pos = 0

    def shape_reward(
        self,
        reward: float,
        done: bool,
        info: dict,
    ) -> float:
        """
        Forma la recompensa del ambiente

        Args:
            reward: Recompensa original del ambiente
            done: Si el episodio terminó
            info: Info del ambiente

        Returns:
            Recompensa shaped
        """
        shaped_reward = float(reward)

        # Extraer información del estado
        x_pos = info.get("x_pos", 0)
        coins = info.get("coins", 0)
        enemies = info.get("enemies_defeated", 0)
        is_dead = info.get("dead", False)

        # Recompensa por progresión horizontal
        if x_pos > self.last_x_pos:
            delta_pos = x_pos - self.last_x_pos
            shaped_reward += delta_pos * self.scale_position
            self.idle_steps = 0  # Reset idle counter
        else:
            self.idle_steps += 1

        # Recompensa por recoger monedas
        if coins > self.last_coins:
            delta_coins = coins - self.last_coins
            shaped_reward += delta_coins * self.scale_coin

        # Recompensa por matar enemigos
        if enemies > self.last_enemies:
            delta_enemies = enemies - self.last_enemies
            shaped_reward += delta_enemies * self.scale_kill

        # Penalización por muerte
        if is_dead:
            shaped_reward += self.penalty_death

        # Penalización por inactividad
        if self.idle_steps > 10:
            shaped_reward += self.penalty_idle

        # Penalización por atascarse
        if self.idle_steps > 50:
            shaped_reward += self.penalty_stuck

        # Actualizar tracking
        self.last_x_pos = x_pos
        self.last_coins = coins
        self.last_enemies = enemies
        self.max_x_pos = max(self.max_x_pos, x_pos)

        return shaped_reward

    def reset(self):
        """Resetea el estado del shaper"""
        self.last_x_pos = 0
        self.last_coins = 0
        self.last_enemies = 0
        self.idle_steps = 0
        self.max_x_pos = 0

    def get_progress_metrics(self) -> dict:
        """
        Obtiene métricas de progreso

        Returns:
            Diccionario con métricas
        """
        return {
            "current_x_pos": self.last_x_pos,
            "max_x_pos": self.max_x_pos,
            "coins": self.last_coins,
            "enemies_defeated": self.last_enemies,
            "idle_steps": self.idle_steps,
        }


class CurriculumRewardShaper:
    """
    Adaptador de recompensas para curriculum learning
    Ajusta el peso de las recompensas según el progreso
    """

    def __init__(self, base_shaper: MarioRewardShaper):
        """
        Inicializa el curriculum shaper

        Args:
            base_shaper: Formador base de recompensas
        """
        self.base_shaper = base_shaper
        self.difficulty_level = 0.0  # 0.0 a 1.0

    def set_difficulty(self, level: float):
        """
        Ajusta el nivel de dificultad

        Args:
            level: Nivel de 0.0 (fácil) a 1.0 (difícil)
        """
        self.difficulty_level = np.clip(level, 0.0, 1.0)

        # Ajustar penalizaciones según dificultad
        # Mayor dificultad = penalizaciones más severas
        self.base_shaper.penalty_idle *= (1.0 + self.difficulty_level)
        self.base_shaper.penalty_stuck *= (1.0 + self.difficulty_level)

    def shape_reward(
        self,
        reward: float,
        done: bool,
        info: dict,
    ) -> float:
        """
        Forma recompensa usando curriculum

        Args:
            reward: Recompensa original
            done: Si terminó
            info: Info del ambiente

        Returns:
            Recompensa shaped
        """
        return self.base_shaper.shape_reward(reward, done, info)


def create_reward_shaper(
    scale_position: float = 1.0,
    scale_coin: float = 50.0,
    scale_kill: float = 100.0,
    penalty_death: float = -50.0,
    penalty_idle: float = -0.01,
    penalty_stuck: float = -0.05,
) -> MarioRewardShaper:
    """
    Factory function para crear un formador de recompensas

    Args:
        Parámetros de escala

    Returns:
        MarioRewardShaper
    """
    return MarioRewardShaper(
        scale_position=scale_position,
        scale_coin=scale_coin,
        scale_kill=scale_kill,
        penalty_death=penalty_death,
        penalty_idle=penalty_idle,
        penalty_stuck=penalty_stuck,
    )
