"""
Utilidades para grabar videos del agente jugando
"""
from pathlib import Path
from typing import Optional
import numpy as np
import cv2


class VideoRecorder:
    """Grabadora de videos para guardar el agente jugando"""

    def __init__(
        self,
        output_dir: Path,
        fps: int = 30,
        frame_width: int = 256,
        frame_height: int = 240,
    ):
        """
        Inicializa la grabadora de video

        Args:
            output_dir: Directorio para guardar videos
            fps: Frames por segundo
            frame_width: Ancho del frame (original de Mario)
            frame_height: Alto del frame (original de Mario)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_writer = None
        self.frames = []
        self.recording = False

    def start_recording(self, episode_num: int):
        """
        Comienza a grabar un video

        Args:
            episode_num: Número del episodio
        """
        self.frames = []
        self.recording = True
        self.current_episode = episode_num

    def add_frame(self, frame: np.ndarray):
        """
        Añade un frame al video

        Args:
            frame: Frame como array numpy (H, W, 3) o (H, W)
        """
        if not self.recording:
            return

        # Convertir a RGB si es escala de grises
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Redimensionar si es necesario
        if frame.shape[:2] != (self.frame_height, self.frame_width):
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        self.frames.append(frame)

    def stop_recording(
        self,
        reward: float = 0.0,
        episode_length: int = 0,
    ) -> Optional[Path]:
        """
        Detiene la grabación y guarda el video

        Args:
            reward: Recompensa del episodio
            episode_length: Duración del episodio

        Returns:
            Path al video guardado, o None si no hay frames
        """
        if not self.recording or not self.frames:
            self.recording = False
            return None

        self.recording = False

        # Crear path del archivo
        video_path = (
            self.output_dir /
            f"episode_{self.current_episode:06d}_"
            f"reward_{reward:.1f}_"
            f"length_{episode_length}.mp4"
        )

        # Escribir video
        if self.frames:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )

            for frame in self.frames:
                out.write(frame)

            out.release()

        self.frames = []
        return video_path

    def save_frame_grid(
        self,
        frames: list,
        episode_num: int,
        rows: int = 2,
        cols: int = 4,
    ) -> Path:
        """
        Guarda una grilla de frames

        Args:
            frames: Lista de frames
            episode_num: Número de episodio
            rows: Filas de la grilla
            cols: Columnas de la grilla

        Returns:
            Path a la imagen guardada
        """
        grid_path = self.output_dir / f"grid_episode_{episode_num:06d}.png"

        # Seleccionar frames uniformemente
        num_frames = min(rows * cols, len(frames))
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        selected_frames = [frames[i] for i in indices]

        # Crear grilla
        grid = self._create_grid(selected_frames, rows, cols)

        # Guardar
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        return grid_path

    def _create_grid(self, frames: list, rows: int, cols: int) -> np.ndarray:
        """
        Crea una grilla de frames

        Args:
            frames: Lista de frames
            rows: Filas
            cols: Columnas

        Returns:
            Array con la grilla
        """
        grid_height = rows * self.frame_height
        grid_width = cols * self.frame_width

        if len(frames[0].shape) == 2:
            # Escala de grises
            grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
            for i, frame in enumerate(frames):
                row = (i // cols) * self.frame_height
                col = (i % cols) * self.frame_width
                grid[row:row + self.frame_height, col:col + self.frame_width] = frame
        else:
            # Color
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            for i, frame in enumerate(frames):
                row = (i // cols) * self.frame_height
                col = (i % cols) * self.frame_width
                grid[row:row + self.frame_height, col:col + self.frame_width] = frame

        return grid


def create_video_recorder(
    output_dir: Path,
    fps: int = 30,
) -> VideoRecorder:
    """
    Factory function para crear una grabadora de video

    Args:
        output_dir: Directorio de salida
        fps: Frames por segundo

    Returns:
        VideoRecorder
    """
    return VideoRecorder(output_dir, fps=fps)
