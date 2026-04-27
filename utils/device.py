"""
Utilidades para manejo de dispositivos (GPU/CPU)
"""
import torch
import torch.cuda as cuda
from typing import Tuple


class DeviceManager:
    """Gestor de dispositivos para entrenamiento"""

    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        """
        Inicializa el gestor de dispositivos

        Args:
            use_gpu: Usar GPU si está disponible
            device_id: ID de la GPU a usar
        """
        self.use_gpu = use_gpu and cuda.is_available()
        self.device_id = device_id if self.use_gpu else 0
        self.device = self._setup_device()
        self._print_device_info()

    def _setup_device(self) -> torch.device:
        """Configura el dispositivo"""
        if self.use_gpu:
            if self.device_id < cuda.device_count():
                cuda.set_device(self.device_id)
                return torch.device(f"cuda:{self.device_id}")
            else:
                print(f"GPU {self.device_id} no disponible. Usando CPU.")
                self.use_gpu = False
                return torch.device("cpu")
        return torch.device("cpu")

    def _print_device_info(self):
        """Imprime información del dispositivo"""
        print(f"\n{'='*60}")
        print(f"Device Manager Inicializado")
        print(f"{'='*60}")
        print(f"Dispositivo activo: {self.device}")
        print(f"Usando GPU: {self.use_gpu}")

        if self.use_gpu:
            print(f"GPU disponibles: {cuda.device_count()}")
            print(f"GPU actual: {cuda.current_device()}")
            print(f"GPU Name: {cuda.get_device_name(self.device_id)}")
            print(f"CUDA Version: {cuda.get_device_capability(self.device_id)}")

            # Memoria GPU
            total_memory = cuda.get_device_properties(self.device_id).total_memory / 1e9
            print(f"Memoria total GPU: {total_memory:.2f} GB")

            # Memory allocated
            allocated = cuda.memory_allocated(self.device_id) / 1e9
            print(f"Memoria asignada: {allocated:.2f} GB")

        print(f"PyTorch Version: {torch.__version__}")
        print(f"{'='*60}\n")

    def get_device(self) -> torch.device:
        """Retorna el dispositivo configurado"""
        return self.device

    def empty_cache(self):
        """Vacía el caché de GPU"""
        if self.use_gpu:
            cuda.empty_cache()

    def get_device_properties(self) -> dict:
        """Obtiene propiedades del dispositivo"""
        if self.use_gpu:
            props = cuda.get_device_properties(self.device_id)
            return {
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": props.compute_capability,
                "multi_processor_count": props.multi_processor_count,
            }
        return {"type": "cpu"}


def get_dtype_and_device(
    use_mixed_precision: bool = False,
    device: torch.device = None,
) -> Tuple[torch.dtype, torch.device]:
    """
    Obtiene el dtype y device apropiados

    Args:
        use_mixed_precision: Usar mixed precision training
        device: Dispositivo a usar (si es None, usa CPU)

    Returns:
        Tupla (dtype, device)
    """
    if device is None:
        device = torch.device("cpu")

    if use_mixed_precision and device.type == "cuda":
        # Usar float16 con autocast en GPU
        return torch.float16, device
    else:
        # Usar float32 por defecto
        return torch.float32, device
