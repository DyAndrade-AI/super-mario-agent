# Guia de Instalacion y Setup

## Requisitos

- Python 3.10 o superior
- RAM: 16GB recomendado, mas si subes `REPLAY_BUFFER_SIZE`
- GPU NVIDIA con CUDA recomendada
- 20GB+ libres para checkpoints, logs y videos

## Instalacion

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Verificaciones rapidas:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import gym_super_mario_bros; print('Mario env OK')"
python -c "import cv2; print(cv2.__version__)"
```

## CUDA

Instala PyTorch con el build CUDA que corresponda a tu sistema. Ejemplo:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

El proyecto selecciona por defecto:

```python
CUDA_VISIBLE_DEVICES=0
USE_GPU = True
DEVICE_ID = 0
```

## Primer Entrenamiento

```bash
python main.py train --world 1 --stage 1 --total-timesteps 1000000
```

Monitoreo:

```bash
tensorboard --logdir outputs/logs
```

## Ajustes de Rendimiento

GPU potente:

```python
NUM_ENVS_PARALLEL = 16
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 300_000
GRADIENT_STEPS = 256
```

Equipo normal:

```python
NUM_ENVS_PARALLEL = 8
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 150_000
GRADIENT_STEPS = 128
```

CPU o poca RAM:

```python
USE_GPU = False
NUM_ENVS_PARALLEL = 2
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50_000
GRADIENT_STEPS = 32
```

## Problemas Comunes

`CUDA out of memory`:

- Baja `BATCH_SIZE`
- Baja `GRADIENT_STEPS`
- Desactiva `USE_MIXED_PRECISION` solo si el error viene de AMP

RAM alta:

- Baja `REPLAY_BUFFER_SIZE`
- Baja `NUM_ENVS_PARALLEL`

Entorno real no disponible:

- El repo intentara usar el simulador local de fallback.
- Para el entorno real instala `gym-super-mario-bros` y `nes-py`.
