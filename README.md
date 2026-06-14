# Super Mario Bros Dueling DQN Agent

Proyecto de reinforcement learning para entrenar un agente Q-learning en Super Mario Bros. La implementacion activa usa Dueling Double DQN con replay priorizado, n-step returns, target network, exploracion epsilon-greedy y Noisy Networks opcional.

## Overview

El pipeline incluye:

- Dueling DQN para estimar `Q(s, a)`
- Double DQN para reducir sobreestimacion de Q-values
- Prioritized Experience Replay
- n-step returns para propagar recompensa mas rapido
- Target network con actualizacion periodica
- CNN residual para frames 84x84 apilados
- Wrappers de frame skip, grayscale, resize, frame stack y normalizacion
- Reward shaping para progreso, monedas, enemigos, inactividad, muerte y final de nivel
- Entornos paralelos por multiprocessing
- Checkpoints, evaluacion, playback y videos

El objetivo por defecto es World 1-1.

## Estructura

```text
super_mario_agent/
├── agents/
│   ├── dqn_agent.py        # Dueling Double DQN + PER + n-step
│   └── memory.py           # Replay buffer priorizado
├── config/
│   ├── config.py
│   └── hyperparameters.py  # Configuracion DQN
├── env/
│   ├── mario_env.py
│   ├── mario_simulator.py
│   ├── reward_shaping.py
│   └── wrappers.py
├── models/
│   ├── cnn_backbone.py
│   └── dueling_dqn.py
├── utils/
├── train.py
├── evaluate.py
├── play_agent.py
└── main.py
```

## Instalacion

```bash
pip install -r requirements.txt
```

Dependencias principales:

- Python 3.10+
- PyTorch
- `gym-super-mario-bros`
- `nes-py`
- OpenCV
- TensorBoard

## Entrenamiento

Entrenar con defaults:

```bash
python train.py
```

Entrenar una etapa concreta:

```bash
python train.py --world 1 --stage 1 --total-timesteps 50000000
```

Reanudar:

```bash
python train.py --resume outputs/checkpoints/last_checkpoint.pt
```

Tambien puedes usar el CLI:

```bash
python main.py train --world 1 --stage 1 --total-timesteps 50000000
```

## Evaluacion

```bash
python evaluate.py outputs/checkpoints/best_model.pt --episodes 10
```

## Playback

```bash
python play_agent.py outputs/checkpoints/best_model.pt --episodes 1
```

Guardar video:

```bash
python play_agent.py outputs/checkpoints/best_model.pt --episodes 1 --save-video --fps 30
```

## Checkpoints

El entrenamiento guarda:

- `outputs/checkpoints/best_model.pt`
- `outputs/checkpoints/best_eval_model.pt`
- `outputs/checkpoints/last_checkpoint.pt`
- `outputs/checkpoints/checkpoint_step_*.pt`

Los checkpoints DQN guardan la Q-network online, la target network, el optimizador y estado de entrenamiento. Checkpoints antiguos de otros algoritmos no son compatibles.

## Hiperparametros Clave

Editar `config/hyperparameters.py`.

```python
REPLAY_BUFFER_SIZE = 150_000
LEARNING_STARTS = 20_000
N_STEP_RETURNS = 3
TARGET_UPDATE_INTERVAL = 10_000
EPSILON_START = 1.0
EPSILON_END = 0.02
USE_NOISY_NETS = True
COLLECT_STEPS = 128
GRADIENT_STEPS = 128
NUM_ENVS_PARALLEL = 8
```

Para mas estabilidad en equipos con poca RAM, baja `REPLAY_BUFFER_SIZE`, `NUM_ENVS_PARALLEL` y `BATCH_SIZE`.
