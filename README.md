# Super Mario Bros - Agente PPO + CNN + LSTM

Un proyecto profesional de **Deep Reinforcement Learning** que entrena un agente PPO para jugar **Super Mario Bros** de forma autónoma usando visión por computadora (CNN) y memoria temporal (LSTM).

## 🎯 Objetivo

Desarrollar un agente de RL capaz de aprender estrategias de juego complejas en Super Mario Bros mediante:
- **Algoritmo PPO**: Estable y eficiente en muestreo
- **CNN Residual**: Extrae características visuales de los frames
- **LSTM**: Proporciona memoria temporal para decision-making complejo
- **Reward Shaping**: Incentivos personalizados para acelerar el aprendizaje

## 📋 Características Principales

✅ **Arquitectura Avanzada**
- CNN con bloques residuales para extracción de características
- LSTM recurrente para memoria temporal
- Normalización y regularización de capas
- Inicialización ortogonal de pesos

✅ **Algoritmo PPO Completo**
- Clipping de ratio de probabilidad
- Clipping de value function
- Generalized Advantage Estimation (GAE)
- Normalización de ventajas y recompensas

✅ **Entrenamiento Profesional**
- Entornos paralelos para aceleración
- Guardado automático de checkpoints
- Evaluación periódica
- Reanudación desde checkpoints
- Mixed precision training (FP16) en GPU

✅ **Logging y Monitoreo**
- TensorBoard para visualización
- Weights & Biases (W&B) opcional
- Grabación de videos del agente
- Métricas detalladas de entrenamiento

✅ **Reward Shaping Inteligente**
- Recompensa por progresión horizontal
- Penalizaciones por inactividad
- Recompensas por recoger monedas y matar enemigos
- Curriculum learning por niveles

## 🛠️ Instalación

### Requisitos Previos
- Python 3.8+
- CUDA 11.8+ (para GPU, opcional pero recomendado)
- 8GB+ RAM

### Pasos de Instalación

1. **Clonar/Descargar el repositorio**
```bash
cd super_mario
```

2. **Crear entorno virtual** (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Verificar instalación**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from gymnasium_super_mario_bros import SuperMarioBrosEnv; print('Mario env OK')"
```

## 📁 Estructura del Proyecto

```
super_mario/
├── config/                    # Configuración y hiperparámetros
│   ├── hyperparameters.py    # Todos los hiperparámetros
│   └── config.py             # Configuración del proyecto
│
├── env/                       # Entorno y wrappers
│   ├── mario_env.py          # Wrapper principal del entorno
│   ├── wrappers.py           # Wrappers de procesamiento
│   └── reward_shaping.py     # Diseño de recompensas
│
├── models/                    # Arquitectura neural
│   ├── cnn_backbone.py       # CNN con bloques residuales
│   ├── recurrent_module.py   # LSTM/GRU para memoria
│   ├── actor_critic.py       # Red Actor-Critic PPO
│   └── distributions.py      # Distribuciones de probabilidad
│
├── agents/                    # Agente RL
│   ├── ppo_agent.py          # Implementación de PPO
│   └── memory.py             # Buffer de rollout y GAE
│
├── utils/                     # Utilidades
│   ├── device.py             # Manejo de GPU/CPU
│   ├── logger.py             # Logging y TensorBoard
│   ├── checkpoint.py         # Guardado de checkpoints
│   ├── metrics.py            # Rastreo de métricas
│   └── video_recorder.py     # Grabación de videos
│
├── scripts/                   # Scripts auxiliares
│   └── train_stage_*.py      # Entrenamiento por etapas
│
├── outputs/                   # Resultados
│   ├── checkpoints/          # Modelos guardados
│   ├── videos/               # Videos del agente
│   ├── logs/                 # TensorBoard logs
│   └── plots/                # Gráficas de entrenamiento
│
├── main.py                    # Entry point principal
├── train.py                   # Script de entrenamiento
├── evaluate.py               # Script de evaluación
├── play_agent.py             # Visualización en tiempo real
├── requirements.txt          # Dependencias
└── README.md                 # Este archivo
```

## 🚀 Uso

### 1. Entrenar desde Cero

```bash
python main.py train \
  --world 1 \
  --stage 1 \
  --total-timesteps 50000000 \
  --use-wandb
```

**Opciones disponibles:**
- `--world`: Mundo (1-8)
- `--stage`: Etapa (1-4)
- `--total-timesteps`: Pasos totales de entrenamiento
- `--checkpoint-dir`: Directorio para checkpoints
- `--resume`: Reanudar desde checkpoint previo
- `--use-wandb`: Usar Weights & Biases
- `--no-tensorboard`: Desabilitar TensorBoard
- `--seed`: Random seed

### 2. Reanudar Entrenamiento

```bash
python main.py train \
  --resume outputs/checkpoints/best_model.pt \
  --total-timesteps 100000000
```

### 3. Evaluar Modelo

```bash
python main.py evaluate \
  outputs/checkpoints/best_model.pt \
  --episodes 10 \
  --world 1 \
  --stage 1
```

### 4. Ver Agente Jugando

```bash
python main.py play \
  outputs/checkpoints/best_model.pt \
  --episodes 3 \
  --save-video \
  --fps 30
```

### 5. Training Alternativo (Script directo)

```bash
python train.py --world 1 --stage 1 --total-timesteps 50000000
```

### 6. Evaluación Alternativa

```bash
python evaluate.py outputs/checkpoints/best_model.pt --episodes 10
```

### 7. Visualización Alternativa

```bash
python play_agent.py outputs/checkpoints/best_model.pt --episodes 1 --save-video
```

## 📊 Hiperparámetros Principales

Editar `config/hyperparameters.py`:

```python
# Entorno
FRAME_WIDTH = 84              # Ancho del frame procesado
FRAME_HEIGHT = 84             # Alto del frame procesado
FRAME_STACK = 4               # Frames apilados (contexto temporal)

# Arquitectura
CNN_FEATURES = [32, 64, 64]  # Canales de CNN por capa
LSTM_HIDDEN_DIM = 512         # Dimensión del hidden LSTM

# PPO
LEARNING_RATE = 3e-4
GAMMA = 0.99                  # Factor de descuento
GAE_LAMBDA = 0.95             # GAE lambda
CLIP_RANGE = 0.1              # PPO clip
ENTROPY_COEFF = 0.01          # Coeficiente de entropía

# Entrenamiento
BATCH_SIZE = 128
NUM_ENVS_PARALLEL = 16        # Entornos paralelos
ROLLOUT_STEPS = 512           # Pasos entre actualizaciones
EPOCHS_PER_UPDATE = 3

# Total
TOTAL_TIMESTEPS = 50_000_000  # 50 millones de pasos
```

## 📈 Monitoreo del Entrenamiento

### TensorBoard

```bash
tensorboard --logdir outputs/logs
```

Luego abrir en el navegador: http://localhost:6006

### Weights & Biases (W&B)

```bash
pip install wandb
wandb login
python main.py train --use-wandb
```

## 🎮 Interpretación de Resultados

**Métricas clave:**
- `mean_episode_reward`: Recompensa promedio (debe crecer)
- `mean_value_loss`: Pérdida de value (debe bajar)
- `mean_policy_loss`: Pérdida de política (debe bajar)
- `mean_entropy`: Entropía de la política (indicador de exploración)
- `max_x_pos`: Progreso horizontal máximo del agente

**Señales de buen entrenamiento:**
✅ Recompensa promedio incrementa constantemente
✅ Pérdidas disminuyen
✅ Agente alcanza cada vez posiciones más lejanas
✅ Evita enemigos y obstáculos

## 🔧 Solución de Problemas

### Error: "gymnasium-super-mario-bros not found"
```bash
pip install gymnasium-super-mario-bros
```

### Error: CUDA out of memory
- Reducir `BATCH_SIZE` en `config/hyperparameters.py`
- Reducir `NUM_ENVS_PARALLEL`
- Usar `--no-mixed-precision` flag

### Entrenamiento muy lento
- Usar GPU: Asegurar CUDA instalado correctamente
- Aumentar `NUM_ENVS_PARALLEL` (si hay GPU/RAM disponible)
- Reducir `EVAL_FREQ` para menos evaluaciones

### Videos no se guardan
- Instalar `opencv-python`: `pip install opencv-python`
- Verificar permisos de escritura en `outputs/videos/`

## 🚀 Optimizaciones y Técnicas Avanzadas

### 1. **Mixed Precision Training**
Usar FP16 en GPU para mayor velocidad:
```python
USE_MIXED_PRECISION = True
```

### 2. **Reward Normalization**
Normaliza recompensas acumulativas:
```python
NORMALIZE_REWARD = True
```

### 3. **Learning Rate Decay**
Reduce gradualmente el learning rate:
```python
LEARNING_RATE_DECAY = True
LEARNING_RATE_DECAY_STEPS = 50_000_000
LR_FINAL_FACTOR = 0.1
```

### 4. **Curriculum Learning**
Entrenar primero en niveles fáciles:
```python
USE_CURRICULUM = True
CURRICULUM_STAGES = [...]
```

### 5. **Clipping de Gradientes**
Previene explosiones de gradiente:
```python
CLIP_GRAD_NORM = 0.5
```

## 📚 Referencias

- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **DQN CNN**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- **GAE**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **Residual Networks**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## 🎓 Aprendizajes Clave

Este proyecto implementa técnicas de RL de nivel profesional:

1. **Arquitecturas Profundas**: CNN residuales para visión, LSTM para memoria
2. **Algoritmos Estables**: PPO con clipping y normalización
3. **Eficiencia**: Entrenamiento paralelo, mixed precision, reward shaping
4. **Reproducibilidad**: Seeds, checkpoints, logging completo
5. **Escalabilidad**: Código modular, fácil de extender

## 📝 Licencia

Proyecto educativo. Libre de usar con fines académicos.

## 🤝 Contribuciones

Se aceptan mejoras y extensiones:
- Soporte para más juegos
- Algoritmos adicionales (A3C, IMPALA, R2D2)
- Arquitecturas mejoradas
- Mejor reward shaping

## 📧 Contacto

Para preguntas o sugerencias sobre este proyecto:
- Consultar documentación del código
- Revisar issues y PRs
- Contactar al autor

---

**Versión**: 1.0.0
**Última actualización**: 2026-04-27
**Status**: Producción ✅
