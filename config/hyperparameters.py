"""
Hiperparámetros de entrenamiento para el agente PPO + CNN + LSTM
Diseñados para aprovechar recursos computacionales potentes
"""

# ==================== ENTORNO ====================
ENV_NAME = "SuperMarioBros-v0"
WORLD = 1  # Mundo 1 de Super Mario Bros
STAGE = 1  # Etapa 1
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_STACK = 4
FRAME_SKIP = 4
MAX_EPISODE_STEPS = 4500  # Máximo de pasos por episodio

# ==================== ARQUITECTURA ====================
# CNN Residual Backbone
CNN_FEATURES = [32, 64, 64]
CNN_KERNEL_SIZES = [8, 4, 3]
CNN_STRIDES = [4, 2, 1]
CNN_PADDINGS = [0, 0, 0]
CNN_USE_BATCH_NORM = True

# LSTM
USE_LSTM = True
LSTM_HIDDEN_DIM = 512
LSTM_NUM_LAYERS = 1

# Policy Head
POLICY_HIDDEN_DIM = 512
VALUE_HIDDEN_DIM = 512

# ==================== PPO HIPERPARÁMETROS ====================
# Learning
LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = True
LEARNING_RATE_DECAY_STEPS = 50_000_000  # Decaimiento a lo largo del entrenamiento
LR_FINAL_FACTOR = 0.1  # LR final = LR inicial * factor

# Clipping y Normalización
CLIP_RANGE = 0.05
CLIP_RANGE_VF = 0.05  # Para value function
CLIP_GRAD_NORM = 0.5
VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.02
ENTROPY_COEFF_DECAY = False
TARGET_KL = 0.02

# Ventaja y Recompensa
GAMMA = 0.99  # Factor de descuento
GAE_LAMBDA = 0.95  # Generalized Advantage Estimation
NORMALIZE_ADVANTAGE = True
NORMALIZE_REWARD = True
REWARD_SCALE = 1.0

# Batch
ROLLOUT_STEPS = 512  # Pasos de recolección antes de update
NUM_ENVS_PARALLEL = 16  # Ambientes paralelos
BATCH_SIZE = 128  # Batch size para SGD
EPOCHS_PER_UPDATE = 2  # Epocas dentro de cada update
SHUFFLE_BATCH = True

# ==================== ENTRENAMIENTO ====================
TOTAL_TIMESTEPS = 50_000_000  # 50M steps de entrenamiento
EVAL_FREQ = 50_000  # Evaluar cada 50k pasos
EVAL_EPISODES = 5  # Episodios para evaluación
EVAL_RENDER = False  # Renderizar durante evaluación

CHECKPOINT_FREQ = 500_000  # Guardar checkpoint cada 500k pasos
SAVE_VIDEO_FREQ = 1_000_000  # Guardar video cada 1M pasos
SAVE_VIDEO_LENGTH = 4500  # Duración del video

# ==================== TÉCNICAS AVANZADAS ====================
# Curiosity Driven Exploration (RND)
USE_CURIOSITY = False
CURIOSITY_COEFF = 0.01
CURIOSITY_LEARNING_RATE = 1e-3

# Normalización
USE_LAYER_NORM = True
USE_BATCH_NORM_INPUT = False  # No usar en input layer de CNN

# Inicialización de Pesos
WEIGHT_INIT_SCALE = 0.5  # Escala para initialización

# Mixed Precision Training
USE_MIXED_PRECISION = True
AUTOCAST_DTYPE = "float16"

# Reward Shaping
REWARD_SCALE_POSITION = 1.0  # Recompensa por avanzar
REWARD_SCALE_COIN = 50.0  # Recompensa por recoger monedas
REWARD_SCALE_KILL = 100.0  # Recompensa por matar enemigos
REWARD_PENALTY_DEATH = -50.0  # Penalización por morir
REWARD_PENALTY_IDLE = -0.01  # Penalización por quedarse quieto

# ==================== DEVICE ====================
USE_GPU = True
DEVICE_ID = 0  # GPU ID si hay múltiples GPUs
NUM_WORKERS_DATALOADER = 4

# ==================== LOGGING ====================
USE_WANDB = False  # Cambiar a True para usar Weights & Biases
WANDB_PROJECT = "super_mario_rl"
WANDB_ENTITY = "yahir"  # Cambiar a tu usuario de W&B

USE_TENSORBOARD = True
TENSORBOARD_LOG_DIR = "./outputs/logs"

LOG_FREQ = 100  # Loguear cada N pasos

# ==================== RANDOM SEED ====================
SEED = 42

# ==================== CURRICULUM LEARNING ====================
USE_CURRICULUM = True
CURRICULUM_STAGES = [
    {"world": 1, "stage": 1, "steps": 10_000_000},  # 10M pasos en 1-1
    {"world": 1, "stage": 2, "steps": 10_000_000},  # 10M pasos en 1-2
    {"world": 1, "stage": 3, "steps": 10_000_000},  # 10M pasos en 1-3
    {"world": 1, "stage": 4, "steps": 20_000_000},  # 20M pasos en 1-4
]

# ==================== EARLY STOPPING ====================
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 50  # Paciencia en evaluaciones
EARLY_STOPPING_MIN_DELTA = 0.0  # Mejora mínima requerida
