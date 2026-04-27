# Arquitectura del Proyecto - Explicación Detallada

## 1. Visión General

El proyecto implementa un **Sistema Completo de Deep Reinforcement Learning** para entrenar un agente que juegue Super Mario Bros. La arquitectura se basa en:

- **Algoritmo**: PPO (Proximal Policy Optimization)
- **Visión**: CNN Residual (84×84 frames en escala de grises)
- **Memoria**: LSTM Recurrente (512 hidden units)
- **Entrenamiento**: Paralelo en múltiples entornos

## 2. Flujo de Datos

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTORNO (Super Mario)                     │
│  Retorna: [Observación, Recompensa, Done, Info]            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          PROCESAMIENTO DE FRAMES (Wrappers)                │
│  • GrayscaleFrame: RGB → Escala de grises                  │
│  • ResizeFrame: 256×240 → 84×84                            │
│  • FrameStack: Apilar últimos 4 frames                      │
│  • FrameSkip: Acción repetida cada 4 frames               │
│  • NormalizeObservation: Dividir por 255                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           MODELO NEURAL (Actor-Critic)                      │
│                                                              │
│  INPUT: (Batch, 4, 84, 84)  # 4 frames apilados            │
│         │                                                   │
│         ▼                                                   │
│  CNN BACKBONE (Residual)                                   │
│  • Conv2d(4→32, k=8, s=4) + ReLU                          │
│  • Conv2d(32→64, k=4, s=2) + ReLU                         │
│  • Conv2d(64→64, k=3, s=1) + ReLU                         │
│  • ResidualBlocks(64→64) ×2                                │
│  Output: (Batch, 64, 7, 7)                                │
│         │                                                   │
│         ▼                                                   │
│  FLATTEN: (Batch, 3136)                                    │
│         │                                                   │
│         ▼                                                   │
│  LSTM LAYER                                                │
│  • Input: (Batch, 1, 3136)                                │
│  • Hidden: (1, Batch, 512)                                │
│  Output: (Batch, 512)                                      │
│         │                                                   │
│         ├─────────────────┬──────────────────────┐         │
│         ▼                 ▼                      ▼         │
│  POLICY HEAD       VALUE HEAD                             │
│  • FC(512→512)     • FC(512→512)                           │
│  • ReLU            • ReLU                                  │
│  • FC(512→18)      • FC(512→1)                            │
│  Output: Logits    Output: V(s)                            │
│         │                 │                               │
│         ▼                 ▼                               │
│  [Action Logits]  [Value Estimate]                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              RECOLECCIÓN DE EXPERIENCIAS                    │
│  • Muestrear acción del modelo                            │
│  • Ejecutar en entorno                                    │
│  • Guardar: (s, a, r, v, log_π)                          │
│  • Repetir 512 pasos (rollout)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│        GAE: Generalized Advantage Estimation                │
│  • Calcular TD-residuals: δ = r + γV(s') - V(s)           │
│  • Aplicar GAE con λ=0.95                                 │
│  • Obtener Advantages y Returns                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             ACTUALIZACIÓN PPO (Mini-batches)               │
│  Para cada época (3 épocas):                              │
│    Para cada mini-batch:                                  │
│      1. Forward pass: obtener π_new(a|s), V_new(s)       │
│      2. Calcular ratio: r_t = π_new / π_old              │
│      3. Policy loss = -min(r_t·A, clip(r_t)·A)          │
│      4. Value loss = MSE(V_new, Returns)                 │
│      5. Total loss = Policy_loss + 0.5·Value_loss       │
│                     - 0.01·Entropy                        │
│      6. Backward + Optimizer step                         │
└─────────────────────────────────────────────────────────────┘
```

## 3. Componentes Principales

### 3.1 CNN Backbone (`models/cnn_backbone.py`)

**Propósito**: Extraer características visuales de los frames

**Características**:
- Convoluciones con ReLU
- Batch normalization
- Bloques residuales
- Layer normalization

**Arquitectura**:
```python
Input: (B, 4, 84, 84)
  ↓
Conv2d(4→32, k=8, s=4) → (B, 32, 20, 20)
  ↓
Conv2d(32→64, k=4, s=2) → (B, 64, 9, 9)
  ↓
Conv2d(64→64, k=3, s=1) → (B, 64, 7, 7)
  ↓
ResidualBlock ×2 → (B, 64, 7, 7)
  ↓
Output: (B, 3136) [64×7×7 flattened]
```

### 3.2 LSTM Recurrente (`models/recurrent_module.py`)

**Propósito**: Proporcionar memoria temporal

**Características**:
- LSTM o GRU (configurable)
- Layer normalization
- Inicialización ortogonal de pesos

**Ventajas para Mario**:
- Recordar posiciones de enemigos
- Anticipar patrones de plataformas
- Contexto sobre events pasados

```python
Input: (B, 1, 3136)  # Sequence length = 1
  ↓
LSTM(input_dim=3136, hidden_dim=512, num_layers=1)
  ↓
Output: (B, 512), Hidden=(h,c)
```

### 3.3 Actor-Critic Network (`models/actor_critic.py`)

**Propósito**: Combinar política (actor) y value function (critic)

**Componentes**:
- CNN Backbone
- LSTM Layer
- Policy Head (logits de acciones)
- Value Head (estimación de value)

```python
class ActorCriticNetwork(nn.Module):
    def forward(obs, lstm_hidden) -> (logits, value, lstm_hidden)
```

### 3.4 PPO Agent (`agents/ppo_agent.py`)

**Propósito**: Implementar el algoritmo de entrenamiento

**Métodos principales**:
- `collect_rollout()`: Recolectar experiencias
- `update()`: Actualizar modelo con PPO
- `_compute_policy_loss()`: PPO clipping
- `_compute_value_loss()`: Value function loss

**Hiperparámetros clave**:
```python
clip_range = 0.1          # PPO clipping
entropy_coeff = 0.01      # Exploración
value_loss_coeff = 0.5    # Peso del value loss
gamma = 0.99              # Descuento
gae_lambda = 0.95         # GAE lambda
```

### 3.5 Rollout Buffer (`agents/memory.py`)

**Propósito**: Almacenar experiencias para actualización

**Almacena**:
- Observaciones
- Acciones
- Recompensas
- Values estimados
- Log probabilities
- Done flags

**Calcula**:
- Generalized Advantage Estimation (GAE)
- Returns (Gt = At + V(st))

## 4. Pipeline de Entrenamiento

### Paso 1: Recolección de Experiencias

```python
for step in range(ROLLOUT_STEPS):
    obs = env.get_state()

    with torch.no_grad():
        action, log_prob, value, lstm_hidden = model(obs, lstm_hidden)

    next_obs, reward, done, info = env.step(action)

    buffer.add(obs, action, reward, value, log_prob, done)
```

### Paso 2: Cálculo de Advantages

```python
advantages = []
gae = 0
for t in reversed(range(ROLLOUT_STEPS)):
    delta = rewards[t] + gamma * V(s_{t+1}) - V(s_t)
    gae = delta + gamma * gae_lambda * gae
    advantages.append(gae)

advantages.reverse()  # Ajustar orden
```

### Paso 3: Actualización PPO

```python
for epoch in range(EPOCHS_PER_UPDATE):
    for batch in buffer.get_batch(BATCH_SIZE):
        obs, actions, advantages, returns = batch

        # Forward
        logits, values = model(obs)
        new_log_probs = compute_log_probs(logits, actions)

        # PPO Loss
        ratio = exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-ε, 1+ε) * advantages
        policy_loss = -min(surr1, surr2).mean()

        # Value Loss
        value_loss = MSE(values, returns)

        # Total
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # Backward
        loss.backward()
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
```

## 5. Reward Shaping

El entorno retorna recompensas baseadas en:

```python
reward = 0

# Avance horizontal
if x_pos > last_x_pos:
    reward += (x_pos - last_x_pos) * scale_position

# Monedas
if coins > last_coins:
    reward += (coins - last_coins) * scale_coin

# Enemigos derrotados
if enemies > last_enemies:
    reward += (enemies - last_enemies) * scale_kill

# Penalizaciones
if idle_steps > 10:
    reward -= penalty_idle
if is_dead:
    reward -= penalty_death
```

## 6. Wrappers del Entorno

Transforman la entrada para facilitar el aprendizaje:

1. **FrameSkip**: Repetir acción 4 veces
2. **GrayscaleFrame**: RGB → Escala de grises
3. **ResizeFrame**: 256×240 → 84×84
4. **FrameStack**: Apilar últimos 4 frames
5. **NormalizeObservation**: Dividir por 255

## 7. Técnicas Avanzadas

### 7.1 Generalized Advantage Estimation (GAE)

Combina ventajas de n-step returns y baseline:

```python
Advantage = sum(λ^n * δ_t+n)
donde δ_t = r_t + γV(s_t+1) - V(s_t)
```

### 7.2 Mixed Precision Training

Usar FP16 en GPU para mejor rendimiento:

```python
with autocast(dtype=torch.float16):
    # Forward pass en FP16
    loss = model(...)
# Backward en FP32 (automático)
```

### 7.3 Learning Rate Decay

Reducir LR durante el entrenamiento:

```python
lr_new = lr_0 * (1 - progress * (1 - final_factor))
```

### 7.4 Reward Normalization

Normalizar recompensas acumulativas:

```python
returns_rms.update(batch_returns)
normalized_reward = reward / sqrt(returns_rms.var + ε)
```

## 8. Métricas de Entrenamiento

**Métricas principales a monitorear**:

- `episode_reward`: Recompensa total por episodio
- `policy_loss`: Pérdida de política (debe bajar)
- `value_loss`: Pérdida de value (debe bajar)
- `entropy`: Entropía de política (indicador de exploración)
- `approx_kl`: KL divergence aproximado (debe ser bajo)
- `max_x_pos`: Progreso máximo del agente (debe crecer)

## 9. Configuración por Defecto

```python
# Entorno
FRAME_SIZE = 84
FRAME_STACK = 4
FRAME_SKIP = 4

# CNN
CNN_FEATURES = [32, 64, 64]
CNN_USE_BATCH_NORM = True

# LSTM
LSTM_HIDDEN_DIM = 512
USE_LAYER_NORM = True

# PPO
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.1
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5

# Entrenamiento
BATCH_SIZE = 128
NUM_ENVS_PARALLEL = 16
ROLLOUT_STEPS = 512
EPOCHS_PER_UPDATE = 3
TOTAL_TIMESTEPS = 50_000_000

# Hardware
USE_GPU = True
USE_MIXED_PRECISION = True
```

## 10. Expansiones Futuras

- **Curiosity-driven Exploration (RND)**: Bonus por novedad
- **Prioritized Experience Replay**: Más énfasis en transiciones importantes
- **Multi-scale Rewards**: Diferentes scales para diferentes objetivos
- **Hierarchical RL**: Políticas para diferentes niveles
- **Transfer Learning**: Entrenar en un juego, transferir a otro

---

**Esta arquitectura proporciona un sistema profesional y escalable para entrenar agentes RL complejos.**
