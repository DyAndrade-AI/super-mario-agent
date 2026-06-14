# Arquitectura del Proyecto

## Vision General

El proyecto entrena un agente off-policy de Q-learning para Super Mario Bros:

- Algoritmo: Dueling Double DQN
- Replay: Prioritized Experience Replay
- Targets: n-step returns + target network
- Vision: CNN residual sobre frames 84x84 apilados
- Acciones: espacio discreto `SIMPLE_MOVEMENT`
- Ejecucion: entornos paralelos para recolectar experiencia

## Flujo de Datos

```text
Super Mario Env
  -> wrappers de observacion
  -> obs (84, 84, 4)
  -> Q-network online
  -> accion epsilon-greedy / greedy
  -> transicion n-step
  -> replay priorizado
  -> mini-batch
  -> Double DQN target
  -> Huber TD loss
  -> update online network
  -> sync target network
```

## Q-network

`models/dueling_dqn.py` define:

```text
Input: (B, 4, 84, 84)
  -> CNNBackbone residual
  -> Linear hidden
  -> Value stream: V(s)
  -> Advantage stream: A(s, a)
  -> Q(s, a) = V(s) + A(s, a) - mean(A)
```

La red puede usar `NoisyLinear` en los heads para exploracion parametrica durante entrenamiento.

## Agente

`agents/dqn_agent.py` implementa:

- Red online y red target
- Double DQN: la red online elige `argmax_a Q_online(s', a)` y la red target evalua ese valor
- Replay priorizado con importance-sampling weights
- n-step returns
- Epsilon schedule
- Clipping de gradiente
- Mixed precision en CUDA

La perdida por batch es:

```text
target = r_n + gamma^n * (1 - done) * Q_target(s', argmax Q_online(s'))
loss = weighted_smooth_l1(Q_online(s, a), target)
```

## Entrenamiento

`train.py` alterna:

1. Recolectar `COLLECT_STEPS * NUM_ENVS_PARALLEL` transiciones.
2. Guardar transiciones procesadas en replay.
3. Ejecutar `GRADIENT_STEPS` updates TD.
4. Actualizar prioridades con `abs(td_error)`.
5. Sincronizar target network cada `TARGET_UPDATE_INTERVAL`.
6. Guardar best/checkpoints y evaluar periodicamente.

## Evaluacion y Playback

`evaluate.py` y `play_agent.py` cargan la Q-network y usan politica greedy:

```python
action = q_values.argmax(dim=1)
```

No hay exploracion durante evaluacion.
