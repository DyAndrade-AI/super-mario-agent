# Guía de Instalación y Setup

## 1. Requisitos del Sistema

- **Python**: 3.8 o superior
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **GPU**: Recomendado NVIDIA con CUDA 11.8+
- **Almacenamiento**: 20GB para checkpoints y logs

## 2. Instalación Paso a Paso

### 2.1 Clonar/Descargar Repositorio

```bash
cd super_mario
```

### 2.2 Crear Entorno Virtual

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.3 Actualizar pip

```bash
pip install --upgrade pip setuptools wheel
```

### 2.4 Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2.5 Verificar Instalación

```bash
# Verificar PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"

# Verificar Gymnasium
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"

# Verificar Mario Env
python -c "from gymnasium_super_mario_bros import SuperMarioBrosEnv; print('Mario env OK')"

# Verificar OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## 3. Configuración de CUDA (Opcional pero Recomendado)

### 3.1 Verificar NVIDIA GPU

```bash
nvidia-smi
```

Si no funciona, instalar NVIDIA drivers desde https://www.nvidia.com/Download/driverDetails.aspx

### 3.2 Instalar CUDA Toolkit

1. Ir a https://developer.nvidia.com/cuda-toolkit
2. Descargar versión 11.8 o superior
3. Instalar siguiendo las instrucciones
4. Verificar: `nvcc --version`

### 3.3 Instalar cuDNN

1. Ir a https://developer.nvidia.com/cudnn
2. Descargar compatible con tu CUDA
3. Seguir instrucciones de instalación

### 3.4 Reinstalar PyTorch con CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 4. Estructura de Directorios

El proyecto creará automáticamente:

```
super_mario/
├── outputs/
│   ├── checkpoints/      # Modelos guardados
│   ├── videos/          # Videos del agente
│   ├── logs/            # TensorBoard logs
│   └── plots/           # Gráficas
```

## 5. Configuración Inicial

### 5.1 Hiperparámetros

Editar `config/hyperparameters.py` según necesidad:

```python
# Para GPU potente (RTX 3090+)
NUM_ENVS_PARALLEL = 32
BATCH_SIZE = 256
USE_MIXED_PRECISION = True

# Para GPU normal (RTX 2080)
NUM_ENVS_PARALLEL = 16
BATCH_SIZE = 128
USE_MIXED_PRECISION = True

# Para CPU
NUM_ENVS_PARALLEL = 4
BATCH_SIZE = 64
USE_MIXED_PRECISION = False
USE_GPU = False
```

### 5.2 Logging

```python
# Para TensorBoard (recomendado)
USE_TENSORBOARD = True
USE_WANDB = False

# Para Weights & Biases
USE_WANDB = True
WANDB_PROJECT = "super_mario_rl"
WANDB_ENTITY = "tu_username"  # Cambiar
```

## 6. Primer Entrenamiento

### 6.1 Entrenamiento Básico (1 millón de pasos)

```bash
python main.py train \
  --world 1 \
  --stage 1 \
  --total-timesteps 1000000 \
  --seed 42
```

Esto debería:
- Crear el entorno
- Inicializar el modelo
- Comenzar a entrenar
- Guardar checkpoints cada 500k pasos
- Mostrar logs en `outputs/logs/`

### 6.2 Monitorear en TensorBoard

En otra terminal:
```bash
tensorboard --logdir outputs/logs
```

Abrir http://localhost:6006 en el navegador

### 6.3 Tiempo Esperado

- 1M pasos: ~2-3 horas en GPU RTX 3090
- 10M pasos: ~20-30 horas en GPU RTX 3090
- 50M pasos: ~100-150 horas en GPU RTX 3090

## 7. Solución de Problemas

### Problema: "No module named 'gymnasium_super_mario_bros'"

**Solución:**
```bash
pip install gymnasium-super-mario-bros
```

### Problema: "CUDA out of memory"

**Soluciones:**
1. Reducir `BATCH_SIZE`: `BATCH_SIZE = 64`
2. Reducir `NUM_ENVS_PARALLEL`: `NUM_ENVS_PARALLEL = 8`
3. Usar CPU: `USE_GPU = False`
4. Usar mixed precision: `USE_MIXED_PRECISION = True`

### Problema: "RuntimeError: Expected all tensors to be on the same device"

**Solución:**
```python
# En config/hyperparameters.py
USE_GPU = False  # Usar CPU temporalmente
```

### Problema: Entrenamiento muy lento

**Causas y soluciones:**
- CPU sin GPU: Comprar/acceder a GPU
- GPU saturada: Reducir batch size
- Pocos ambientes: Aumentar `NUM_ENVS_PARALLEL`

### Problema: Logs no aparecen en TensorBoard

```bash
# Limpiar caché
rm -rf outputs/logs

# Reiniciar entrenamiento
python main.py train --world 1 --stage 1 --total-timesteps 1000000
```

### Problema: Videos no se graban

```bash
# Verificar OpenCV
pip install --upgrade opencv-python

# Verificar permisos
chmod 777 outputs/videos/  # Linux/Mac
```

## 8. Optimización del Rendimiento

### 8.1 Para GPU

```python
# Máximo rendimiento (RTX 4090)
NUM_ENVS_PARALLEL = 64
BATCH_SIZE = 256
USE_MIXED_PRECISION = True
FRAME_SKIP = 4

# Máximo rendimiento (RTX 3090)
NUM_ENVS_PARALLEL = 32
BATCH_SIZE = 128
USE_MIXED_PRECISION = True

# Balance (RTX 2080)
NUM_ENVS_PARALLEL = 16
BATCH_SIZE = 128
USE_MIXED_PRECISION = True
```

### 8.2 Para CPU

```python
NUM_ENVS_PARALLEL = 2
BATCH_SIZE = 32
ROLLOUT_STEPS = 256
USE_MIXED_PRECISION = False
USE_GPU = False
```

### 8.3 Para Notebook/Colab

```python
# Google Colab GPU
NUM_ENVS_PARALLEL = 8
BATCH_SIZE = 64
ROLLOUT_STEPS = 256
USE_MIXED_PRECISION = True
```

## 9. Próximos Pasos

1. ✅ Instalación completada
2. ▶️ Ejecutar primer entrenamiento (ver sección 6.1)
3. 📊 Monitorear en TensorBoard
4. 🎮 Evaluar modelo cuando esté entrenado
5. 📝 Ajustar hiperparámetros según resultados

## 10. Recursos Útiles

- **PyTorch**: https://pytorch.org/
- **Gymnasium**: https://gymnasium.farama.org/
- **Super Mario Env**: https://github.com/Farama-Foundation/gym-super-mario-bros
- **TensorBoard**: https://www.tensorflow.org/tensorboard
- **Weights & Biases**: https://wandb.ai/

## 11. Contacto y Soporte

Si encuentras problemas:

1. Verificar logs: `outputs/logs/training.log`
2. Revisar README.md
3. Consultar documentación del código
4. Verificar GitHub issues

---

**Instalación completada exitosamente** ✅

Procede a ejecutar: `python main.py train --help`
