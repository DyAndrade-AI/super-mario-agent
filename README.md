# Super Mario Bros PPO Agent

Deep reinforcement learning project for training a PPO agent to play Super Mario Bros. The agent uses a convolutional visual encoder, optional recurrent memory, reward shaping, vectorized environment execution, checkpointing, and evaluation tools.

## Overview

This repository implements an end-to-end training pipeline for Super Mario Bros using:

- Proximal Policy Optimization (PPO)
- CNN-based visual feature extraction
- LSTM memory for temporal context
- Frame preprocessing, frame skipping, frame stacking, and normalization
- Reward shaping for progress, coins, enemies, inactivity, death, and level completion
- Multiprocessing-based vectorized environments
- CUDA training with mixed precision when available
- Periodic checkpointing and automatic best-model replacement
- Evaluation and playback scripts

The default training target is World 1-1.

## Current Status

The current pipeline supports:

- Real `gym-super-mario-bros` environments through `nes-py`
- `SuperMarioBros-*-*-v3` with `SIMPLE_MOVEMENT` action space
- Parallel rollout collection through a custom `SubprocMarioVectorEnv`
- GPU training on CUDA devices
- Automatic saving of:
  - `outputs/checkpoints/best_model.pt`
  - `outputs/checkpoints/best_model_metadata.json`
  - `outputs/checkpoints/best_eval_model.pt`
  - `outputs/checkpoints/last_checkpoint.pt`
  - periodic `checkpoint_step_*.pt` files

Gym warnings about the old step API may appear because `gym-super-mario-bros` depends on older Gym APIs. The project includes compatibility wrappers for old and new step/reset formats.

## Repository Structure

```text
super_mario/
├── agents/
│   ├── memory.py              # Rollout buffer and GAE support
│   └── ppo_agent.py           # PPO implementation
├── config/
│   ├── config.py              # Project paths and training config helpers
│   └── hyperparameters.py     # Training, PPO, model, and device settings
├── env/
│   ├── mario_env.py           # Single and parallel environment interfaces
│   ├── mario_simulator.py     # Fallback simulator
│   ├── reward_shaping.py      # Reward shaping logic
│   └── wrappers.py            # Frame processing and multiprocessing vector env
├── models/
│   ├── actor_critic.py        # Actor-critic network
│   ├── cnn_backbone.py        # CNN feature extractor
│   ├── distributions.py       # Action distributions
│   └── recurrent_module.py    # LSTM/GRU modules
├── utils/
│   ├── checkpoint.py          # Checkpoint save/load utilities
│   ├── device.py              # CPU/GPU device selection
│   ├── logger.py              # Metrics logging
│   ├── metrics.py             # Metrics tracking
│   └── video_recorder.py      # Video recording
├── evaluate.py                # Evaluation script
├── play_agent.py              # Playback script
├── train.py                   # Main training script
├── requirements.txt
└── README.md
```

## Requirements

Recommended environment:

- Python 3.8 or newer
- NVIDIA GPU with CUDA support for training acceleration
- PyTorch with CUDA support
- `gym-super-mario-bros`
- `nes-py`
- OpenCV
- TensorBoard

Install dependencies:

```bash
pip install -r requirements.txt
```

If using Conda, activate your environment before running the commands:

```bash
conda activate dqn
```

## GPU Configuration

The training, evaluation, and playback scripts set:

```python
CUDA_VISIBLE_DEVICES=0
```

This is intentional for machines with multiple NVIDIA cards where only one card is compatible with the installed PyTorch/CUDA build. In the current setup this selects the RTX A4000 and hides unsupported Quadro P620 devices.

GPU usage is controlled in `config/hyperparameters.py`:

```python
USE_GPU = True
DEVICE_ID = 0
USE_MIXED_PRECISION = True
```

At startup, the training script prints the selected device. A correct CUDA run should show a line similar to:

```text
Dispositivo activo: cuda:0
Usando GPU: True
GPU Name: NVIDIA RTX A4000
```

## Training

Run default training:

```bash
python train.py
```

Run a specific world and stage:

```bash
python train.py --world 1 --stage 1 --total-timesteps 50000000
```

Resume from a checkpoint:

```bash
python train.py --resume outputs/checkpoints/last_checkpoint.pt
```

Resume from the best model:

```bash
python train.py --resume outputs/checkpoints/best_model.pt
```

## Checkpoints and Best Model Saving

The training loop saves the best registered model automatically.

Main files:

- `outputs/checkpoints/best_model.pt`
  - Replaced whenever the current training score improves.
  - Intended for quick playback or evaluation.

- `outputs/checkpoints/best_model_metadata.json`
  - Contains the step, timestamp, score source, and metrics for the best model.

- `outputs/checkpoints/best_eval_model.pt`
  - Replaced whenever periodic evaluation achieves a new best evaluation score.

- `outputs/checkpoints/last_checkpoint.pt`
  - Updated whenever a regular checkpoint is saved.

- `outputs/checkpoints/checkpoint_step_*.pt`
  - Periodic checkpoints kept for recovery.

Periodic checkpointing and evaluation are configured in `config/hyperparameters.py`:

```python
CHECKPOINT_FREQ = 500_000
EVAL_FREQ = 50_000
SAVE_VIDEO_FREQ = 1_000_000
```

The scheduler triggers when a threshold is crossed, not only when the step count exactly matches the frequency.

## Evaluation

Evaluate a trained checkpoint:

```bash
python evaluate.py outputs/checkpoints/best_model.pt --episodes 10
```

Evaluate the best evaluation model:

```bash
python evaluate.py outputs/checkpoints/best_eval_model.pt --episodes 10
```

## Playback

Run the trained agent:

```bash
python play_agent.py outputs/checkpoints/best_model.pt --episodes 1
```

Run and save video:

```bash
python play_agent.py outputs/checkpoints/best_model.pt --episodes 1 --save-video --fps 30
```

## Key Hyperparameters

Edit `config/hyperparameters.py` to change training behavior.

Environment:

```python
WORLD = 1
STAGE = 1
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_STACK = 4
FRAME_SKIP = 4
```

Model:

```python
CNN_FEATURES = [32, 64, 64]
USE_LSTM = True
LSTM_HIDDEN_DIM = 512
POLICY_HIDDEN_DIM = 512
VALUE_HIDDEN_DIM = 512
```

PPO:

```python
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.1
ENTROPY_COEFF = 0.01
```

Rollout and optimization:

```python
NUM_ENVS_PARALLEL = 16
ROLLOUT_STEPS = 512
BATCH_SIZE = 128
EPOCHS_PER_UPDATE = 3
TOTAL_TIMESTEPS = 50_000_000
```

## Parallel Environment Execution

The project uses a custom multiprocessing vector environment:

```text
SubprocMarioVectorEnv
```

It launches one Mario environment per process and returns batched observations, rewards, done flags, and info dictionaries.

Expected training output:

```text
[OK] VectorEnv paralelo creado con 16 procesos
```

If the system cannot create the parallel environment, the code falls back to sequential environments. Sequential execution is significantly slower.

## Monitoring

TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

Open:

```text
http://localhost:6006
```

Important training metrics:

- `mean_episode_reward`
- `mean_episode_length`
- `rollout_mean_reward`
- `rollout_total_reward`
- `rollout_completed_episodes`
- `rollout_steps_per_sec`
- `policy_loss`
- `value_loss`
- `entropy`
- `approx_kl`

## Troubleshooting

### Training is using CPU

Check `config/hyperparameters.py`:

```python
USE_GPU = True
```

Then run:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### CUDA or cuDNN errors with unsupported GPUs

If the machine has multiple GPUs and some are unsupported by the installed PyTorch build, restrict visible devices before importing PyTorch:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python train.py
```

The scripts already set this by default with `os.environ.setdefault`.

### Training is slow

Most rollout time is spent in the emulator, which runs on CPU. Confirm that parallel environments are active:

```text
[OK] VectorEnv paralelo creado con 16 procesos
```

If not, reduce or inspect:

```python
NUM_ENVS_PARALLEL
ROLLOUT_STEPS
```

### Best model is not updating

The best model is updated only when the selected score improves. Check:

```text
outputs/checkpoints/best_model_metadata.json
```

### Evaluation or playback fails to load a checkpoint

Use one of:

```bash
python evaluate.py outputs/checkpoints/best_model.pt --episodes 10
python play_agent.py outputs/checkpoints/best_model.pt --episodes 1
```

Both scripts support checkpoints that store weights under either `model` or `model_state`.

## References

- Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Playing Atari with Deep Reinforcement Learning: https://arxiv.org/abs/1312.5602
- Generalized Advantage Estimation: https://arxiv.org/abs/1506.02438
- Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385

## License

Educational project. Use, modify, and extend according to your academic or research needs.

## Version

- Version: 1.1.0
- Last updated: 2026-05-06
- Status: Active development
