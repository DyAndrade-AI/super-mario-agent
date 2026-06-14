"""
Dueling Double DQN agent with prioritized replay and n-step targets.
"""
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from agents.memory import create_replay_buffer
from models.dueling_dqn import create_dueling_dqn


class DQNAgent:
    """Off-policy Q-learning agent for discrete Mario action spaces."""

    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        replay_buffer_size: int = 150_000,
        learning_starts: int = 20_000,
        gradient_steps: int = 128,
        target_update_interval: int = 10_000,
        target_soft_update_tau: float = 1.0,
        n_step: int = 3,
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_frames: int = 5_000_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_steps: int = 1_000_000,
        reward_scale: float = 1.0,
        reward_clip: Optional[float] = 10.0,
        clip_grad_norm: float = 10.0,
        use_mixed_precision: bool = True,
        device: Optional[torch.device] = None,
        **model_kwargs,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device or torch.device("cpu")
        self.num_actions = action_space.n

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.learning_starts = learning_starts
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.target_soft_update_tau = target_soft_update_tau
        self.n_step = max(1, n_step)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.clip_grad_norm = clip_grad_norm

        self.model = create_dueling_dqn(
            observation_space,
            action_space,
            **model_kwargs,
        ).to(self.device)
        self.target_model = create_dueling_dqn(
            observation_space,
            action_space,
            **model_kwargs,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            eps=1e-5,
            weight_decay=1e-5,
        )
        self.scheduler = None

        self.replay_buffer = create_replay_buffer(
            capacity=replay_buffer_size,
            obs_shape=observation_space.shape,
            device=self.device,
            alpha=priority_alpha,
            beta_start=priority_beta_start,
            beta_frames=priority_beta_frames,
        )

        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_mixed_precision else None

        self.current_obs = None
        self.n_step_buffers: List[deque] = []
        self.total_steps = 0
        self.update_count = 0
        self.optimizer_steps = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.last_collection_mean_reward = 0.0
        self.last_collection_total_reward = 0.0
        self.last_collection_completed_episodes = 0
        self.last_collection_episode_returns: List[float] = []
        self.last_collection_episode_lengths: List[int] = []
        self.last_action_entropy = 0.0
        self.last_collected_transitions = 0
        self._episode_reward_state = None
        self._episode_length_state = None

    def _prepare_obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        obs_tensor = torch.from_numpy(np.asarray(obs)).to(self.device, dtype=torch.float32)
        if obs_tensor.ndim == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        if obs_tensor.ndim == 4 and obs_tensor.shape[-1] in (1, 3, 4):
            obs_tensor = obs_tensor.permute(0, 3, 1, 2).contiguous()
        if obs_tensor.max() > 1.0:
            obs_tensor = obs_tensor / 255.0
        return obs_tensor

    def _num_envs_from_obs(self, obs: np.ndarray) -> int:
        return obs.shape[0] if obs.ndim == len(self.observation_space.shape) + 1 else 1

    @staticmethod
    def _as_batch(obs: np.ndarray, num_envs: int) -> np.ndarray:
        return obs if num_envs > 1 else np.expand_dims(obs, axis=0)

    @staticmethod
    def _info_list(info, num_envs: int) -> List[dict]:
        if isinstance(info, dict) and isinstance(info.get("all"), list):
            return info["all"]
        if isinstance(info, list):
            return info
        return [info if isinstance(info, dict) else {} for _ in range(num_envs)]

    def _epsilon(self) -> float:
        progress = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_actions(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        num_envs = self._num_envs_from_obs(obs)
        epsilon = 0.0 if evaluate else self._epsilon()

        was_training = self.model.training
        if evaluate:
            self.model.eval()
        else:
            self.model.train()
            if hasattr(self.model, "reset_noise"):
                self.model.reset_noise()

        with torch.no_grad():
            obs_tensor = self._prepare_obs_tensor(obs)
            q_values = self.model(obs_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy().astype(np.int64)

        if not evaluate and epsilon > 0.0:
            random_mask = np.random.random(size=num_envs) < epsilon
            if np.any(random_mask):
                actions[random_mask] = np.random.randint(
                    0,
                    self.num_actions,
                    size=int(random_mask.sum()),
                    dtype=np.int64,
                )

        if was_training:
            self.model.train()
        else:
            self.model.eval()
        return actions

    def _process_rewards(self, rewards: np.ndarray) -> np.ndarray:
        processed = rewards.astype(np.float32) * self.reward_scale
        if self.reward_clip is not None:
            processed = np.clip(processed, -self.reward_clip, self.reward_clip)
        return processed

    def _add_transition(
        self,
        env_idx: int,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        transition = (obs, int(action), float(reward), next_obs, bool(done))
        buffer = self.n_step_buffers[env_idx]
        buffer.append(transition)

        if len(buffer) < self.n_step and not done:
            return

        if done:
            while buffer:
                self._flush_n_step(buffer)
                buffer.popleft()
        else:
            self._flush_n_step(buffer)
            buffer.popleft()

    def _flush_n_step(self, buffer: deque):
        reward_sum = 0.0
        discount = 1.0
        first_obs, first_action, _, _, _ = buffer[0]
        last_next_obs = buffer[-1][3]
        last_done = buffer[-1][4]

        for _, _, reward, next_obs, done in buffer:
            reward_sum += discount * reward
            last_next_obs = next_obs
            last_done = done
            if done:
                break
            discount *= self.gamma

        self.replay_buffer.add(
            obs=first_obs,
            action=first_action,
            reward=reward_sum,
            next_obs=last_next_obs,
            done=last_done,
            discount=discount,
        )

    def collect_experience(self, env, progress_interval: int = 0) -> Tuple[float, int]:
        """Collects environment transitions into replay memory."""
        if self.current_obs is None:
            self.current_obs, _ = env.reset()

        obs_shape = self.observation_space.shape
        actual_num_envs = self._num_envs_from_obs(np.asarray(self.current_obs))

        if not self.n_step_buffers or len(self.n_step_buffers) != actual_num_envs:
            self.n_step_buffers = [deque(maxlen=self.n_step) for _ in range(actual_num_envs)]
            self._episode_reward_state = np.zeros(actual_num_envs, dtype=np.float32)
            self._episode_length_state = np.zeros(actual_num_envs, dtype=np.int32)

        collection_steps = getattr(self, "collect_steps", 1)
        collection_rewards = []
        collection_episode_returns = []
        collection_episode_lengths = []
        completed_episodes = 0
        action_counts = np.zeros(self.num_actions, dtype=np.int64)
        collected_transitions = 0

        for step in range(collection_steps):
            actions = self.select_actions(self.current_obs, evaluate=False)
            action_counts += np.bincount(actions, minlength=self.num_actions)

            action_to_env = int(actions[0]) if actual_num_envs == 1 else actions
            next_obs, reward, done, info = env.step(action_to_env)

            reward_array = np.asarray(reward, dtype=np.float32).reshape(actual_num_envs)
            done_array = np.asarray(done, dtype=np.bool_).reshape(actual_num_envs)
            processed_rewards = self._process_rewards(reward_array)
            collection_rewards.append(reward_array.copy())

            obs_batch = self._as_batch(np.asarray(self.current_obs), actual_num_envs)
            next_obs_batch = self._as_batch(np.asarray(next_obs), actual_num_envs)
            info_list = self._info_list(info, actual_num_envs)

            for env_idx in range(actual_num_envs):
                transition_next_obs = next_obs_batch[env_idx]
                if done_array[env_idx] and env_idx < len(info_list):
                    terminal_obs = info_list[env_idx].get("terminal_observation")
                    if terminal_obs is not None:
                        transition_next_obs = terminal_obs

                self._add_transition(
                    env_idx=env_idx,
                    obs=obs_batch[env_idx],
                    action=int(actions[env_idx]),
                    reward=float(processed_rewards[env_idx]),
                    next_obs=transition_next_obs,
                    done=bool(done_array[env_idx]),
                )

            self._episode_reward_state += reward_array
            self._episode_length_state += 1
            self.total_steps += actual_num_envs
            collected_transitions += actual_num_envs

            for env_idx, is_done in enumerate(done_array):
                if is_done:
                    episode_return = float(self._episode_reward_state[env_idx])
                    episode_length = int(self._episode_length_state[env_idx])
                    self.episode_returns.append(episode_return)
                    self.episode_lengths.append(episode_length)
                    collection_episode_returns.append(episode_return)
                    collection_episode_lengths.append(episode_length)
                    self._episode_reward_state[env_idx] = 0.0
                    self._episode_length_state[env_idx] = 0
                    completed_episodes += 1

            if actual_num_envs == 1 and done_array[0]:
                self.current_obs, _ = env.reset()
            else:
                self.current_obs = next_obs

            if progress_interval and (
                (step + 1) % progress_interval == 0 or step + 1 == collection_steps
            ):
                collected = (step + 1) * actual_num_envs
                total = collection_steps * actual_num_envs
                print(f"  Replay: {collected:,} / {total:,} transiciones", flush=True)

        rewards_flat = np.concatenate(collection_rewards) if collection_rewards else np.array([0.0])
        self.last_collection_mean_reward = float(np.mean(rewards_flat))
        self.last_collection_total_reward = float(np.sum(rewards_flat))
        self.last_collection_completed_episodes = completed_episodes
        self.last_collection_episode_returns = collection_episode_returns
        self.last_collection_episode_lengths = collection_episode_lengths
        self.last_collected_transitions = collected_transitions

        action_total = action_counts.sum()
        if action_total > 0:
            action_probs = action_counts[action_counts > 0] / action_total
            self.last_action_entropy = float(-np.sum(action_probs * np.log(action_probs + 1e-8)))
        else:
            self.last_action_entropy = 0.0

        if completed_episodes > 0 and self.episode_returns:
            avg_reward = float(np.mean(self.episode_returns[-100:]))
        else:
            avg_reward = self.last_collection_mean_reward

        return avg_reward, self.total_steps

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < max(self.learning_starts, self.batch_size):
            return {
                "q_loss": 0.0,
                "td_error": 0.0,
                "q_value": 0.0,
                "target_q_value": 0.0,
                "epsilon": self._epsilon(),
                "replay_size": len(self.replay_buffer),
                "optimizer_steps": 0,
                "priority_beta": self.replay_buffer.beta_by_frame(self.total_steps),
                "action_entropy": self.last_action_entropy,
            }

        self.model.train()
        metrics = {
            "q_loss": [],
            "td_error": [],
            "q_value": [],
            "target_q_value": [],
            "grad_norm": [],
        }
        beta = self.replay_buffer.beta_by_frame(self.total_steps)
        optimizer_steps = 0

        for _ in range(self.gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size, beta=beta)

            if self.use_mixed_precision:
                with autocast(dtype=torch.float16):
                    loss, td_errors, q_values, target_q = self._compute_dqn_loss(batch)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, td_errors, q_values, target_q = self._compute_dqn_loss(batch)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            self.replay_buffer.update_priorities(
                batch["indices"],
                td_errors.detach().abs().cpu().numpy() + self.replay_buffer.priority_eps,
            )

            self.optimizer_steps += 1
            optimizer_steps += 1
            self._maybe_update_target_network()

            if hasattr(self.model, "reset_noise"):
                self.model.reset_noise()
                self.target_model.reset_noise()

            metrics["q_loss"].append(float(loss.detach().cpu().item()))
            metrics["td_error"].append(float(td_errors.detach().abs().mean().cpu().item()))
            metrics["q_value"].append(float(q_values.detach().mean().cpu().item()))
            metrics["target_q_value"].append(float(target_q.detach().mean().cpu().item()))
            metrics["grad_norm"].append(float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm))

        self.update_count += 1
        averaged_metrics = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in metrics.items()
        }
        averaged_metrics.update({
            "epsilon": self._epsilon(),
            "replay_size": len(self.replay_buffer),
            "optimizer_steps": optimizer_steps,
            "priority_beta": beta,
            "action_entropy": self.last_action_entropy,
        })
        return averaged_metrics

    def _compute_dqn_loss(self, batch: Dict[str, torch.Tensor]):
        actions = batch["actions"].long().unsqueeze(1)
        rewards = batch["rewards"]
        dones = batch["dones"]
        discounts = batch["discounts"]
        weights = batch["weights"]

        q_values_all = self.model(batch["obs"])
        q_values = q_values_all.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(batch["next_obs"]).argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(batch["next_obs"]).gather(1, next_actions).squeeze(1)
            target_q = rewards + discounts * (1.0 - dones) * next_q_values

        td_errors = target_q - q_values
        elementwise_loss = F.smooth_l1_loss(q_values, target_q, reduction="none")
        loss = (weights * elementwise_loss).mean()
        return loss, td_errors, q_values, target_q

    def _maybe_update_target_network(self):
        if self.target_update_interval <= 0:
            self._soft_update_target()
            return

        if self.optimizer_steps % self.target_update_interval == 0:
            if self.target_soft_update_tau >= 1.0:
                self.target_model.load_state_dict(self.model.state_dict())
            else:
                self._soft_update_target()

    def _soft_update_target(self):
        tau = self.target_soft_update_tau
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def get_checkpoint_extra_state(self) -> Dict:
        return {
            "algorithm": "dueling_double_dqn",
            "target_model_state": self.target_model.state_dict(),
            "training_state_extra": {
                "optimizer_steps": self.optimizer_steps,
                "epsilon": self._epsilon(),
                "replay_size": len(self.replay_buffer),
            },
        }

    def save(self, path: Path):
        checkpoint = {
            "algorithm": "dueling_double_dqn",
            "model": self.model.state_dict(),
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
            "optimizer_steps": self.optimizer_steps,
        }
        torch.save(checkpoint, path)

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        model_state = checkpoint.get("model_state", checkpoint.get("model"))
        optimizer_state = checkpoint.get("optimizer_state", checkpoint.get("optimizer"))
        target_state = checkpoint.get("target_model_state")
        if model_state is None:
            raise KeyError("Checkpoint sin 'model_state' ni 'model'")

        self.model.load_state_dict(model_state)
        if target_state is not None:
            self.target_model.load_state_dict(target_state)
        else:
            self.target_model.load_state_dict(self.model.state_dict())

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        training_state = checkpoint.get("training_state", {})
        training_extra = checkpoint.get("training_state_extra", {})
        self.total_steps = checkpoint.get("total_steps", training_state.get("total_steps", 0))
        self.update_count = checkpoint.get("update_count", training_state.get("update_count", 0))
        self.optimizer_steps = checkpoint.get(
            "optimizer_steps",
            training_extra.get("optimizer_steps", training_state.get("optimizer_steps", 0)),
        )


def create_dqn_agent(
    observation_space,
    action_space,
    device: Optional[torch.device] = None,
    **kwargs,
) -> DQNAgent:
    return DQNAgent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **kwargs,
    )
