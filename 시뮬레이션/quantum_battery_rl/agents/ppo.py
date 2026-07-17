import math
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.mean_head = nn.Linear(32, action_dim)
        self.log_std_head = nn.Linear(32, 1)
        self.softplus = nn.Softplus()
        self.max_action = max_action

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        mean = torch.tanh(self.mean_head(x)) * self.max_action
        log_std_raw = self.softplus(self.log_std_head(x))
        log_std = torch.log(log_std_raw + 1e-8)
        log_std = torch.clamp(log_std, min=-5.0, max=0.0)
        return mean, log_std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.value_head(x)


class PPOAgent:
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        max_action: float = 0.25,
        lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        entropy_decay: float = 0.99,
        n_envs: int = 4,
        ppo_epochs: int = 4,
        batch_size: int = 32,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff_base = entropy_coeff
        self.entropy_coeff = entropy_coeff
        self.entropy_decay = entropy_decay
        self.n_envs = n_envs
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.policy = PolicyNetwork(state_dim, action_dim, max_action).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self._all_params = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = torch.optim.Adam(self._all_params, lr=lr, betas=(0.9, 0.999))

    def _gaussian_log_prob(
        self, action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        log_std = log_std.expand_as(mean)
        var = torch.exp(2.0 * log_std)
        log_2pi = math.log(2.0 * math.pi)
        return -0.5 * (
            ((action - mean) ** 2) / (var + 1e-8) + 2.0 * log_std + log_2pi
        ).sum(dim=-1)

    def _gaussian_entropy(self, log_std: torch.Tensor) -> torch.Tensor:
        log_std = log_std.expand(-1, self.action_dim)
        return 0.5 * (
            2.0 * log_std + math.log(2.0 * math.e * math.pi)
        ).sum(dim=-1)

    def _value_from_state(self, state: np.ndarray) -> float:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        was_training = self.value.training
        self.value.eval()
        with torch.no_grad():
            value = self.value(state_tensor).squeeze(-1)
        if was_training:
            self.value.train()
        return float(value[-1].item())

    def act(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        self.policy.eval()
        self.value.eval()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.policy(state_tensor)
            value = self.value(state_tensor).squeeze(-1)
            if deterministic:
                action = mean
            else:
                std = torch.exp(log_std)
                action = mean + std * torch.randn_like(mean)
            action = torch.clamp(action, -self.max_action, self.max_action)
            log_prob = self._gaussian_log_prob(action, mean, log_std)
        action_np = action.squeeze(0).cpu().numpy()
        return action_np, float(log_prob.item()), float(value.item())

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, trajectories: List[dict]) -> dict:
        self.policy.train()
        self.value.train()

        states_list = []
        actions_list = []
        log_probs_list = []
        advantages_list = []
        returns_list = []

        for traj in trajectories:
            rewards = np.asarray(traj["rewards"], dtype=np.float32)
            values = np.asarray(traj["values"], dtype=np.float32)
            dones = np.asarray(traj["dones"], dtype=np.float32)

            if "next_value" in traj:
                next_value = float(traj["next_value"])
            elif "next_state" in traj:
                next_value = self._value_from_state(traj["next_state"])
            elif "next_states" in traj and len(traj["next_states"]) > 0:
                next_value = self._value_from_state(traj["next_states"][-1])
            else:
                if len(values) == 0:
                    next_value = 0.0
                elif len(dones) > 0 and dones[-1] > 0.5:
                    next_value = 0.0
                else:
                    next_value = float(values[-1])

            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            advantages_list.append(advantages)
            returns_list.append(returns)
            states_list.append(np.asarray(traj["states"], dtype=np.float32))
            actions_list.append(np.asarray(traj["actions"], dtype=np.float32))
            log_probs_list.append(np.asarray(traj["log_probs"], dtype=np.float32))

        if len(states_list) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "total_loss": 0.0,
            }

        states = np.concatenate(states_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)
        old_log_probs = np.concatenate(log_probs_list, axis=0)
        advantages = np.concatenate(advantages_list, axis=0)
        returns = np.concatenate(returns_list, axis=0)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(
            old_log_probs, dtype=torch.float32, device=self.device
        )
        advantages_t = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        policy_losses = []
        value_losses = []
        entropies = []
        total_losses = []

        num_samples = states_t.shape[0]
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                mean, log_std = self.policy(batch_states)
                log_probs = self._gaussian_log_prob(
                    batch_actions, mean, log_std
                )
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    )
                    * batch_advantages
                )
                policy_loss = torch.min(surr1, surr2).mean()

                values_pred = self.value(batch_states).squeeze(-1)
                mse = torch.mean((values_pred - batch_returns) ** 2)
                value_loss = 0.5 * mse

                entropy = self._gaussian_entropy(log_std).mean()
                total_loss = -policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._all_params, self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                total_losses.append(total_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "total_loss": float(np.mean(total_losses)),
        }

    def save(self, path: str):
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "value_state": self.value.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "entropy_coeff": self.entropy_coeff,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state"])
        self.value.load_state_dict(checkpoint["value_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "entropy_coeff" in checkpoint:
            self.entropy_coeff = float(checkpoint["entropy_coeff"])

    def get_param_count(self) -> int:
        return int(sum(p.numel() for p in self._all_params if p.requires_grad))

    def set_entropy_coeff(self, episode: int):
        self.entropy_coeff = self.entropy_coeff_base * (self.entropy_decay ** episode)


def _unpack_reset(reset_result):
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _unpack_step(step_result):
    if not isinstance(step_result, tuple):
        raise ValueError("env.step must return a tuple")
    if len(step_result) == 4:
        next_state, reward, done, info = step_result
    elif len(step_result) == 5:
        next_state, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
    elif len(step_result) == 3:
        next_state, reward, done = step_result
        info = {}
    else:
        raise ValueError("env.step returned an unsupported tuple size")
    return next_state, reward, done, info


def collect_trajectories(agent, envs, n_steps: int = 100):
    trajectories = []
    states = []

    for env in envs:
        reset_result = env.reset()
        state = _unpack_reset(reset_result)
        states.append(np.asarray(state, dtype=np.float32))
        trajectories.append(
            {
                "states": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "log_probs": [],
                "values": [],
            }
        )

    for _ in range(n_steps):
        actions = []
        log_probs = []
        values = []
        for state in states:
            action, log_prob, value = agent.act(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

        for i, env in enumerate(envs):
            next_state, reward, done, _info = _unpack_step(env.step(actions[i]))
            traj = trajectories[i]
            traj["states"].append(states[i])
            traj["actions"].append(actions[i])
            traj["rewards"].append(float(reward))
            traj["dones"].append(float(done))
            traj["log_probs"].append(float(log_probs[i]))
            traj["values"].append(float(values[i]))

            if done:
                reset_result = env.reset()
                next_state = _unpack_reset(reset_result)
            states[i] = np.asarray(next_state, dtype=np.float32)

    for i, traj in enumerate(trajectories):
        traj["states"] = np.asarray(traj["states"], dtype=np.float32)
        traj["actions"] = np.asarray(traj["actions"], dtype=np.float32)
        traj["rewards"] = np.asarray(traj["rewards"], dtype=np.float32)
        traj["dones"] = np.asarray(traj["dones"], dtype=np.float32)
        traj["log_probs"] = np.asarray(traj["log_probs"], dtype=np.float32)
        traj["values"] = np.asarray(traj["values"], dtype=np.float32)
        traj["next_state"] = np.asarray(states[i], dtype=np.float32)

    return trajectories


if __name__ == "__main__":
    agent = PPOAgent(state_dim=5, action_dim=2)
    print(f"Total params: {agent.get_param_count()}")

    state = np.random.randn(5).astype(np.float32)
    action, log_prob, value = agent.act(state)
    print(
        f"Action shape: {action.shape}, range: [{action.min():.3f}, {action.max():.3f}]"
    )
    print(f"Log prob: {log_prob:.3f}, Value: {value:.3f}")

    set_seed(42)
    a1, _, _ = agent.act(state)
    set_seed(42)
    a2, _, _ = agent.act(state)
    print(f"Deterministic: {np.allclose(a1, a2)}")
