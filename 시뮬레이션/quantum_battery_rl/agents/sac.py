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


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 50000):
        self.capacity = int(capacity)
        self.state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((capacity,), dtype=np.float32)
        self.next_state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ):
        self.state_buffer[self.ptr] = np.asarray(state, dtype=np.float32)
        self.action_buffer[self.ptr] = np.asarray(action, dtype=np.float32)
        self.reward_buffer[self.ptr] = float(reward)
        self.next_state_buffer[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.done_buffer[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        states = self.state_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        next_states = self.next_state_buffer[indices]
        dones = self.done_buffer[indices]
        return states, actions, rewards, next_states, dones


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.mean_head = nn.Linear(32, action_dim)
        self.log_std_head = nn.Linear(32, action_dim)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        mean = self.mean_head(x)
        log_std_raw = self.softplus(self.log_std_head(x))
        log_std = torch.log(log_std_raw + 1e-8)
        log_std = torch.clamp(log_std, min=-5.0, max=0.0)
        return mean, log_std


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.q_head = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.q_head(x)


class SACAgent:
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        max_action: float = 0.25,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 50000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.target_critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(0.9, 0.999)
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=critic_lr, betas=(0.9, 0.999)
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=critic_lr, betas=(0.9, 0.999)
        )

        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(0.9, 0.999)
        )
        self.target_entropy = -float(action_dim)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_capacity)
        self._all_params = (
            list(self.actor.parameters())
            + list(self.critic1.parameters())
            + list(self.critic2.parameters())
            + [self.log_alpha]
        )

    def _gaussian_log_prob(
        self, action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        log_std = log_std.expand_as(mean)
        var = torch.exp(2.0 * log_std)
        log_2pi = math.log(2.0 * math.pi)
        return -0.5 * (
            ((action - mean) ** 2) / (var + 1e-8) + 2.0 * log_std + log_2pi
        ).sum(dim=-1)

    def _sample_action(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        tanh_action = torch.tanh(pre_tanh)
        action = tanh_action * self.max_action
        log_prob = self._gaussian_log_prob(pre_tanh, mean, log_std)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob, pre_tanh

    def _deterministic_action(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.actor(state)
        pre_tanh = mean
        tanh_action = torch.tanh(pre_tanh)
        action = tanh_action * self.max_action
        log_prob = self._gaussian_log_prob(pre_tanh, mean, log_std)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def act(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action_tensor, log_prob = self._deterministic_action(state_tensor)
            else:
                action_tensor, log_prob, _ = self._sample_action(state_tensor)
            q1 = self.critic1(state_tensor, action_tensor).squeeze(-1)
            q2 = self.critic2(state_tensor, action_tensor).squeeze(-1)
            q_value = torch.min(q1, q2)
        action_np = action_tensor.squeeze(0).cpu().numpy()
        return action_np, float(log_prob.item()), float(q_value.item())

    def update(self, batch_size: int = None) -> dict:
        if batch_size is None:
            batch_size = self.batch_size
        if self.replay_buffer.size < batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "alpha": float(self.log_alpha.exp().item()),
                "entropy": 0.0,
            }

        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self._sample_action(next_states_t)
            target_q1 = self.target_critic1(next_states_t, next_actions).squeeze(-1)
            target_q2 = self.target_critic2(next_states_t, next_actions).squeeze(-1)
            target_q = torch.min(target_q1, target_q2)
            alpha = self.log_alpha.exp()
            target_value = rewards_t + self.gamma * (1.0 - dones_t) * (
                target_q - alpha * next_log_probs
            )

        q1 = self.critic1(states_t, actions_t).squeeze(-1)
        q2 = self.critic2(states_t, actions_t).squeeze(-1)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        actions_pi, log_probs_pi, _ = self._sample_action(states_t)
        q1_pi = self.critic1(states_t, actions_pi).squeeze(-1)
        q2_pi = self.critic2(states_t, actions_pi).squeeze(-1)
        min_q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_probs_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        entropy = float((-log_probs_pi).mean().item())

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "entropy": entropy,
        }

    def add_to_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save(self, path: str):
        torch.save(
            {
                "actor_state": self.actor.state_dict(),
                "critic1_state": self.critic1.state_dict(),
                "critic2_state": self.critic2.state_dict(),
                "target_critic1_state": self.target_critic1.state_dict(),
                "target_critic2_state": self.target_critic2.state_dict(),
                "actor_optimizer_state": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state": self.critic2_optimizer.state_dict(),
                "alpha_optimizer_state": self.alpha_optimizer.state_dict(),
                "log_alpha": float(self.log_alpha.item()),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic1.load_state_dict(checkpoint["critic1_state"])
        self.critic2.load_state_dict(checkpoint["critic2_state"])
        if "target_critic1_state" in checkpoint:
            self.target_critic1.load_state_dict(checkpoint["target_critic1_state"])
        else:
            self.target_critic1.load_state_dict(self.critic1.state_dict())
        if "target_critic2_state" in checkpoint:
            self.target_critic2.load_state_dict(checkpoint["target_critic2_state"])
        else:
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state"])
        if "log_alpha" in checkpoint:
            log_alpha_tensor = torch.as_tensor(
                checkpoint["log_alpha"], dtype=torch.float32, device=self.device
            )
            self.log_alpha.data.copy_(log_alpha_tensor)

    def get_param_count(self) -> int:
        return int(sum(p.numel() for p in self._all_params if p.requires_grad))

    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.batch_size


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


def collect_sac_episodes(
    agent: SACAgent,
    envs: list,
    n_steps: int = 100,
    updates_per_step: int = 1,
    warmup_steps: int = 1000,
) -> list:
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

    total_steps = 0
    for _ in range(n_steps):
        actions = []
        log_probs = []
        values = []
        for state in states:
            action, log_prob, q_value = agent.act(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(q_value)

        for i, env in enumerate(envs):
            next_state, reward, done, _info = _unpack_step(env.step(actions[i]))
            traj = trajectories[i]
            traj["states"].append(states[i])
            traj["actions"].append(actions[i])
            traj["rewards"].append(float(reward))
            traj["dones"].append(float(done))
            traj["log_probs"].append(float(log_probs[i]))
            traj["values"].append(float(values[i]))

            agent.add_to_buffer(
                states[i],
                actions[i],
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
            total_steps += 1
            if total_steps >= warmup_steps:
                for _ in range(updates_per_step):
                    if agent.can_update():
                        agent.update()

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
    agent = SACAgent(state_dim=5, action_dim=2)
    print(f"Total params: {agent.get_param_count()}")

    state = np.random.randn(5).astype(np.float32)
    action, log_prob, q_value = agent.act(state)
    print(
        f"Action shape: {action.shape}, range: [{action.min():.3f}, {action.max():.3f}]"
    )
    print(f"Log prob: {log_prob:.3f}, Q value: {q_value:.3f}")

    for _ in range(agent.batch_size):
        s = np.random.randn(5).astype(np.float32)
        a, _, _ = agent.act(s)
        r = float(np.random.randn())
        ns = np.random.randn(5).astype(np.float32)
        d = float(random.random() < 0.1)
        agent.add_to_buffer(s, a, r, ns, d)

    update_info = agent.update()
    print(
        "Update: actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}, "
        "alpha={alpha:.4f}, entropy={entropy:.4f}".format(**update_info)
    )

    det_action_1, _, _ = agent.act(state, deterministic=True)
    det_action_2, _, _ = agent.act(state, deterministic=True)
    print(f"Deterministic: {np.allclose(det_action_1, det_action_2)}")
