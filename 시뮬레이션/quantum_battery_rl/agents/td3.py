import math
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sac import ReplayBuffer, _unpack_reset, _unpack_step


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.action_head = nn.Linear(32, action_dim)
        self.max_action = max_action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        action = torch.tanh(self.action_head(x)) * self.max_action
        return action


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


class TD3Agent:
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        max_action: float = 0.25,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        buffer_capacity: int = 50000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.target_actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.target_critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
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

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_capacity)
        self._all_params = (
            list(self.actor.parameters())
            + list(self.critic1.parameters())
            + list(self.critic2.parameters())
        )
        self._update_step = 0

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
            action_tensor = self.actor(state_tensor)
            if not deterministic:
                noise = torch.randn_like(action_tensor) * (
                    self.exploration_noise * self.max_action
                )
                action_tensor = torch.clamp(
                    action_tensor + noise, -self.max_action, self.max_action
                )
            q1 = self.critic1(state_tensor, action_tensor).squeeze(-1)
            q2 = self.critic2(state_tensor, action_tensor).squeeze(-1)
            q_value = torch.min(q1, q2)
        action_np = action_tensor.squeeze(0).cpu().numpy()
        return action_np, 0.0, float(q_value.item())

    def update(self, batch_size: int = None) -> dict:
        if batch_size is None:
            batch_size = self.batch_size
        if self.replay_buffer.size < batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "policy_updated": False,
            }

        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self._update_step += 1

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
            target_actions = self.target_actor(next_states_t)
            noise = torch.randn_like(target_actions) * (
                self.policy_noise * self.max_action
            )
            noise = torch.clamp(
                noise,
                -self.noise_clip * self.max_action,
                self.noise_clip * self.max_action,
            )
            target_actions = torch.clamp(
                target_actions + noise, -self.max_action, self.max_action
            )
            target_q1 = self.target_critic1(next_states_t, target_actions).squeeze(-1)
            target_q2 = self.target_critic2(next_states_t, target_actions).squeeze(-1)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards_t + self.gamma * (1.0 - dones_t) * target_q

        q1 = self.critic1(states_t, actions_t).squeeze(-1)
        q2 = self.critic2(states_t, actions_t).squeeze(-1)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        actor_loss = 0.0
        policy_updated = False
        if self._update_step % self.policy_delay == 0:
            actions_pi = self.actor(states_t)
            actor_loss_t = -self.critic1(states_t, actions_pi).squeeze(-1).mean()
            self.actor_optimizer.zero_grad()
            actor_loss_t.backward()
            self.actor_optimizer.step()
            actor_loss = float(actor_loss_t.item())
            policy_updated = True

            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

        return {
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss.item()),
            "policy_updated": policy_updated,
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
                "target_actor_state": self.target_actor.state_dict(),
                "critic1_state": self.critic1.state_dict(),
                "critic2_state": self.critic2.state_dict(),
                "target_critic1_state": self.target_critic1.state_dict(),
                "target_critic2_state": self.target_critic2.state_dict(),
                "actor_optimizer_state": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state": self.critic2_optimizer.state_dict(),
                "update_step": self._update_step,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic1.load_state_dict(checkpoint["critic1_state"])
        self.critic2.load_state_dict(checkpoint["critic2_state"])
        if "target_actor_state" in checkpoint:
            self.target_actor.load_state_dict(checkpoint["target_actor_state"])
        else:
            self.target_actor.load_state_dict(self.actor.state_dict())
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
        if "update_step" in checkpoint:
            self._update_step = int(checkpoint["update_step"])

    def get_param_count(self) -> int:
        return int(sum(p.numel() for p in self._all_params if p.requires_grad))

    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.batch_size


def collect_td3_episodes(
    agent: TD3Agent,
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
    agent = TD3Agent(state_dim=5, action_dim=2)
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

    params_before = [p.detach().clone() for p in agent.actor.parameters()]
    update_info_1 = agent.update()
    params_after = [p.detach().clone() for p in agent.actor.parameters()]
    unchanged_1 = all(
        torch.allclose(a, b) for a, b in zip(params_before, params_after)
    )

    params_before = [p.detach().clone() for p in agent.actor.parameters()]
    update_info_2 = agent.update()
    params_after = [p.detach().clone() for p in agent.actor.parameters()]
    changed_2 = any(
        not torch.allclose(a, b) for a, b in zip(params_before, params_after)
    )

    print(
        "Update 1: actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}, "
        "policy_updated={policy_updated}".format(**update_info_1)
    )
    print(
        "Update 2: actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}, "
        "policy_updated={policy_updated}".format(**update_info_2)
    )
    print(f"Policy delay check: step1_unchanged={unchanged_1}, step2_changed={changed_2}")

    det_action_1, _, _ = agent.act(state, deterministic=True)
    det_action_2, _, _ = agent.act(state, deterministic=True)
    print(f"Deterministic: {np.allclose(det_action_1, det_action_2)}")
