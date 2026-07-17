import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from quantum_battery_rl.agents.baselines import (
    BangBangAgent,
    GRAPEController,
    RandomAgent,
    SinusoidalAgent,
)
from quantum_battery_rl.agents.ppo import PPOAgent, collect_trajectories, set_seed
from quantum_battery_rl.agents.sac import SACAgent, collect_sac_episodes
from quantum_battery_rl.agents.td3 import TD3Agent, collect_td3_episodes
from quantum_battery_rl.env.lindblad_env import LindbladBatteryEnv, NOISE_CONFIGS


NOISE_NAME_MAP = {
    "ideal": "Ideal",
    "low": "Low",
    "ibm": "IBM-like",
    "ibm-like": "IBM-like",
    "high": "High",
    "extreme": "Extreme",
}
NOISE_ORDER = ["ideal", "low", "ibm", "high", "extreme"]
ALGO_LABEL = "PPO"
OFFPOLICY_INTERRUPTED = False


class TrackingEnv:
    def __init__(self, env: LindbladBatteryEnv) -> None:
        self.env = env
        self.last_ergotropy: Optional[float] = None

    def reset(self, seed: Optional[int] = None):
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done:
            if isinstance(info, dict) and "ergotropy" in info:
                self.last_ergotropy = float(info["ergotropy"])
            else:
                self.last_ergotropy = float(self.env.get_ergotropy())
        return obs, reward, done, truncated, info

    def get_ergotropy(self) -> float:
        return self.env.get_ergotropy()

    def __getattr__(self, name):
        return getattr(self.env, name)


def parse_seeds(seeds_arg: str) -> List[int]:
    seeds: List[int] = []
    for token in seeds_arg.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def parse_noise_arg(noise_arg: str) -> List[str]:
    token = noise_arg.strip().lower()
    if token == "all":
        keys = NOISE_ORDER
    else:
        keys = [t.strip().lower() for t in noise_arg.split(",") if t.strip()]
    resolved = []
    for key in keys:
        if key in NOISE_NAME_MAP:
            resolved.append(NOISE_NAME_MAP[key])
        elif key in NOISE_CONFIGS:
            resolved.append(key)
        else:
            valid = ", ".join(sorted(NOISE_NAME_MAP.keys()))
            raise ValueError(f"Unknown noise '{key}'. Valid: {valid}, all")
    return resolved


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    curves = output_dir / "curves"
    checkpoints.mkdir(parents=True, exist_ok=True)
    curves.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "checkpoints": checkpoints, "curves": curves}


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _evaluate_agent(agent, T1: float, T2: float, n_episodes: int, seed: int) -> float:
    ergotropies: List[float] = []
    for i in range(n_episodes):
        env = LindbladBatteryEnv(T1=T1, T2=T2)
        state = env.reset(seed=seed + i)
        done = False
        while not done:
            action, _log_prob, _value = agent.act(state, deterministic=True)
            state, _reward, done, _truncated, _info = env.step(action)
        ergotropies.append(float(env.get_ergotropy()))
    return _mean(ergotropies)


def _run_baseline_episode(agent, env: LindbladBatteryEnv, seed: int) -> float:
    state = env.reset(seed=seed)
    agent.reset(seed=seed)
    done = False
    while not done:
        action = agent.act(state)
        state, _reward, done, _truncated, _info = env.step(action)
    return float(env.get_ergotropy())


def _baseline_seed_list(seeds: List[int], target: int = 10) -> List[int]:
    if len(seeds) >= target:
        return list(seeds[:target])
    baseline = list(seeds)
    next_seed = baseline[-1] + 1 if baseline else 42
    while len(baseline) < target:
        baseline.append(int(next_seed))
        next_seed += 1
    return baseline


def _run_baselines(
    T1: float,
    T2: float,
    seeds: List[int],
    skip_grape: bool,
) -> Dict[str, Dict[str, List[float]]]:
    baseline_seeds = _baseline_seed_list(seeds, target=10)
    results: Dict[str, Dict[str, List[float]]] = {
        "BangBang": {"seed_list": baseline_seeds, "seed_values": []},
        "Sinusoidal": {"seed_list": baseline_seeds, "seed_values": []},
        "Random": {"seed_list": baseline_seeds, "seed_values": []},
        "GRAPE": {"seed_list": [], "seed_values": []},
    }

    for seed in baseline_seeds:
        env = LindbladBatteryEnv(T1=T1, T2=T2)
        agent = BangBangAgent(seed=seed, T1=T1, T2=T2)
        results["BangBang"]["seed_values"].append(_run_baseline_episode(agent, env, seed))

    for seed in baseline_seeds:
        env = LindbladBatteryEnv(T1=T1, T2=T2)
        agent = SinusoidalAgent(seed=seed, T1=T1, T2=T2)
        results["Sinusoidal"]["seed_values"].append(_run_baseline_episode(agent, env, seed))

    for seed in baseline_seeds:
        env = LindbladBatteryEnv(T1=T1, T2=T2)
        agent = RandomAgent(seed=seed)
        results["Random"]["seed_values"].append(_run_baseline_episode(agent, env, seed))

    if not skip_grape:
        grape_seed = baseline_seeds[0] if baseline_seeds else 42
        grape = GRAPEController(n_steps=100, n_iters=200, seed=grape_seed)
        ergotropy, _pulse = grape.optimize(T1=T1, T2=T2)
        results["GRAPE"]["seed_list"] = [grape_seed]
        results["GRAPE"]["seed_values"] = [float(ergotropy)]

    return results


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _print_summary_table(rows: List[Dict[str, Optional[float]]]) -> None:
    if not rows:
        return
    algo_label = ALGO_LABEL.upper()
    header = [
        "Noise",
        algo_label,
        "BangBang",
        "Sinusoidal",
        "Random",
        "GRAPE",
    ]
    widths = [max(len(header[0]), max(len(row["noise"]) for row in rows))]
    widths.extend([len(h) for h in header[1:]])
    print("\nFinal summary (mean ergotropy)")
    print(
        f"{header[0]:<{widths[0]}}  {header[1]:>{widths[1]}}  {header[2]:>{widths[2]}}  "
        f"{header[3]:>{widths[3]}}  {header[4]:>{widths[4]}}  {header[5]:>{widths[5]}}"
    )
    for row in rows:
        print(
            f"{row['noise']:<{widths[0]}}  "
            f"{_format_value(row['ppo']):>{widths[1]}}  "
            f"{_format_value(row['BangBang']):>{widths[2]}}  "
            f"{_format_value(row['Sinusoidal']):>{widths[3]}}  "
            f"{_format_value(row['Random']):>{widths[4]}}  "
            f"{_format_value(row['GRAPE']):>{widths[5]}}"
        )


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _train_offpolicy(
    algo: str,
    noise_name: str,
    T1: float,
    T2: float,
    seed: int,
    episodes: int,
    output_dirs: Dict[str, Path],
) -> Tuple[float, List[float], List[float]]:
    global OFFPOLICY_INTERRUPTED
    OFFPOLICY_INTERRUPTED = False
    algo = algo.lower()
    set_seed(seed)

    collect_fn: Callable[..., list]
    if algo == "sac":
        agent = SACAgent(state_dim=5, action_dim=2, max_action=0.25, device="cpu")
        collect_fn = collect_sac_episodes
    elif algo == "td3":
        agent = TD3Agent(state_dim=5, action_dim=2, max_action=0.25, device="cpu")
        collect_fn = collect_td3_episodes
    else:
        raise ValueError(f"Unsupported algo '{algo}'")

    envs = [TrackingEnv(LindbladBatteryEnv(T1=T1, T2=T2)) for _ in range(4)]
    episode_ergotropies: List[float] = []
    episode_rewards: List[float] = []
    convergence_episode: Optional[int] = None
    warmup_steps = 1000
    updates_per_step = 1
    steps_per_episode = 100 * len(envs)
    total_steps = 0
    curves_dir = output_dirs["curves"]
    checkpoints_dir = output_dirs["checkpoints"]
    interrupted = False

    try:
        for episode in range(episodes):
            for env in envs:
                env.last_ergotropy = None
            warmup_remaining = max(warmup_steps - total_steps, 0)
            trajectories = collect_fn(
                agent,
                envs,
                n_steps=100,
                updates_per_step=updates_per_step,
                warmup_steps=warmup_remaining,
            )
            total_steps += steps_per_episode

            ergotropy_values = [
                env.last_ergotropy
                if env.last_ergotropy is not None
                else env.get_ergotropy()
                for env in envs
            ]
            mean_ergotropy = float(np.mean(ergotropy_values))
            mean_reward = float(
                np.mean([float(np.sum(traj["rewards"])) for traj in trajectories])
            )
            episode_ergotropies.append(mean_ergotropy)
            episode_rewards.append(mean_reward)

            if convergence_episode is None and len(episode_ergotropies) >= 10:
                recent = float(np.mean(episode_ergotropies[-5:]))
                previous = float(np.mean(episode_ergotropies[-10:-5]))
                if abs(recent - previous) < 0.01:
                    convergence_episode = episode + 1

            if (episode + 1) % 10 == 0 or episode == 0 or episode + 1 == episodes:
                print(
                    f"[{noise_name}][seed {seed}] "
                    f"Episode {episode + 1}/{episodes} "
                    f"erg={mean_ergotropy:.3f} "
                    f"reward={mean_reward:.3f}"
                )
    except KeyboardInterrupt:
        interrupted = True
        OFFPOLICY_INTERRUPTED = True
        print("\nKeyboard interrupt detected. Saving partial results...")

    if convergence_episode is None:
        convergence_episode = len(episode_ergotropies) if episode_ergotropies else 0

    final_ergotropy = 0.0
    if episode_ergotropies:
        if not interrupted:
            final_ergotropy = _evaluate_agent(
                agent, T1=T1, T2=T2, n_episodes=10, seed=seed + 1000
            )
        else:
            final_ergotropy = float(episode_ergotropies[-1])

        curve_payload = {
            "seed": int(seed),
            "noise": str(noise_name),
            "episodes": [float(x) for x in episode_ergotropies],
            "rewards": [float(x) for x in episode_rewards],
            "final_ergotropy": float(final_ergotropy),
            "convergence_episode": int(convergence_episode),
        }
        curve_path = curves_dir / f"seed_{seed}_{noise_name}.json"
        _save_json(curve_path, curve_payload)

        checkpoint_path = checkpoints_dir / f"seed_{seed}_{noise_name}.pt"
        agent.save(str(checkpoint_path))

    return float(final_ergotropy), episode_ergotropies, episode_rewards


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PPO on Lindblad battery env.")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac", "td3"], help="RL algorithm")
    parser.add_argument("--noise", default="ibm", help="Noise config name or list")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per seed")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--skip-grape", action="store_true", help="Skip GRAPE baseline")
    args = parser.parse_args()

    noise_names = parse_noise_arg(args.noise)
    seeds = parse_seeds(args.seeds)
    output_dirs = ensure_output_dirs(Path(args.output))

    summary_rows: List[Dict[str, Optional[float]]] = []
    interrupted = False

    if args.algo != "ppo":
        global ALGO_LABEL
        global OFFPOLICY_INTERRUPTED
        ALGO_LABEL = args.algo.upper()
        OFFPOLICY_INTERRUPTED = False

        for noise_name in noise_names:
            config = NOISE_CONFIGS[noise_name]
            T1 = float(config["T1"])
            T2 = float(config["T2"])
            print(f"\nTraining {ALGO_LABEL} for noise={noise_name} (T1={T1}, T2={T2})")

            ppo_seed_values: List[float] = []
            ppo_seed_list: List[int] = []

            for seed in seeds:
                final_ergotropy, episode_ergotropies, _episode_rewards = _train_offpolicy(
                    args.algo, noise_name, T1, T2, seed, args.episodes, output_dirs
                )

                if episode_ergotropies:
                    ppo_seed_values.append(float(final_ergotropy))
                    ppo_seed_list.append(int(seed))

                if OFFPOLICY_INTERRUPTED:
                    interrupted = True
                    break

            baseline_results: Dict[str, Dict[str, List[float]]] = {
                "BangBang": {"seed_list": [], "seed_values": []},
                "Sinusoidal": {"seed_list": [], "seed_values": []},
                "Random": {"seed_list": [], "seed_values": []},
                "GRAPE": {"seed_list": [], "seed_values": []},
            }
            if not interrupted:
                baseline_results = _run_baselines(
                    T1=T1, T2=T2, seeds=seeds, skip_grape=args.skip_grape
                )

            summary_payload = {
                "noise_config": {"name": str(noise_name), "T1": T1, "T2": T2},
                "ppo": {
                    "seed_list": ppo_seed_list,
                    "seed_values": ppo_seed_values,
                    "episodes_per_seed": int(args.episodes),
                },
                "baselines": baseline_results,
            }
            summary_path = output_dirs["root"] / f"training_summary_{noise_name}.json"
            _save_json(summary_path, summary_payload)

            summary_rows.append(
                {
                    "noise": noise_name,
                    "ppo": _mean(ppo_seed_values) if ppo_seed_values else None,
                    "BangBang": _mean(baseline_results["BangBang"]["seed_values"])
                    if baseline_results["BangBang"]["seed_values"]
                    else None,
                    "Sinusoidal": _mean(baseline_results["Sinusoidal"]["seed_values"])
                    if baseline_results["Sinusoidal"]["seed_values"]
                    else None,
                    "Random": _mean(baseline_results["Random"]["seed_values"])
                    if baseline_results["Random"]["seed_values"]
                    else None,
                    "GRAPE": _mean(baseline_results["GRAPE"]["seed_values"])
                    if baseline_results["GRAPE"]["seed_values"]
                    else None,
                }
            )

            if interrupted:
                break

        _print_summary_table(summary_rows)
        return 0

    for noise_name in noise_names:
        config = NOISE_CONFIGS[noise_name]
        T1 = float(config["T1"])
        T2 = float(config["T2"])
        print(f"\nTraining PPO for noise={noise_name} (T1={T1}, T2={T2})")

        ppo_seed_values: List[float] = []
        ppo_seed_list: List[int] = []
        curves_dir = output_dirs["curves"]
        checkpoints_dir = output_dirs["checkpoints"]

        for seed in seeds:
            set_seed(seed)
            envs = [TrackingEnv(LindbladBatteryEnv(T1=T1, T2=T2)) for _ in range(4)]
            agent = PPOAgent(state_dim=5, action_dim=2, max_action=0.25, n_envs=4, device="cpu")
            episode_ergotropies: List[float] = []
            episode_rewards: List[float] = []
            convergence_episode: Optional[int] = None

            try:
                for episode in range(args.episodes):
                    for env in envs:
                        env.last_ergotropy = None
                    agent.set_entropy_coeff(episode)
                    trajectories = collect_trajectories(agent, envs, n_steps=100)
                    loss_info = agent.update(trajectories)

                    ergotropy_values = [
                        env.last_ergotropy
                        if env.last_ergotropy is not None
                        else env.get_ergotropy()
                        for env in envs
                    ]
                    mean_ergotropy = float(np.mean(ergotropy_values))
                    mean_reward = float(
                        np.mean([float(np.sum(traj["rewards"])) for traj in trajectories])
                    )
                    episode_ergotropies.append(mean_ergotropy)
                    episode_rewards.append(mean_reward)

                    if convergence_episode is None and len(episode_ergotropies) >= 10:
                        recent = float(np.mean(episode_ergotropies[-5:]))
                        previous = float(np.mean(episode_ergotropies[-10:-5]))
                        if abs(recent - previous) < 0.01:
                            convergence_episode = episode + 1

                    if (episode + 1) % 10 == 0 or episode == 0 or episode + 1 == args.episodes:
                        print(
                            f"[{noise_name}][seed {seed}] "
                            f"Episode {episode + 1}/{args.episodes} "
                            f"erg={mean_ergotropy:.3f} "
                            f"reward={mean_reward:.3f} "
                            f"loss={loss_info['total_loss']:.4f}"
                        )
            except KeyboardInterrupt:
                interrupted = True
                print("\nKeyboard interrupt detected. Saving partial results...")

            if convergence_episode is None:
                convergence_episode = len(episode_ergotropies) if episode_ergotropies else 0

            if episode_ergotropies:
                if not interrupted:
                    final_ergotropy = _evaluate_agent(
                        agent, T1=T1, T2=T2, n_episodes=10, seed=seed + 1000
                    )
                else:
                    final_ergotropy = float(episode_ergotropies[-1])

                curve_payload = {
                    "seed": int(seed),
                    "noise": str(noise_name),
                    "episodes": [float(x) for x in episode_ergotropies],
                    "rewards": [float(x) for x in episode_rewards],
                    "final_ergotropy": float(final_ergotropy),
                    "convergence_episode": int(convergence_episode),
                }
                curve_path = curves_dir / f"seed_{seed}_{noise_name}.json"
                _save_json(curve_path, curve_payload)

                checkpoint_path = checkpoints_dir / f"seed_{seed}_{noise_name}.pt"
                agent.save(str(checkpoint_path))

                ppo_seed_values.append(float(final_ergotropy))
                ppo_seed_list.append(int(seed))

            if interrupted:
                break

        baseline_results: Dict[str, Dict[str, List[float]]] = {
            "BangBang": {"seed_list": [], "seed_values": []},
            "Sinusoidal": {"seed_list": [], "seed_values": []},
            "Random": {"seed_list": [], "seed_values": []},
            "GRAPE": {"seed_list": [], "seed_values": []},
        }
        if not interrupted:
            baseline_results = _run_baselines(T1=T1, T2=T2, seeds=seeds, skip_grape=args.skip_grape)

        summary_payload = {
            "noise_config": {"name": str(noise_name), "T1": T1, "T2": T2},
            "ppo": {
                "seed_list": ppo_seed_list,
                "seed_values": ppo_seed_values,
                "episodes_per_seed": int(args.episodes),
            },
            "baselines": baseline_results,
        }
        summary_path = output_dirs["root"] / f"training_summary_{noise_name}.json"
        _save_json(summary_path, summary_payload)

        summary_rows.append(
            {
                "noise": noise_name,
                "ppo": _mean(ppo_seed_values) if ppo_seed_values else None,
                "BangBang": _mean(baseline_results["BangBang"]["seed_values"])
                if baseline_results["BangBang"]["seed_values"]
                else None,
                "Sinusoidal": _mean(baseline_results["Sinusoidal"]["seed_values"])
                if baseline_results["Sinusoidal"]["seed_values"]
                else None,
                "Random": _mean(baseline_results["Random"]["seed_values"])
                if baseline_results["Random"]["seed_values"]
                else None,
                "GRAPE": _mean(baseline_results["GRAPE"]["seed_values"])
                if baseline_results["GRAPE"]["seed_values"]
                else None,
            }
        )

        if interrupted:
            break

    _print_summary_table(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
