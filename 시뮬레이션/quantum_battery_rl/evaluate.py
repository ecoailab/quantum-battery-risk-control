import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kruskal, norm, shapiro


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _looks_like_results(values: List[float]) -> bool:
    if not values:
        return False
    try:
        return all(0.0 <= float(v) <= 1.0 for v in values)
    except (TypeError, ValueError):
        return False


def _bca_interval(
    values: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[float], Optional[float]]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return None, None
    if rng is None:
        rng = np.random.default_rng()

    boot_samples = rng.choice(data, size=(n_bootstrap, data.size), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    theta_hat = float(np.mean(data))

    prop = float(np.mean(boot_means < theta_hat))
    prop = float(np.clip(prop, 1.0 / (2.0 * n_bootstrap), 1.0 - 1.0 / (2.0 * n_bootstrap)))
    z0 = float(norm.ppf(prop))

    if data.size <= 1:
        acceleration = 0.0
    else:
        jack_means = np.array([
            np.mean(np.delete(data, i)) for i in range(data.size)
        ])
        jack_mean = float(np.mean(jack_means))
        diffs = jack_mean - jack_means
        denom = 6.0 * (np.sum(diffs ** 2) ** 1.5)
        acceleration = float(np.sum(diffs ** 3) / denom) if denom != 0.0 else 0.0

    z_low = float(norm.ppf(alpha / 2.0))
    z_high = float(norm.ppf(1.0 - alpha / 2.0))

    alpha1 = norm.cdf(
        z0 + (z0 + z_low) / (1.0 - acceleration * (z0 + z_low))
    )
    alpha2 = norm.cdf(
        z0 + (z0 + z_high) / (1.0 - acceleration * (z0 + z_high))
    )

    alpha1 = float(np.clip(alpha1, 0.0, 1.0))
    alpha2 = float(np.clip(alpha2, 0.0, 1.0))

    ci_low = float(np.quantile(boot_means, alpha1))
    ci_high = float(np.quantile(boot_means, alpha2))
    return ci_low, ci_high


def _compute_stats(values: List[float], n_bootstrap: int, rng: np.random.Generator) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None}
    data = np.asarray(values, dtype=float)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    ci_low, ci_high = _bca_interval(values, n_bootstrap=n_bootstrap, rng=rng)
    return {"mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high}


def _cohens_d(values_a: List[float], values_b: List[float]) -> Optional[float]:
    if not values_a or not values_b:
        return None
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n1, n2 = a.size, b.size
    if n1 < 2 and n2 < 2:
        return None
    var1 = float(np.var(a, ddof=1)) if n1 > 1 else 0.0
    var2 = float(np.var(b, ddof=1)) if n2 > 1 else 0.0
    denom = n1 + n2 - 2
    if denom <= 0:
        return None
    pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / denom
    if pooled <= 0.0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def _shapiro_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if len(values) < 3:
        return {"W": None, "p": None}
    stat, p_value = shapiro(values)
    return {"W": float(stat), "p": float(p_value)}


def _load_curve_final(results_dir: Path, seed: int, noise_name: str) -> Optional[float]:
    curve_path = results_dir / "curves" / f"seed_{seed}_{noise_name}.json"
    if not curve_path.exists():
        return None
    payload = _load_json(curve_path)
    if "final_ergotropy" in payload:
        return float(payload["final_ergotropy"])
    return None


def _seed_list_from_summary(summary: dict, results_dir: Path, noise_name: str) -> List[int]:
    ppo_node = summary.get("ppo", {})
    seed_list = ppo_node.get("seed_list")
    if seed_list:
        return [int(s) for s in seed_list]
    seed_values = ppo_node.get("seed_values", [])
    if seed_values and not _looks_like_results(seed_values):
        return [int(s) for s in seed_values]
    curves_dir = results_dir / "curves"
    if curves_dir.exists():
        seeds = []
        for path in curves_dir.glob(f"seed_*_{noise_name}.json"):
            parts = path.stem.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                seeds.append(int(parts[1]))
        if seeds:
            return sorted(set(seeds))
    return []


def _extract_method_values(
    summary: dict,
    method_name: str,
    results_dir: Path,
    noise_name: str,
) -> List[float]:
    if method_name == "PPO":
        node = summary.get("ppo", {})
    else:
        node = summary.get("baselines", {}).get(method_name, {})

    seed_values = node.get("seed_values", [])
    if seed_values and _looks_like_results(seed_values):
        return [float(v) for v in seed_values]

    if method_name == "PPO":
        seeds = node.get("seed_list") or seed_values
        results: List[float] = []
        for seed in seeds:
            value = _load_curve_final(results_dir, int(seed), noise_name)
            if value is not None:
                results.append(float(value))
        return results

    baseline_results_path = results_dir / f"baseline_results_{noise_name}.json"
    if baseline_results_path.exists():
        payload = _load_json(baseline_results_path)
        values = payload.get(method_name, {}).get("seed_values", [])
        if values and _looks_like_results(values):
            return [float(v) for v in values]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate training summaries.")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output", default="paper_results.json", help="Output JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_paths = sorted(results_dir.glob("training_summary_*.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No training_summary_*.json in {results_dir}")

    summaries = [_load_json(path) for path in summary_paths]
    rng = np.random.default_rng(123)
    bootstrap_samples = 1000

    noise_robustness = []
    for summary in summaries:
        noise_config = summary.get("noise_config", {})
        noise_name = str(noise_config.get("name", ""))
        T1 = float(noise_config.get("T1", 0.0))
        T2 = float(noise_config.get("T2", 0.0))

        ppo_values = _extract_method_values(summary, "PPO", results_dir, noise_name)
        bangbang_values = _extract_method_values(summary, "BangBang", results_dir, noise_name)
        sinusoidal_values = _extract_method_values(summary, "Sinusoidal", results_dir, noise_name)
        random_values = _extract_method_values(summary, "Random", results_dir, noise_name)
        grape_values = _extract_method_values(summary, "GRAPE", results_dir, noise_name)

        noise_robustness.append(
            {
                "name": noise_name,
                "T1": T1,
                "T2": T2,
                "ppo": {
                    "seed_values": ppo_values,
                    "stats": _compute_stats(ppo_values, bootstrap_samples, rng),
                },
                "baselines": {
                    "BangBang": {
                        "seed_values": bangbang_values,
                        "stats": _compute_stats(bangbang_values, bootstrap_samples, rng),
                    },
                    "Sinusoidal": {
                        "seed_values": sinusoidal_values,
                        "stats": _compute_stats(sinusoidal_values, bootstrap_samples, rng),
                    },
                    "Random": {
                        "seed_values": random_values,
                        "stats": _compute_stats(random_values, bootstrap_samples, rng),
                    },
                    "GRAPE": {
                        "seed_values": grape_values,
                        "stats": _compute_stats(grape_values, bootstrap_samples, rng),
                    },
                },
            }
        )

    ibm_entry = next((entry for entry in noise_robustness if entry["name"] == "IBM-like"), None)
    if ibm_entry:
        ppo_values = ibm_entry["ppo"]["seed_values"]
        bangbang_values = ibm_entry["baselines"]["BangBang"]["seed_values"]
        sinusoidal_values = ibm_entry["baselines"]["Sinusoidal"]["seed_values"]
        random_values = ibm_entry["baselines"]["Random"]["seed_values"]
        grape_values = ibm_entry["baselines"]["GRAPE"]["seed_values"]

        groups = [values for values in [ppo_values, bangbang_values, sinusoidal_values, random_values, grape_values] if values]
        if len(groups) >= 2:
            H, p_value = kruskal(*groups)
            kruskal_result = {"H": float(H), "p": float(p_value)}
        else:
            kruskal_result = {"H": None, "p": None}

        baseline_comparison = {
            "noise": "IBM-like",
            "PPO": {
                "seed_values": ppo_values,
                "stats": _compute_stats(ppo_values, bootstrap_samples, rng),
            },
            "BangBang": {
                "seed_values": bangbang_values,
                "stats": _compute_stats(bangbang_values, bootstrap_samples, rng),
            },
            "Sinusoidal": {
                "seed_values": sinusoidal_values,
                "stats": _compute_stats(sinusoidal_values, bootstrap_samples, rng),
            },
            "Random": {
                "seed_values": random_values,
                "stats": _compute_stats(random_values, bootstrap_samples, rng),
            },
            "GRAPE": {
                "seed_values": grape_values,
                "stats": _compute_stats(grape_values, bootstrap_samples, rng),
            },
            "kruskal_wallis": kruskal_result,
            "cohens_d": {
                "PPO_vs_BangBang": _cohens_d(ppo_values, bangbang_values),
                "PPO_vs_Sinusoidal": _cohens_d(ppo_values, sinusoidal_values),
                "PPO_vs_Random": _cohens_d(ppo_values, random_values),
                "PPO_vs_GRAPE": _cohens_d(ppo_values, grape_values),
            },
            "shapiro_wilk": {
                "PPO": _shapiro_stats(ppo_values),
                "BangBang": _shapiro_stats(bangbang_values),
                "Sinusoidal": _shapiro_stats(sinusoidal_values),
                "Random": _shapiro_stats(random_values),
                "GRAPE": _shapiro_stats(grape_values),
            },
        }
    else:
        baseline_comparison = {
            "noise": "IBM-like",
            "PPO": {"seed_values": [], "stats": {"mean": None, "std": None, "ci_low": None, "ci_high": None}},
            "BangBang": {"seed_values": [], "stats": {"mean": None, "std": None, "ci_low": None, "ci_high": None}},
            "Sinusoidal": {"seed_values": [], "stats": {"mean": None, "std": None, "ci_low": None, "ci_high": None}},
            "Random": {"seed_values": [], "stats": {"mean": None, "std": None, "ci_low": None, "ci_high": None}},
            "GRAPE": {"seed_values": [], "stats": {"mean": None, "std": None, "ci_low": None, "ci_high": None}},
            "kruskal_wallis": {"H": None, "p": None},
            "cohens_d": {
                "PPO_vs_BangBang": None,
                "PPO_vs_Sinusoidal": None,
                "PPO_vs_Random": None,
                "PPO_vs_GRAPE": None,
            },
            "shapiro_wilk": {
                "PPO": {"W": None, "p": None},
                "BangBang": {"W": None, "p": None},
                "Sinusoidal": {"W": None, "p": None},
                "Random": {"W": None, "p": None},
                "GRAPE": {"W": None, "p": None},
            },
        }

    seed_list = []
    episodes_per_seed = None
    for summary in summaries:
        noise_name = str(summary.get("noise_config", {}).get("name", ""))
        seeds = _seed_list_from_summary(summary, results_dir, noise_name)
        if seeds:
            seed_list = seeds
            episodes_per_seed = summary.get("ppo", {}).get("episodes_per_seed")
            break

    output_payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds": seed_list,
            "episodes": episodes_per_seed,
            "bootstrap_samples": bootstrap_samples,
            "synthetic": False,
        },
        "noise_robustness": noise_robustness,
        "baseline_comparison": baseline_comparison,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
