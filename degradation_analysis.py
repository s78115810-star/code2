"""Pure Python degradation model demonstration with EKF and RUL estimation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


# --------------------------------------------------------------------------------------
# Data containers


@dataclass
class ModelParameters:
    lambda_param: float
    sigma_B: float
    sigma_e: float
    C: float
    m: float = 1.2


@dataclass
class Trajectory:
    times: List[float]
    states: List[float]
    observations: List[float]


# --------------------------------------------------------------------------------------
# Utility functions


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def simulate_trajectory(
    params: ModelParameters,
    rng: random.Random,
    dt: float = 0.1,
    max_time: float = 20.0,
    a0: float = 1.0e-3,
) -> Trajectory:
    n_steps = int(math.ceil(max_time / dt))
    times = [0.0]
    states = [a0]
    observations: List[float] = []

    a_k = a0
    for step in range(1, n_steps + 1):
        if a_k >= 1.0:
            break

        drift = params.C * (max(a_k, 1.0e-9) ** params.m) * dt
        diffusion = params.sigma_B * math.sqrt(dt) * rng.gauss(0.0, 1.0)
        a_k1 = _clip(a_k + drift + diffusion, 1.0e-6, 1.0)

        sqrt_term = math.sqrt(max(1.0 - a_k1, 1.0e-12))
        obs_mean = params.lambda_param * (sqrt_term - 1.0)
        y_k1 = obs_mean + params.sigma_e * rng.gauss(0.0, 1.0)

        times.append(step * dt)
        states.append(a_k1)
        observations.append(y_k1)
        a_k = a_k1

    return Trajectory(times=times, states=states, observations=observations)


def generate_dataset(
    params: ModelParameters,
    n_paths: int = 100,
    dt: float = 0.1,
    max_time: float = 20.0,
    seed: int = 2024,
) -> List[Trajectory]:
    rng = random.Random(seed)
    return [simulate_trajectory(params, rng, dt=dt, max_time=max_time) for _ in range(n_paths)]


# --------------------------------------------------------------------------------------
# Extended Kalman filter helpers (scalar case)


def _predict_state(mean: float, cov: float, params: ModelParameters, dt: float) -> Tuple[float, float]:
    a_safe = max(mean, 1.0e-9)
    drift = params.C * (a_safe**params.m) * dt
    mean_pred = mean + drift
    jacobian = 1.0 + params.C * params.m * (a_safe ** (params.m - 1.0)) * dt
    process_var = params.sigma_B**2 * dt
    cov_pred = jacobian * cov * jacobian + process_var
    return mean_pred, max(cov_pred, 1.0e-10)


def _measurement_function(state: float, params: ModelParameters) -> Tuple[float, float]:
    residual = max(1.0 - state, 1.0e-12)
    sqrt_term = math.sqrt(residual)
    mean = params.lambda_param * (sqrt_term - 1.0)
    jac = -0.5 * params.lambda_param / sqrt_term
    return mean, jac


def ekf_loglikelihood(
    observations: Sequence[float],
    params: ModelParameters,
    dt: float,
    initial_mean: float,
    initial_var: float,
) -> float:
    mean = initial_mean
    cov = initial_var
    loglik = 0.0
    for obs in observations:
        mean_pred, cov_pred = _predict_state(mean, cov, params, dt)
        meas_mean, meas_jac = _measurement_function(mean_pred, params)

        innov = obs - meas_mean
        S = meas_jac * cov_pred * meas_jac + params.sigma_e**2
        S = max(S, 1.0e-9)
        loglik -= 0.5 * (math.log(2.0 * math.pi * S) + (innov**2) / S)

        K = cov_pred * meas_jac / S
        mean = mean_pred + K * innov
        cov = (1.0 - K * meas_jac) * cov_pred

        mean = _clip(mean, 1.0e-6, 0.999999)
        cov = max(cov, 1.0e-9)

    return loglik


def run_ekf(
    observations: Sequence[float],
    params: ModelParameters,
    dt: float,
    initial_mean: float,
    initial_var: float,
) -> Tuple[List[float], List[float]]:
    means: List[float] = []
    variances: List[float] = []
    mean = initial_mean
    cov = initial_var

    for obs in observations:
        mean_pred, cov_pred = _predict_state(mean, cov, params, dt)
        meas_mean, meas_jac = _measurement_function(mean_pred, params)
        innov = obs - meas_mean
        S = meas_jac * cov_pred * meas_jac + params.sigma_e**2
        S = max(S, 1.0e-9)
        K = cov_pred * meas_jac / S
        mean = mean_pred + K * innov
        cov = (1.0 - K * meas_jac) * cov_pred
        mean = _clip(mean, 1.0e-6, 0.999999)
        cov = max(cov, 1.0e-9)
        means.append(mean)
        variances.append(cov)

    return means, variances


# --------------------------------------------------------------------------------------
# Parameter estimation via randomised search


def estimate_parameters(
    dataset: Iterable[Trajectory],
    dt: float,
    initial_mean: float,
    initial_var: float,
    m: float,
    initial_guess: ModelParameters,
    seed: int = 77,
    iterations: int = 800,
) -> ModelParameters:
    rng = random.Random(seed)
    best_params = initial_guess
    best_score = -math.inf

    def score(params: ModelParameters) -> float:
        total = 0.0
        for traj in dataset:
            total += ekf_loglikelihood(traj.observations, params, dt, initial_mean, initial_var)
        return total

    best_score = score(best_params)

    # Random perturbation in log-space to maintain positivity
    for it in range(iterations):
        scale = 0.25 if it < iterations // 2 else 0.1
        proposal = ModelParameters(
            lambda_param=_clip(best_params.lambda_param * math.exp(rng.gauss(0.0, scale)), 0.1, 3.0),
            sigma_B=_clip(best_params.sigma_B * math.exp(rng.gauss(0.0, scale)), 1.0e-4, 1.0),
            sigma_e=_clip(best_params.sigma_e * math.exp(rng.gauss(0.0, scale)), 1.0e-4, 1.0),
            C=_clip(best_params.C * math.exp(rng.gauss(0.0, scale)), 0.01, 2.0),
            m=m,
        )

        proposal_score = score(proposal)
        if proposal_score > best_score:
            best_params = proposal
            best_score = proposal_score

    return best_params


# --------------------------------------------------------------------------------------
# Remaining useful life estimation


def sample_remaining_life(
    current_mean: float,
    current_var: float,
    params: ModelParameters,
    dt: float,
    n_samples: int = 5000,
    max_time: float = 20.0,
    seed: int = 2025,
) -> List[float]:
    rng = random.Random(seed)
    max_steps = int(math.ceil(max_time / dt))
    samples: List[float] = []

    std = math.sqrt(max(current_var, 1.0e-12))
    for _ in range(n_samples):
        a = _clip(rng.gauss(current_mean, std), 1.0e-6, 0.999)
        t = 0.0
        for _ in range(max_steps):
            if a >= 1.0:
                break
            drift = params.C * (max(a, 1.0e-9) ** params.m) * dt
            diffusion = params.sigma_B * math.sqrt(dt) * rng.gauss(0.0, 1.0)
            a = _clip(a + drift + diffusion, 1.0e-6, 1.0)
            t += dt
        samples.append(t)

    return samples


def save_histogram(rul_samples: Sequence[float], output_path: Path, bins: int = 30) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rul_samples:
        raise ValueError("No RUL samples provided")

    min_val = min(rul_samples)
    max_val = max(rul_samples)
    if math.isclose(min_val, max_val):
        max_val = min_val + 1.0

    bin_width = (max_val - min_val) / bins
    counts = [0 for _ in range(bins)]

    for sample in rul_samples:
        idx = int((sample - min_val) / bin_width)
        if idx == bins:
            idx -= 1
        counts[idx] += 1

    density = [c / (len(rul_samples) * bin_width) for c in counts]
    centers = [min_val + (i + 0.5) * bin_width for i in range(bins)]

    text_path = output_path.with_suffix(".txt")
    with text_path.open("w", encoding="utf-8") as f:
        f.write("RUL histogram (density)\n")
        f.write("bin_center,density\n")
        for center, dens in zip(centers, density):
            f.write(f"{center:.6f},{dens:.6f}\n")

    return text_path


# --------------------------------------------------------------------------------------
# Main routine tying everything together


def main() -> None:
    true_params = ModelParameters(lambda_param=0.9, sigma_B=0.04, sigma_e=0.02, C=0.35, m=1.2)
    dt = 0.1
    initial_mean = 1.0e-3
    initial_var = 1.0e-4

    dataset = generate_dataset(true_params, n_paths=100, dt=dt, max_time=20.0, seed=2024)
    print("Generated dataset with 100 trajectories")

    initial_guess = ModelParameters(lambda_param=0.8, sigma_B=0.05, sigma_e=0.03, C=0.3, m=1.2)
    estimated = estimate_parameters(
        dataset,
        dt=dt,
        initial_mean=initial_mean,
        initial_var=initial_var,
        m=true_params.m,
        initial_guess=initial_guess,
    )

    print("Estimated parameters:")
    print(estimated)

    reference_traj = dataset[0]
    means, variances = run_ekf(
        reference_traj.observations,
        estimated,
        dt=dt,
        initial_mean=initial_mean,
        initial_var=initial_var,
    )

    print(f"EKF final state mean: {means[-1]:.4f}")

    rul_samples = sample_remaining_life(
        current_mean=means[-1],
        current_var=variances[-1],
        params=estimated,
        dt=dt,
        n_samples=5000,
        max_time=20.0,
    )

    output_dir = Path("outputs")
    saved_path = save_histogram(rul_samples, output_dir / "rul_histogram.png")
    print(f"Saved histogram data to {saved_path.resolve()}")


if __name__ == "__main__":
    main()

