"""
Controlled crossover toy experiment for frozen vs fine-tuned transfer.

Model:
    X ~ N(0, I_2)
    y = u_*^T X + eps, eps ~ N(0, sigma^2)

Frozen encoder:
    z_0 = u(Delta)^T X

Fine-tuned encoder:
    estimate u_* from n labelled samples using sample Cov(X, y)

Population profiled risk:
    F(alpha) = sigma^2 + sin^2(alpha - alpha_star)

Freezing bias:
    B = F(Delta) - F(alpha_star) = sin^2(Delta)

Fine-tuning variance term:
    approximately V / n

Freezing is favoured when:
    B < V / n

Fine-tuning is favoured when:
    B > V / n
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------

SEED = 42
rng = np.random.default_rng(SEED)

OUTDIR = "toy_crossover_outputs"
os.makedirs(OUTDIR, exist_ok=True)

sigma = 1.0
alpha_star = 0.0

# Mismatch of pretrained representation from Bayes-optimal direction
deltas = np.array([0.025, 0.05, 0.10, 0.20, 0.40, 0.80])

# Label budgets
sample_sizes = np.array([10, 25, 50, 100, 250, 500, 1000, 2500])

# Monte Carlo repetitions per setting
reps = 5000


# -----------------------------
# Helper functions
# -----------------------------

def unit_vector(alpha: float) -> np.ndarray:
    """Return u(alpha) = (cos alpha, sin alpha)."""
    return np.array([np.cos(alpha), np.sin(alpha)])


def profiled_population_risk(alpha: float, sigma: float = 1.0,
                             alpha_star: float = 0.0) -> float:
    """
    Population profiled risk for one-dimensional encoder direction alpha.

    For X ~ N(0, I_2), y = u_*^T X + eps, and optimal scalar head,
    F(alpha) = sigma^2 + sin^2(alpha - alpha_star).
    """
    return sigma**2 + np.sin(alpha - alpha_star)**2


def freezing_bias(delta: float) -> float:
    """B = F(delta) - F(alpha_star)."""
    return np.sin(delta - alpha_star)**2


def estimate_finetuned_angle(X: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate the encoder direction from labelled data.

    The population optimal direction is proportional to Cov(X, y).
    The sample estimator uses X^T y / n.
    """
    c = X.T @ y / len(y)
    return np.arctan2(c[1], c[0])


def run_one_setting(delta: float, n: int, reps: int, sigma: float,
                    rng: np.random.Generator) -> dict:
    """
    Run one (delta, n) setting.

    Frozen risk is computed exactly from the population risk.
    Fine-tuned risk is averaged over repeated labelled samples.
    """
    frozen_risk = profiled_population_risk(delta, sigma=sigma,
                                           alpha_star=alpha_star)
    optimal_risk = profiled_population_risk(alpha_star, sigma=sigma,
                                            alpha_star=alpha_star)
    B = frozen_risk - optimal_risk

    ft_risks = np.empty(reps)
    alpha_hats = np.empty(reps)

    u_star = unit_vector(alpha_star)

    for r in range(reps):
        X = rng.normal(size=(n, 2))
        eps = rng.normal(scale=sigma, size=n)
        y = X @ u_star + eps

        alpha_hat = estimate_finetuned_angle(X, y)
        alpha_hats[r] = alpha_hat

        ft_risks[r] = profiled_population_risk(
            alpha_hat,
            sigma=sigma,
            alpha_star=alpha_star
        )

    ft_risk_mean = ft_risks.mean()
    ft_risk_sd = ft_risks.std(ddof=1)

    gap = ft_risk_mean - frozen_risk
    # Positive gap means fine-tuning has higher risk, so freezing is better.
    # Negative gap means fine-tuning has lower risk, so fine-tuning is better.
    winner = "Frozen" if gap > 0 else "FineTune"

    # Empirical estimate of V from local approximation:
    # F(alpha_hat) - F(alpha_star) approx alpha_hat^2.
    # Since Var(alpha_hat) approx V / n, estimate V as n * Var(alpha_hat).
    V_hat = n * np.var(alpha_hats, ddof=1)

    # Predicted leading-order gap:
    # E[F(alpha_hat) - F(delta)] approx V/n - B.
    predicted_gap = V_hat / n - B

    return {
        "delta": delta,
        "n": n,
        "sigma": sigma,
        "B": B,
        "V_hat": V_hat,
        "V_over_n": V_hat / n,
        "predicted_gap": predicted_gap,
        "frozen_risk": frozen_risk,
        "ft_risk_mean": ft_risk_mean,
        "ft_risk_sd": ft_risk_sd,
        "observed_gap_ft_minus_frozen": gap,
        "winner": winner,
    }


# -----------------------------
# Run experiment
# -----------------------------

records = []

for delta in deltas:
    for n in sample_sizes:
        result = run_one_setting(
            delta=delta,
            n=n,
            reps=reps,
            sigma=sigma,
            rng=rng
        )
        records.append(result)

df = pd.DataFrame(records)

csv_path = os.path.join(OUTDIR, "toy_crossover_results.csv")
df.to_csv(csv_path, index=False)

print(df)
print(f"\nSaved results to: {csv_path}")


# -----------------------------
# Plot 1: observed fine-tuning minus frozen risk
# -----------------------------

plt.figure(figsize=(8, 5))

for delta in deltas:
    sub = df[df["delta"] == delta].sort_values("n")
    plt.plot(
        sub["n"],
        sub["observed_gap_ft_minus_frozen"],
        marker="o",
        label=rf"$\Delta={delta}$"
    )

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xscale("log")
plt.xlabel("Labelled sample size n")
plt.ylabel("Mean risk gap: FineTune - Frozen")
plt.title("Observed crossover in the controlled toy experiment")
plt.legend()
plt.tight_layout()

fig1_path = os.path.join(OUTDIR, "toy_crossover_observed_gap.png")
plt.savefig(fig1_path, dpi=300)
plt.close()

print(f"Saved figure to: {fig1_path}")


# -----------------------------
# Plot 2: theoretical comparison V/n versus B
# -----------------------------

plt.figure(figsize=(8, 5))

for delta in deltas:
    sub = df[df["delta"] == delta].sort_values("n")
    plt.plot(
        sub["n"],
        sub["V_over_n"] - sub["B"],
        marker="o",
        label=rf"$\Delta={delta}$"
    )

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xscale("log")
plt.xlabel("Labelled sample size n")
plt.ylabel(r"Estimated leading term $\widehat{V}/n - B$")
plt.title("Estimated bias--variance crossover term")
plt.legend()
plt.tight_layout()

fig2_path = os.path.join(OUTDIR, "toy_crossover_theory_term.png")
plt.savefig(fig2_path, dpi=300)
plt.close()

print(f"Saved figure to: {fig2_path}")


# -----------------------------
# Plot 3: heatmap-style winner table
# -----------------------------

winner_matrix = df.pivot(index="delta", columns="n", values="winner")
gap_matrix = df.pivot(index="delta", columns="n",
                      values="observed_gap_ft_minus_frozen")

# Encode winner only for visualization:
# Frozen = 1, FineTune = -1
encoded = winner_matrix.replace({"Frozen": 1, "FineTune": -1}).astype(float)

plt.figure(figsize=(9, 4.8))
plt.imshow(encoded.values, aspect="auto")

plt.xticks(np.arange(len(sample_sizes)), sample_sizes)
plt.yticks(np.arange(len(deltas)), deltas)

plt.xlabel("Labelled sample size n")
plt.ylabel(r"Pretrained mismatch $\Delta$")
plt.title("Winner map: Frozen versus FineTune")

for i, delta in enumerate(deltas):
    for j, n in enumerate(sample_sizes):
        gap_value = gap_matrix.loc[delta, n]
        label = "Froz" if gap_value > 0 else "FT"
        plt.text(j, i, label, ha="center", va="center")

plt.tight_layout()

fig3_path = os.path.join(OUTDIR, "toy_crossover_winner_map.png")
plt.savefig(fig3_path, dpi=300)
plt.close()

print(f"Saved figure to: {fig3_path}")


# -----------------------------
# Optional LaTeX table
# -----------------------------

latex_df = df.copy()
latex_df["delta"] = latex_df["delta"].map(lambda x: f"{x:.3f}")
latex_df["B"] = latex_df["B"].map(lambda x: f"{x:.4f}")
latex_df["V_over_n"] = latex_df["V_over_n"].map(lambda x: f"{x:.4f}")
latex_df["observed_gap_ft_minus_frozen"] = latex_df[
    "observed_gap_ft_minus_frozen"
].map(lambda x: f"{x:.4f}")

latex_table = latex_df[
    ["delta", "n", "B", "V_over_n", "observed_gap_ft_minus_frozen", "winner"]
].to_latex(index=False, escape=False)

tex_path = os.path.join(OUTDIR, "toy_crossover_table.tex")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(f"Saved LaTeX table to: {tex_path}")