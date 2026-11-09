"""Plotting helpers for the knapsack notebooks (01, 02).

Kept out of the notebooks so the notebooks show the model, not the plumbing.
2026 Quantum Accelerator, Day 1.
"""
import numpy as np
import matplotlib.pyplot as plt


def show_qubo_matrix(Q, n_items, title="The QUBO matrix"):
    """Heatmap of the Q matrix, with the item block separated from the slack block."""
    from matplotlib.colors import SymLogNorm
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    lim = float(np.abs(Q).max())
    # penalty terms are orders of magnitude larger than the item values, so a
    # linear colour scale shows one block of colour and nothing else
    im = ax.imshow(Q, cmap="RdBu_r",
                   norm=SymLogNorm(linthresh=max(1.0, lim / 1e4), vmin=-lim, vmax=lim))
    ax.axhline(n_items - 0.5, color="black", lw=1.2)
    ax.axvline(n_items - 0.5, color="black", lw=1.2)
    ax.set_title(f"{title}\nleft/top block = items, right/bottom block = slack bits",
                 fontsize=10)
    ax.set_xlabel("variable")
    ax.set_ylabel("variable")
    fig.colorbar(im, ax=ax, shrink=0.8, label="coefficient")
    plt.tight_layout()
    plt.show()


def plot_brute_force(weights, values, capacity, subsets):
    """Every subset as a point: weight against value, feasible ones highlighted."""
    w = np.array([sum(weights[i] for i in s) for s in subsets])
    v = np.array([sum(values[i] for i in s) for s in subsets])
    ok = w <= capacity
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(w[~ok], v[~ok], s=28, c="#cccccc", label="too heavy")
    ax.scatter(w[ok], v[ok], s=34, c="#1f77b4", label="fits")
    best = int(np.argmax(np.where(ok, v, -1)))
    ax.scatter([w[best]], [v[best]], s=180, facecolors="none",
               edgecolors="crimson", lw=2.2, label="best that fits")
    ax.axvline(capacity, ls="--", color="crimson", lw=1.4)
    ax.text(capacity, ax.get_ylim()[0], " capacity", color="crimson", fontsize=9,
            va="bottom")
    ax.set_xlabel("total weight")
    ax.set_ylabel("total value")
    ax.set_title(f"All {len(subsets)} possible choices")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_lambda_sweep(lambdas, feasible_pct, gap_pct):
    """The headline plot for M2: feasibility and solution quality against lambda.

    Left arm: lambda too small, the capacity rule gets broken.
    Right arm: lambda too large, the objective is swamped and quality decays.
    """
    lams = np.asarray(lambdas, dtype=float)
    fig, ax1 = plt.subplots(figsize=(8.4, 4.4))
    ax1.semilogx(lams, feasible_pct, "o-", lw=2.5, ms=8, color="#1f77b4",
                 label="runs that respect the capacity (%)")
    ax1.set_ylabel("feasible runs (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(-5, 105)
    ax1.set_xlabel("penalty weight  lambda  (log scale)")
    ax1.grid(alpha=0.3, which="both")

    gaps = np.array([np.nan if g is None else g for g in gap_pct], dtype=float)
    ax2 = ax1.twinx()
    ax2.semilogx(lams, gaps, "s--", lw=2, ms=7, color="#d62728",
                 label="gap to the exact optimum (%)")
    ax2.set_ylabel("gap to exact optimum (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    if np.isfinite(gaps).any():
        best = int(np.nanargmin(gaps))
        ax2.scatter([lams[best]], [gaps[best]], s=200, facecolors="none",
                    edgecolors="black", lw=2)
        ax2.annotate(f"best answer found here\n(lambda={lams[best]:g}, but only "
                     f"{feasible_pct[best]:.0f}% of runs were legal)",
                     xy=(lams[best], gaps[best]), xytext=(14, 26),
                     textcoords="offset points", fontsize=8.5,
                     arrowprops=dict(arrowstyle="->", color="black"))
        first = next((i for i, f in enumerate(feasible_pct) if f > 0), None)
        if first is not None:
            ax1.axvspan(lams[0], lams[first], color="crimson", alpha=0.07)
            ax1.text(lams[0], 50, " nothing legal\n at all", color="crimson",
                     fontsize=8.5, va="center")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="center left")
    ax1.set_title("Too small and nothing is legal. Too large and nothing is good.")
    plt.tight_layout()
    plt.show()
