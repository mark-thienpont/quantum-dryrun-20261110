"""Helpers for the QUBO ladder notebooks L1-L5.

Kept out of the notebooks so they show the model, not the plumbing.
Convention: Q is a dense symmetric numpy array and the objective is x^T Q x, exactly as on
the slides. dimod.BinaryQuadraticModel(Q, "BINARY") reproduces that energy (it adds Q_ij and
Q_ji). 2026 Quantum Accelerator, Day 1.
"""
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from ortools.sat.python import cp_model

SAMPLER = SimulatedAnnealingSampler()


# ----------------------------------------------------------------------------------------
# 1. Generic: matrices, exact solving, annealing
# ----------------------------------------------------------------------------------------
def bqm_from_symmetric(Q):
    """dimod BQM whose energy is x^T Q x for the symmetric matrix Q."""
    return dimod.BinaryQuadraticModel(np.asarray(Q, dtype=float), "BINARY")


def symmetric_from_qubo_dict(qubo, labels):
    """Turn PyQUBO's upper-triangular to_qubo() dict into the slide's symmetric matrix.
    Off-diagonal coefficients are halved and mirrored; `labels` fixes the row order."""
    idx = {l: k for k, l in enumerate(labels)}
    Q = np.zeros((len(labels), len(labels)))
    for (a, b), v in qubo.items():
        i, j = idx[a], idx[b]
        if i == j:
            Q[i, i] += v
        else:
            Q[i, j] += v / 2
            Q[j, i] += v / 2
    return Q


def print_matrix(Q, labels, title="Q"):
    """Print an integer-valued symmetric matrix with row/column labels (no plot)."""
    Q = np.asarray(Q)
    integral = np.allclose(Q, np.round(Q))
    fmt = (lambda v: str(int(round(v)))) if integral else (lambda v: f"{v:.2f}")
    cells = [[fmt(v) for v in row] for row in Q]
    width = max(max(len(c) for row in cells for c in row), max(len(str(l)) for l in labels)) + 2
    print(title)
    print(" " * width + "".join(f"{str(l):>{width}}" for l in labels))
    for l, row in zip(labels, cells):
        print(f"{str(l):>{width}}" + "".join(f"{c:>{width}}" for c in row))


def exact_table(Q, labels, const=0.0, top=None, ties_asc=True):
    """Every assignment with dimod.ExactSolver, sorted by energy. Returns a pandas
    DataFrame with one column per label, then 'energy' and 'objective' (= energy + const).
    `top=k` keeps the k best rows; equal energies are listed in binary order (reverse if
    ties_asc=False, which puts the slide's (1,0,0) check line first in L1)."""
    import pandas as pd
    ss = dimod.ExactSolver().sample(bqm_from_symmetric(Q))
    rows = []
    for sample, energy in ss.data(["sample", "energy"]):
        row = {l: int(sample[k]) for k, l in enumerate(labels)}
        row["energy"] = float(energy)
        row["objective"] = float(energy) + const
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(
        by=["energy"] + list(labels), ascending=[True] + [ties_asc] * len(labels), kind="stable")
    df = df.reset_index(drop=True)
    for c in ("energy", "objective"):
        if np.allclose(df[c], np.round(df[c])):
            df[c] = df[c].round().astype(int)
    return df.head(top) if top else df


def anneal(Q, num_reads=100, num_sweeps=2000, seed=1):
    """Simulated annealing on x^T Q x. Returns (sampleset, seconds)."""
    bqm = bqm_from_symmetric(Q)
    t = time.time()
    ss = SAMPLER.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    return ss, time.time() - t


def read_matrix(sampleset, n):
    """All reads as a (num_reads x n) int array in variable order 0..n-1."""
    return np.array([[s[i] for i in range(n)] for s in sampleset.samples()], dtype=int)


def best_row(sampleset, n):
    """(x, energy) of the lowest-energy read."""
    first = sampleset.first
    return np.array([first.sample[i] for i in range(n)], dtype=int), float(first.energy)


# ----------------------------------------------------------------------------------------
# 2. L1 - Max-Cut
# ----------------------------------------------------------------------------------------
def maxcut_slide():
    """The 3-node slide instance: labels ['A','B','C'] and weight matrix W (8, -2, -3)."""
    W = np.array([[0, 8, -2], [8, 0, -3], [-2, -3, 0]])
    return ["A", "B", "C"], W


def maxcut_Q(W):
    """Q for min(-cut): Q_ii = -sum_j W_ij, Q_ij = +W_ij (symmetric)."""
    W = np.asarray(W)
    return -np.diag(W.sum(axis=1)) + W


def cut_value(W, x):
    """Total weight of the edges with endpoints on different sides."""
    x = np.asarray(x)
    diff = (x[:, None] != x[None, :])
    return float((np.asarray(W) * diff).sum() / 2)


def maxcut_instance(n=120, p=0.15, w_low=1, w_high=10, seed=1):
    """Random G(n,p) with integer edge weights in [w_low, w_high]. Returns (W, G)."""
    rng = np.random.default_rng(seed)
    G = nx.gnp_random_graph(n, p, seed=seed)
    W = np.zeros((n, n))
    for i, j in G.edges():
        W[i, j] = W[j, i] = rng.integers(w_low, w_high + 1)
    return W, G


def cpsat_maxcut(W, time_limit=30.0, workers=2):
    """CP-SAT on the standard linearisation (y_ij <= x_i + x_j, y_ij <= 2 - x_i - x_j).
    Returns dict(value, bound, status, seconds)."""
    n = len(W)
    m = cp_model.CpModel()
    xs = [m.NewBoolVar(f"x{i}") for i in range(n)]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] != 0:
                y = m.NewBoolVar(f"y{i}_{j}")
                m.Add(y <= xs[i] + xs[j])
                m.Add(y <= 2 - xs[i] - xs[j])
                terms.append(int(W[i, j]) * y)
    m.Maximize(sum(terms))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    t = time.time()
    st = s.Solve(m)
    return dict(value=s.ObjectiveValue(), bound=s.BestObjectiveBound(),
                status=s.StatusName(st), seconds=time.time() - t)


def draw_cut(G, W, x, title=""):
    """Spring layout, nodes coloured by side, cut edges drawn in orange, others grey."""
    pos = nx.spring_layout(G, seed=1)
    cut = [(i, j) for i, j in G.edges() if x[i] != x[j]]
    rest = [(i, j) for i, j in G.edges() if x[i] == x[j]]
    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(G, pos, edgelist=rest, edge_color="#cccccc", width=0.4, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut, edge_color="#ff7f0e", width=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=40, ax=ax,
                           node_color=["#1f77b4" if v else "#2ca02c" for v in x])
    ax.set_title(title or f"Cut weight {cut_value(W, x):.0f}  (orange = edges between the groups)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------
# 3. L2 - Number partitioning
# ----------------------------------------------------------------------------------------
def partition_slide():
    """S = {3, 5, 8} as a numpy array and labels ['x1','x2','x3']."""
    return np.array([3, 5, 8]), ["x1", "x2", "x3"]


def partition_Q(a):
    """Q = a a^T - diag(a * T), T = sum(a).  x^T Q x = (sum a_i x_i)^2 - T sum a_i x_i;
    minimum -T^2/4 at a perfect split."""
    a = np.asarray(a)
    return np.outer(a, a) - np.diag(a * a.sum())


def partition_diff(a, x):
    """|sum(A) - sum(B)| for the split encoded by x."""
    a = np.asarray(a)
    return int(abs(2 * int(a @ np.asarray(x)) - int(a.sum())))


def partition_dp(a):
    """Exact minimum difference by reachable-subset-sum DP (numpy bool array of size T//2+1).
    O(n * T) time, O(T) memory: fine up to T ~ 5e7."""
    a = [int(v) for v in a]
    T = sum(a)
    half = T // 2
    reach = np.zeros(half + 1, dtype=bool)
    reach[0] = True
    for v in a:
        if v <= half:
            reach[v:] |= reach[:-v].copy()
    return int(T - 2 * np.max(np.nonzero(reach)[0]))


def partition_instance(n=40, max_value=100, seed=1):
    """n integers uniform in [1, max_value]."""
    rng = np.random.default_rng(seed)
    return rng.integers(1, max_value + 1, n)


def partition_sweep(n=40, max_values=(100, 10_000, 1_000_000), seed=1, **sa):
    """For each max_value: build a, anneal, return rows of
    dict(max_value, T, dp_diff, sa_best_diff, hit_pct, seconds, dp_seconds)."""
    out = []
    for mv in max_values:
        a = partition_instance(n, mv, seed)
        t = time.time()
        dp = partition_dp(a)
        dp_s = time.time() - t
        ss, dt = anneal(partition_Q(a), **sa)
        R = read_matrix(ss, n)
        diffs = np.array([partition_diff(a, x) for x in R])
        out.append(dict(max_value=mv, T=int(a.sum()), dp_diff=dp, sa_best_diff=int(diffs.min()),
                        hit_pct=float(100 * np.mean(diffs == dp)), seconds=dt, dp_seconds=dp_s))
    return out


def plot_partition_sweep(rows):
    """Bar/line chart: hit % and best difference against bits per number (log x)."""
    bits = [np.log2(r["max_value"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.bar(bits, [r["hit_pct"] for r in rows], width=1.2, color="#1f77b4", alpha=0.8)
    ax1.set_xlabel("bits per number (log2 of the largest number)")
    ax1.set_ylabel("reads that reach the DP optimum (%)", color="#1f77b4")
    ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(bits, [r["sa_best_diff"] for r in rows], "o-", color="crimson")
    ax2.set_ylabel("best difference found by SA", color="crimson")
    ax2.set_ylim(bottom=-5)
    ax1.set_xticks(bits)
    ax1.set_xticklabels([f"{b:.0f}\n(up to {r['max_value']:,})" for b, r in zip(bits, rows)])
    ax1.set_title("Same n, same solver: only the size of the numbers changes")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------
# 4. L3 - Max-2-SAT   clause = (i, s_i, j, s_j, w), s = +1 for x, -1 for NOT x
# ----------------------------------------------------------------------------------------
def max2sat_slide():
    """The slide's 5 clauses on 3 variables."""
    return [(0, 1, 1, 1, 1), (0, 1, 1, -1, 2), (0, -1, 1, -1, 3), (0, -1, 2, -1, 2),
            (1, 1, 2, 1, 2)]


def clause_text(c, labels):
    """'x1 OR NOT x2  (w=2)'."""
    i, si, j, sj, w = c
    lit = lambda k, s: labels[k] if s == 1 else f"NOT {labels[k]}"
    return f"{lit(i, si)} OR {lit(j, sj)}  (w={w})"


def max2sat_Q(clauses, n):
    """Violated-weight objective sum_c w_c (1-l_1)(1-l_2) as (Q, const)."""
    Q = np.zeros((n, n))
    const = 0.0
    for (i, si, j, sj, w) in clauses:
        A = {"c": 1, i: -1} if si == 1 else {i: 1}
        B = {"c": 1, j: -1} if sj == 1 else {j: 1}
        for ka, va in A.items():
            for kb, vb in B.items():
                if ka == "c" and kb == "c":
                    const += w * va * vb
                elif ka == "c":
                    Q[kb, kb] += w * va * vb
                elif kb == "c":
                    Q[ka, ka] += w * va * vb
                else:
                    Q[ka, kb] += w * va * vb / 2
                    Q[kb, ka] += w * va * vb / 2
    return Q, const


def violated_weight(clauses, x):
    """Total weight of clauses with both literals false under x."""
    tot = 0
    for (i, si, j, sj, w) in clauses:
        li = x[i] if si == 1 else 1 - x[i]
        lj = x[j] if sj == 1 else 1 - x[j]
        if li == 0 and lj == 0:
            tot += w
    return tot


def max2sat_instance(n=100, m=400, w_high=5, seed=1):
    """m random clauses on n variables, two distinct variables each, random polarities,
    weights uniform in [1, w_high]."""
    rng = np.random.default_rng(seed)
    clauses = []
    for _ in range(m):
        i, j = rng.choice(n, 2, replace=False)
        si, sj = rng.choice([1, -1], 2)
        w = int(rng.integers(1, w_high + 1))
        clauses.append((int(i), int(si), int(j), int(sj), w))
    return clauses


def cpsat_max2sat(clauses, n, time_limit=30.0, workers=2):
    """CP-SAT: bool c per clause with c <= l_i + l_j, maximise sum w c.
    Returns dict(violated, bound_violated, status, seconds) where violated = W - objective."""
    m = cp_model.CpModel()
    xs = [m.NewBoolVar(f"x{i}") for i in range(n)]
    terms = []
    for (i, si, j, sj, w) in clauses:
        li = xs[i] if si == 1 else xs[i].Not()
        lj = xs[j] if sj == 1 else xs[j].Not()
        c = m.NewBoolVar("")
        m.Add(c <= li + lj)
        terms.append(w * c)
    m.Maximize(sum(terms))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    t = time.time()
    st = s.Solve(m)
    W = sum(c[4] for c in clauses)
    return dict(violated=W - s.ObjectiveValue(), bound_violated=W - s.BestObjectiveBound(),
                status=s.StatusName(st), seconds=time.time() - t)


# ----------------------------------------------------------------------------------------
# 5. L4 - MWIS
# ----------------------------------------------------------------------------------------
def mwis_slide():
    """4-cycle A-B-C-D, weights (4, 6, 5, 2), edges [(0,1),(1,2),(2,3),(3,0)], P = 8.
    Returns (points, w, edges, P); points are square corners for drawing."""
    points = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=float)
    return points, np.array([4, 6, 5, 2]), [(0, 1), (1, 2), (2, 3), (3, 0)], 8


def mwis_Q(w, edges, P):
    """Q = -diag(w) with +P/2 on Q_ij and Q_ji for every edge (so the cross term is P)."""
    Q = -np.diag(np.asarray(w, dtype=float))
    for i, j in edges:
        Q[i, j] += P / 2
        Q[j, i] += P / 2
    return Q


def is_independent(edges, x):
    """True if no edge has both endpoints selected."""
    return all(not (x[i] and x[j]) for i, j in edges)


def mwis_instance(n=150, radius=0.12, w_high=10, seed=1):
    """Unit-disk graph: n random points in the unit square, edge if distance < radius,
    integer weights in [1, w_high]. Returns (points, w, edges)."""
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    w = rng.integers(1, w_high + 1, n)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if np.hypot(*(points[i] - points[j])) < radius]
    return points, w, edges


def cpsat_mwis(w, edges, time_limit=30.0, workers=2):
    """x_i + x_j <= 1 per edge, maximise sum w x. Returns dict(value, status, seconds)."""
    m = cp_model.CpModel()
    xs = [m.NewBoolVar(f"x{i}") for i in range(len(w))]
    for i, j in edges:
        m.Add(xs[i] + xs[j] <= 1)
    m.Maximize(sum(int(w[i]) * xs[i] for i in range(len(w))))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    t = time.time()
    st = s.Solve(m)
    return dict(value=s.ObjectiveValue(), status=s.StatusName(st), seconds=time.time() - t)


def mwis_stats(sampleset, w, edges, n):
    """dict(feasible_pct, best_feasible_value or None, best_x)."""
    R = read_matrix(sampleset, n)
    feas = [x for x in R if is_independent(edges, x)]
    if not feas:
        return dict(feasible_pct=0.0, best_feasible_value=None, best_x=None)
    vals = [int(np.asarray(w) @ x) for x in feas]
    k = int(np.argmax(vals))
    return dict(feasible_pct=float(100 * len(feas) / len(R)), best_feasible_value=vals[k],
                best_x=feas[k])


def penalty_sweep_mwis(w, edges, Ps, **sa):
    """For each P: anneal, return rows dict(P, feasible_pct, best_value)."""
    n = len(w)
    rows = []
    for P in Ps:
        ss, _ = anneal(mwis_Q(w, edges, P), **sa)
        st = mwis_stats(ss, w, edges, n)
        rows.append(dict(P=P, feasible_pct=st["feasible_pct"], best_value=st["best_feasible_value"]))
    return rows


def draw_disk_graph(points, edges, x=None, radius=None, title=""):
    """Scatter of points with edges; selected nodes filled, violated edges in red."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, j in edges:
        bad = x is not None and x[i] and x[j]
        ax.plot(*zip(points[i], points[j]), color="red" if bad else "#bbbbbb",
                lw=1.6 if bad else 0.6, zorder=1)
    sel = np.zeros(len(points), bool) if x is None else np.asarray(x, bool)
    ax.scatter(*points[~sel].T, s=30, facecolors="white", edgecolors="#555555", zorder=2)
    if sel.any():
        ax.scatter(*points[sel].T, s=60, color="#1f77b4", zorder=3, label="selected")
        ax.legend(loc="upper right", fontsize=9)
    if radius:
        ax.set_xlabel(f"edge if distance < {radius}")
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_penalty_sweep(rows, reference=None, xlabel="penalty P"):
    """Twin-axis plot: feasible % (left) and best feasible value (right) against P,
    reference optimum as a dashed line. Reused by L5 with xlabel='lambda'."""
    Ps = [r["P"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax1.plot(Ps, [r["feasible_pct"] for r in rows], "o-", color="#1f77b4")
    ax1.set_xscale("log")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("reads that are feasible (%)", color="#1f77b4")
    ax1.set_ylim(-3, 103)
    ax2 = ax1.twinx()
    pts = [(r["P"], r["best_value"]) for r in rows if r["best_value"] is not None]
    if pts:
        ax2.plot(*zip(*pts), "s-", color="crimson")
    if reference is not None:
        ax2.axhline(reference, ls="--", color="grey")
        ax2.text(Ps[0], reference, " reference optimum", va="bottom", fontsize=9, color="grey")
    ax2.set_ylabel("best feasible value", color="crimson")
    ax1.grid(alpha=0.3)
    ax1.set_title("The penalty dial: legality against quality")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------
# 6. L5 - Knapsack
# ----------------------------------------------------------------------------------------
def knapsack_slide():
    """Items A,B,C: weights (2,3,4), values (6,4,5), capacity 7, P = 7, slack coeffs (1,2,4).
    Returns (weights, values, C, P, coeffs)."""
    return np.array([2, 3, 4]), np.array([6, 4, 5]), 7, 7, np.array([1, 2, 4])


def slack_bits(capacity):
    """Binary-expansion coefficients covering exactly 0..capacity (powers of two, last
    coefficient capacity + 1 - 2^M)."""
    coeffs, rem, pw = [], int(capacity), 1
    while rem > 0:
        k = min(pw, rem)
        coeffs.append(k)
        rem -= k
        pw *= 2
    return np.array(coeffs)


def knapsack_Q(values, weights, capacity, P, coeffs=None):
    """Variables y = (x_1..x_n, s_1..s_M), a = (weights..., coeffs...).
    Q_ii = -v_i + P (a_i^2 - 2 C a_i) (v_i = 0 for slack bits); Q_ij = P a_i a_j.
    Returns (Q, const = P C^2, coeffs)."""
    if coeffs is None:
        coeffs = slack_bits(capacity)
    a = np.concatenate([weights, coeffs]).astype(float)
    v = np.concatenate([values, np.zeros(len(coeffs))]).astype(float)
    Q = P * np.outer(a, a)
    np.fill_diagonal(Q, -v + P * (a * a - 2 * capacity * a))
    return Q, P * capacity * capacity, coeffs


def knapsack_check(values, weights, capacity, x_items):
    """dict(weight, value, feasible)."""
    x = np.asarray(x_items)
    wt, val = int(np.asarray(weights) @ x), int(np.asarray(values) @ x)
    return dict(weight=wt, value=val, feasible=wt <= capacity)


def knapsack_dp(values, weights, capacity):
    """Exact optimum by dynamic programming over the capacity."""
    best = np.zeros(int(capacity) + 1, dtype=int)
    for v, w in zip(values, weights):
        w, v = int(w), int(v)
        if w <= capacity:
            best[w:] = np.maximum(best[w:], best[:-w] + v) if w > 0 else best
    return int(best[-1])


def knapsack_m2_instance():
    """The notebook-02 instance, byte for byte. DP optimum 1789."""
    rng = np.random.default_rng(7)
    values = rng.integers(10, 100, 40)
    weights = rng.integers(5, 50, 40)
    return values, weights, 525


def knapsack_stats(sampleset, values, weights, capacity, n_items):
    """dict(feasible_pct, best_value or None, gap_pct or None, best_x)."""
    R = read_matrix(sampleset, n_items + (sampleset.record.sample.shape[1] - n_items))[:, :n_items]
    feas = [x for x in R if int(np.asarray(weights) @ x) <= capacity]
    if not feas:
        return dict(feasible_pct=0.0, best_value=None, gap_pct=None, best_x=None)
    vals = [int(np.asarray(values) @ x) for x in feas]
    k = int(np.argmax(vals))
    opt = knapsack_dp(values, weights, capacity)
    return dict(feasible_pct=float(100 * len(feas) / len(R)), best_value=vals[k],
                gap_pct=float(100 * (opt - vals[k]) / opt), best_x=feas[k])


def penalty_sweep_knapsack(values, weights, capacity, Ps, **sa):
    """For each P: build Q, anneal, return rows dict(P, feasible_pct, best_value, gap_pct)."""
    rows = []
    n = len(values)
    for P in Ps:
        Q, _, _ = knapsack_Q(values, weights, capacity, P)
        ss, _ = anneal(Q, **sa)
        st = knapsack_stats(ss, values, weights, capacity, n)
        rows.append(dict(P=P, feasible_pct=st["feasible_pct"], best_value=st["best_value"],
                         gap_pct=st["gap_pct"]))
    return rows


# ----------------------------------------------------------------------------------------
# 7. Self-test
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    def ground(Q, const=0):
        t = exact_table(Q, [f"v{k}" for k in range(len(Q))], const)
        e0 = t.energy.iloc[0]
        return e0, t[t.energy == e0].iloc[:, :len(Q)].values.tolist()

    labels, W = maxcut_slide()
    Q = maxcut_Q(W)
    assert (Q == [[-6, 8, -2], [8, -5, -3], [-2, -3, 5]]).all()
    e, am = ground(Q)
    assert e == -6 and sorted(am) == [[0, 1, 1], [1, 0, 0]]

    a, _ = partition_slide()
    Q = partition_Q(a)
    assert (Q == [[-39, 15, 24], [15, -55, 40], [24, 40, -64]]).all()
    e, am = ground(Q)
    assert e == -64 and sorted(am) == [[0, 0, 1], [1, 1, 0]]

    Q, c = max2sat_Q(max2sat_slide(), 3)
    assert (Q == [[-1, 1, 1], [1, -1, 1], [1, 1, -2]]).all() and c == 3
    e, am = ground(Q, c)
    assert e == -2 and am == [[0, 0, 1]]

    _, w, edges, P = mwis_slide()
    Q = mwis_Q(w, edges, P)
    assert (Q == [[-4, 4, 0, 4], [4, -6, 4, 0], [0, 4, -5, 4], [4, 0, 4, -2]]).all()
    e, am = ground(Q)
    assert e == -9 and am == [[1, 0, 1, 0]]

    wt, val, C, P, co = knapsack_slide()
    Q, c, _ = knapsack_Q(val, wt, C, P, co)
    Q5 = [[-174, 42, 56, 14, 28, 56], [42, -235, 84, 21, 42, 84], [56, 84, -285, 28, 56, 112],
          [14, 21, 28, -91, 14, 28], [28, 42, 56, 14, -168, 56], [56, 84, 112, 28, 56, -280]]
    assert (Q == Q5).all() and c == 343
    e, am = ground(Q, c)
    assert e == -354 and am == [[1, 0, 1, 1, 0, 0]]

    print("helpers_ladder self-test OK")
