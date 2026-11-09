# 2026 Quantum Accelerator — Day 1 notebooks

Hands-on material for the quantum-inspired combinatorial optimization day.
Design rationale and full agenda: `../QuantumAccelerator_Day1_Design.md`.

## Environment

The Littlest JupyterHub on an Azure VM (`https://135.116.82.187`, logins `qubo01`-`qubo15`).
Attendees open one link and the notebooks are pulled from this repository into their home
directory. Operations: `2026 Quantum Accelerator\Day 1 - Hands-on Workshop\RUNBOOK_TLJH.md`.
No tokens, no cloud solver accounts — every exercise runs on the VM's CPU with
`dwave-samplers` simulated annealing and OR-Tools CP-SAT as the honest baseline.

Quick check that the environment is ready: `00_environment_check.ipynb`, or

```bash
python -c "import pyqubo, dimod, ortools; from dwave.samplers import SimulatedAnnealingSampler; print('ok')"
```

## The QUBO ladder (L1–L5)

Five short notebooks that follow the slide deck's exercises one by one, each with the slide
instance rebuilt by hand (numpy Q, every answer with `dimod.ExactSolver`, the same Q from
PyQUBO) and a case too big for a slide (simulated annealing against an honest reference).
One new idea per step:

| File | Step | New idea | Case B | Runtime |
|---|---|---|---|---|
| `L1_maxcut.ipynb` | Max-Cut | nothing to encode | n = 120 random graph vs CP-SAT (30 s cap) | ~35 s |
| `L2_number_partitioning.ipynb` | number partitioning | square a difference | n = 40 at three bit-sizes vs DP | ~5 s |
| `L3_max2sat.ipynb` | Max-2-SAT | coefficients from logic, keep the constant | n = 100, m = 400 vs CP-SAT (30 s cap) | ~35 s |
| `L4_mwis.ipynb` | independent set | a penalty per edge, P > max w | unit-disk n = 150 vs CP-SAT, P sweep | ~10 s |
| `L5_knapsack.ipynb` | knapsack | slack bits, the encoding tax | the notebook-02 bag, P sweep | ~5 s |

Shared helpers: `helpers_ladder.py` (`python helpers_ladder.py` runs its self-test).
Generators: `build_notebook_L1.py` … `L5.py` (`--solved` also writes the executed copy).
Calibration behind the quoted numbers: `ladder_plan/calibrate_ladder.py`.

## The ladder in one paragraph

Five steps, one new idea each: Max-Cut (Q read off the graph), number partitioning (square a
linear expression), Max-2-SAT (coefficients from logic, keep the constant), independent set (a
penalty per edge with the bound P > max w) and knapsack (slack bits for a global inequality).
Every slide matrix is reproduced digit for digit, solved with `dimod.ExactSolver` and
cross-checked against PyQUBO. Three case-B signals are worth remembering. Max-Cut: SA reaches
4 061 against CP-SAT's 3 958, and CP-SAT's bound is useless (it equals the total weight).
Number partitioning: same n, same solver, success drops from 30 % to 0 % as the numbers grow from
7 to 20 bits. MWIS: nothing is feasible below P = max w + 1, then quality decays as P grows.


