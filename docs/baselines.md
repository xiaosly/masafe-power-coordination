# MPC baselines

[`baselines/`](../baselines) contains two model-based online controllers for the three cases. Both are built
on the optimization model below and re-plan at every hour.

| | Decentralized MPC | Distributed MPC (ADMM) |
|---|---|---|
| Global (coupling) constraint | not modeled by the agents | coordinated by ADMM |
| Online communication | not required | required: one exchange between the agents and a coordinator per ADMM iteration, 16 to 33 rounds per decision |
| Command (C1) | `python -m baselines.run --case C1 --method dec_mpc` | `python -m baselines.run --case C1 --method dist_mpc` |

Use `--case C2` or `--case C3` for the other cases. `python scripts/evaluate_all.py --mpc-only` runs both
controllers on all three cases. The optimization model is in `baselines/model.py`, the two controllers are in
`baselines/admm.py` and the entry point is `baselines/run.py`. Both controllers are scored with the same
ledger as the RL policies (`evaluation/ledger.py`); the 120 per-day results are in
`results/paper/mpc_per_day.csv`.

Distributed MPC requires online communication. Every hour, the agents and a coordinator (the TSO in C1, the
network operator in C2, the community operator in C3) exchange their coupling variables until ADMM converges.

## Rolling horizon

At every hour $`t`$ each controller plans the remaining day $`\mathcal{H}_t=\{t,\dots,23\}`$ and applies the
set-points of hour $`t`$. Forecasts: realized load and PV for hour $`t`$, base profiles for the later hours.

## Optimization model

C1 and C2 use the radial branch-flow (DistFlow) model with the second-order-cone relaxation. For every network,
hour $`t`$ and line $`(i,j)`$, with squared voltages $`v`$, squared currents $`I`$ and sending-end flows
$`(P_{ij},Q_{ij})`$:

```math
\begin{aligned}
&v_{j,t}=v_{i,t}-2\big(r_{ij}P_{ij,t}+x_{ij}Q_{ij,t}\big)+\big(r_{ij}^2+x_{ij}^2\big)I_{ij,t},\qquad
P_{ij,t}^2+Q_{ij,t}^2\le v_{i,t}\,I_{ij,t},\\
&P_{ij,t}-r_{ij}I_{ij,t}+p^{\mathrm{gen}}_{j,t}=\textstyle\sum_{k:(j,k)}P_{jk,t}+p^{\mathrm{load}}_{j,t},\qquad
Q_{ij,t}-x_{ij}I_{ij,t}+q^{\mathrm{gen}}_{j,t}=\textstyle\sum_{k:(j,k)}Q_{jk,t}+q^{\mathrm{load}}_{j,t},\\
&0.95^2\le v_{j,t}\le 1.05^2 .
\end{aligned}
```

The loads are $`(p_b,q_b)\,\hat\ell_t(1-\rho_t)`$ with the curtailment ratio $`\rho_t`$ as a decision
variable, the DG limits are those of the environment, and each battery has charge/discharge binaries
$`u_t\in\{0,1\}`$: $`0\le P^{\mathrm{ch}}_t\le\bar P^{B}u_t`$, $`0\le P^{\mathrm{dis}}_t\le\bar P^{B}(1-u_t)`$,
$`E_{t}=E_{t-1}+\eta P^{\mathrm{ch}}_t-P^{\mathrm{dis}}_t/\eta`$. The C1 model has three DSO feeders and the coupling
constraint $`\sum_j P^{\mathrm{in}}_{j,t}\le 8`$ MW. The C2 model has five MG networks and the 33-bus network. The MG exchanges
$`(P^{\mathrm{in}}_{i,t},Q^{\mathrm{in}}_{i,t})`$ are loads at bus $`i`$ of the network, and the MG root voltage of
hour $`t+1`$ equals the network voltage at bus $`i`$ of hour $`t`$.

C3 uses a mixed-integer linear model with the energy balance
$`P^{\mathrm{grid}}+P^{\mathrm{gas}}+P^{\mathrm{dis}}+\mathrm{PV}\ge L+P^{\mathrm{ch}}+F+P^{\mathrm{EV}}`$, battery binaries,
EV energy dynamics with $`E^{\mathrm{EV}}_{24}\ge0.3\bar E^{\mathrm{EV}}+N`$, local caps
$`M_{i}^{\mathrm{past}}+\sum_{t\in\mathcal{H}}m_{i,t}\le\bar M_i`$ and the community cap
$`M^{\mathrm{past}}+\sum_i\sum_{t\in\mathcal{H}}m_{i,t}\le\bar M`$.

The objective is the operating cost $`\sum_{t\in\mathcal{H}}\sum_i f_{i,t}`$ of [environments.md](environments.md),
plus a network-loss term $`w\sum r_{ij}I_{ij,t}`$ in C1 ($`w=0.01`$) and C2 ($`w=30`$).

## Distributed MPC by ADMM

Each agent keeps its own part of the model; the coupling constraints are coordinated by ADMM
([`baselines/admm.py`](../baselines/admm.py)).

- C1 and C3 use sharing ADMM. Agent $`i`$ owns a coupling vector $`x_i`$ (C1: its import of every remaining
  hour; C3: its emissions of the remaining day), and the coordinator enforces $`\sum_i x_i\le\text{cap}`$:

  ```math
  x_i^{k+1}=\arg\min_{x_i\in\mathcal{X}_i} f_i(x_i)+\tfrac{\rho}{2}\big\lVert x_i-z_i^{k}+u_i^{k}\big\rVert^2,\qquad
  z^{k+1}=\Pi_{\{\sum_i z_i\le\text{cap}\}}\big(x^{k+1}+u^{k}\big),\qquad
  u_i^{k+1}=u_i^{k}+x_i^{k+1}-z_i^{k+1}.
  ```
- C2 uses consensus ADMM between the five MGs and the network operator on the exchanges
  $`(P^{\mathrm{in}}_{i,t},Q^{\mathrm{in}}_{i,t})`$ and the MG root voltages, with relaxed battery binaries,
  over-relaxation $`\alpha=1.6`$ and penalties $`\rho_P=\rho_Q=100`$, $`\rho_V=10^4`$.
- Stopping rule on the primal and dual residuals (Boyd et al., 2011); in C1 and C3, $`\rho`$ follows residual
  balancing during the first 50 iterations; warm start from the previous hour.

| | C1 | C2 | C3 |
|---|---|---|---|
| ADMM | sharing (hourly import cap) | consensus (MG-network exchange and voltage) | sharing (community carbon cap) |
| Max. iterations | 300 | 500 | 300 |
| $`(\epsilon_{\mathrm{abs}},\epsilon_{\mathrm{rel}})`$ | $`(10^{-4},10^{-3})`$ | $`(10^{-4},10^{-3})`$ for $`P,Q`$; $`10^{-5}`$ for $`V`$ | $`(10^{-3},10^{-4})`$ |
| Communication rounds per decision | 30 | 16 | 33 |
| Time per decision | 13.8 s | 4.9 s | 0.37 s |
| Time per decision, decentralized MPC | 0.40 s | 0.11 s | 0.02 s |

All subproblems are solved by Gurobi 13 with one thread. The time per decision is the computation time of
all agents solved one after another.

## Decentralized MPC

The same per-agent models without coordination: C1 without the import cap, C3 without the community cap, C2
with the MG root voltage held at its measured value and without the network model.
