# Experimental settings

Proposed, MAPPO-PF and MAPPO-Vanilla use the trainer in [`safe_marl/`](../safe_marl). MAPPO-PF subtracts
the agent's local cost and the shared global cost from its reward with coefficient 1.0, and MAPPO-Vanilla
uses the environment reward unchanged. The multiplier settings apply to Proposed only.

| Item | Setting |
|---|---|
| Observations and critic inputs | Local observation $`o_{i,t}`$ of 8 (C1, C2) or 11 (C3) features; exact components are listed in [environments.md](environments.md). Reward and global-cost critics: concatenated observations of all agents (24, 40 and 132 features in C1, C2, C3). Local-cost critic: $`o_{i,t}`$. |
| Action | The Gaussian policy outputs $`u_{i,t}\in\mathbb{R}^{d}`$; the environment receives $`a_{i,t}=\mathrm{clip}((u_{i,t}+1)/2,0,1)`$, with $`d=4`$ (C1, C2) or $`d=5`$ (C3). Device set-points are defined in [environments.md](environments.md). |
| Episode length | One day of 24 hourly steps. |
| Stochastic profiles | Hourly load and PV multipliers $`\xi\sim\mathcal{N}(1,0.05^2)`$, i.i.d., clipped at 0, applied to fixed base profiles; one load sequence per case (C1, C2) or per prosumer (C3); a new day at every reset; rollout worker $`k`$ seeded with `seed + 10000 k`. |
| Reward scaling | C1: $`0.003\times`$ saving in $ relative to buying all demand from the grid; C2: $`0.008\times`$ the same saving; C3: $`-0.008\times`$ community cost in $, minus $`2\times`$ the agent's relative unmet EV energy at the end of the day. |
| Cost scaling | C1: local $`100\times`$ squared voltage violation (p.u.²), global squared import excess (MW²). C2: local and global $`100\times`$ squared voltage violation. C3: local and global $`5\times`$ relative excess of the cumulative emissions over the cap. |
| Normalization | PopArt normalization of the targets of each critic; advantages centered per channel and divided by the standard deviation of the hybrid advantage. |
| Networks | Per agent: one actor and three critics (reward, global cost, local cost), no parameter sharing. Input LayerNorm, 3 hidden layers of 256 units with ReLU and LayerNorm, orthogonal initialization. Actor: linear mean (gain 0.01) and state-independent standard deviation $`0.5\,\mathrm{sigmoid}(w)`$, $`w_0=1`$. |
| Optimizer | Adam ($`\epsilon=10^{-5}`$), learning rate $`3\times10^{-4}`$ (actor) and $`5\times10^{-4}`$ (critics), gradient-norm clipping at 10. |
| PPO | Clipping $`\epsilon=0.2`$; 5 epochs on one batch of 96 transitions (4 days × 24 h) per iteration; entropy coefficient 0.008; Huber value loss ($`\delta=10`$) with value clipping; $`\gamma=0.99`$, GAE $`\lambda=0.95`$. |
| Multipliers | $`\eta_\lambda=5\times10^{-4}`$; $`\lambda_i^{\ell}=0.5`$ and $`\lambda^{g}=0.78`$ at initialization; $`d_i^{\ell}=d^{g}=0`$; one update per iteration from the discounted episode costs of the rollout. |
| Seeds | Training seeds 1 to 5. |
| Training budget | 480,096 environment steps per run: 5,001 iterations × 4 parallel days × 24 h (20,004 days). |
| Compute | CPU (AMD Ryzen Threadripper PRO 7985WX), one learner and 4 rollout processes per run; 1.5 h (C1), 2.6 h (C2) and 1.4 h (C3) per run. |

## MAPPO-Lag

MAPPO-Lag uses the trainer in [`mappo_lag/`](../mappo_lag), started with
`python -m mappo_lag.train --case C1 --seed 1`. `python -m safe_marl.train --case C1 --method mappo_lag --seed 1`
calls the same entry point.

| Item | Setting |
|---|---|
| PPO | 1 epoch, 24 mini-batches of 4 transitions. |
| Learning rates | Actor and critics: $`5\times10^{-4}`$. |
| Multipliers | Initial values 1.2, 0.78 and 0.5; step size $`5\times10^{-4}`$; both constraint budgets are zero. |
| Episode and rollouts | 24 hourly steps, 4 rollout processes. |
| Seeds and budget | Seeds 1 to 5; 480,096 environment steps per run. |

## Evaluation

The final policies act deterministically (mean action) on 20 test days with unseen load and PV
realizations, generated with seeds 20261005 to 20261024 ([`evaluation/`](../evaluation)).

| Metric | Definition |
|---|---|
| $`\bar R`$ | daily sum of the mean environment reward across agents, including the terminal EV penalty in C3 |
| $`C_g`$, $`C_l`$ | daily global cost and daily local cost summed over agents |
| $`N_v`$ | hours of the day with $`C_g>0`$ or $`C_l>0`$ |
| $`t`$ | computation time per decision for all agents; for distributed MPC including all ADMM iterations |

Results are averaged over the 20 days for each training seed and then over the 5 seeds
([`results/paper/rl_per_seed.csv`](../results/paper/rl_per_seed.csv)).

## Training curves

Fig. 2 shows, per training episode, the daily reward and the daily global and local costs. Each seed's
curve is smoothed by a centered moving average over 201 episodes; lines are the mean over 5 seeds and bands
±1 standard error ([`scripts/plot_training_curves.py`](../scripts/plot_training_curves.py)).

`bash scripts/download_curves.sh` downloads the curves of all 60 runs to
`results/paper/training_curves.npz`, and `python scripts/plot_training_curves.py --paper` plots them. The
reward curves show the environment reward; for MAPPO-PF this is the reward before the penalty
(`raw_rewards`). The local-cost curves sum the daily local cost over agents. Without `--paper`, the script
plots new runs from `results/runs/`. `--usetex` sets the paper's LaTeX fonts and needs a local TeX
installation.
