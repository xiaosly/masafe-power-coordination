# Environments: CMG-LG formulation of the three cases

Each case is a constrained Markov game with local observations and separate local and global costs
(CMG-LG):

```math
\mathcal{G}^{\mathrm{LG}}=\big\langle \mathcal{I},\mathcal{S},\{\mathcal{O}_i\}_{i\in\mathcal{I}},\{\mathcal{A}_i\}_{i\in\mathcal{I}},\mathcal{P},\{r_i\}_{i\in\mathcal{I}},\{c_i^{\ell}\}_{i\in\mathcal{I}},c^{g},\gamma\big\rangle .
```

Here $`r_i`$ is the reward returned to agent $`i`$, $`c_i^{\ell}`$ measures its local constraint violation,
and $`c^{g}`$ is one scalar global cost shared by all agents. The discounted cost budgets are
$`d_i^{\ell}=d^{g}=0`$; the constrained objective and policy updates are given in [algorithm.md](algorithm.md).

Power flow, energy balance and storage dynamics define the transition $`\mathcal{P}`$. Device bounds
are applied when decoding actions and updating storage. Voltage and import violations give squared
costs in C1/C2; carbon-cap violations give scaled relative-excess costs in C3. The equations and
parameters below follow [`power_envs/`](../power_envs).

| | C1: TSO-DSO | C2: microgrids | C3: carbon community |
|---|---|---|---|
| Code | [`c1_tso_dso.py`](../power_envs/c1_tso_dso.py) `TSODSOEnv` | [`c2_microgrids.py`](../power_envs/c2_microgrids.py) `MGVoltageEnv` | [`c3_carbon.py`](../power_envs/c3_carbon.py) `CarbonMarketEnv` |
| Agents $`\mathcal{I}`$ | 3 DSOs | 5 microgrids (MGs) | 12 prosumers |
| Observation / action dim. | 8 / 4 | 8 / 4 | 11 / 5 |
| Local constraint | DSO bus voltages in [0.95, 1.05] p.u. | MG bus voltages in [0.95, 1.05] p.u. | prosumer daily carbon cap |
| Global constraint | total TSO→DSO import ≤ 8 MW | distribution-network bus voltages in [0.95, 1.05] p.u. | community daily carbon cap |
| Physics | AC power flow (DSOs), DC power flow (TSO) | AC power flow (MGs and network) | energy and carbon balance |
| Units | MW, MWh, $ | MW, MWh, $ | kW, kWh, kg CO₂, $ |

## Common conventions

One episode is one day of $`T=24`$ hourly steps, $`t=0,\dots,23`$ (hour $`t`$ covers
$`[t, t{+}1)`$ h); there is no early termination. Storage states start at fixed values every day.

The full simulator state $`s_t`$ includes the hour, storage states,
cumulative emissions where applicable, stored exchange and coordination signals, and the daily
load/PV profiles sampled at reset. Agent $`i`$ receives the observation $`o_{i,t}=\omega_i(s_t)`$
listed for its case. The joint action $`\mathbf{a}_t=(a_{i,t})_{i\in\mathcal{I}}`$ drives
$`s_{t+1}\sim\mathcal{P}(\cdot\mid s_t,\mathbf{a}_t)`$. Proposed's centralized critics use concatenated agent
observations; the sampled future profiles remain internal to the simulator.

Each policy outputs $`u_{i,t}\in\mathbb{R}^{d}`$ (Gaussian); the environment receives
$`a_{i,t}=\mathrm{clip}\big((u_{i,t}+1)/2,\,0,\,1\big)\in[0,1]^{d}`$, which is scaled to device set-points
as described below. Storage set-points are also limited by the state of charge.

All cases share a base load shape $`\bar{\ell}_t`$ (peak 1.0 at 18:00) and a PV
shape $`\bar{s}^{\mathrm{PV}}_t=\sin\!\big(\pi(t-6)/13\big)`$ for $`6\le t\le 19`$ (else 0), see
[`profiles.py`](../power_envs/profiles.py). At every reset a new day is drawn,

```math
\ell_t=\big[\bar{\ell}_t\,\xi^{\ell}_t\big]_+,\qquad s^{\mathrm{PV}}_t=\big[\bar{s}^{\mathrm{PV}}_t\,\xi^{\mathrm{PV}}_t\big]_+,\qquad
\xi^{\ell}_t,\ \xi^{\mathrm{PV}}_t\ \overset{\text{i.i.d.}}{\sim}\ \mathcal{N}(1,\,0.05^2),
```

with one load noise sequence per case (C1, C2) or per prosumer (C3). Test day $`n`$ is generated with
`np.random.seed(20261005 + n)`, $`n=0,\dots,19`$.

C1/C2 return the same reward to all agents: scaled operating-cost savings against
an action-independent reference. C3 returns the shared negative operating cost, minus each agent's
terminal EV-shortfall penalty. Its EV charging requirement is handled by this reward term; the
local/global cost channels measure carbon-cap violations. Evaluation reports the daily sum of the
mean reward across agents and records operating cost separately.

Constraint costs are evaluated after each joint action. Local and global refer to the operating limits
being measured; the resulting costs can depend on the coupled physical state. If a feeder or MG power
flow fails, its local cost is 100; if the C2 distribution-network power flow fails, its global cost is 100.

## C1: TSO-DSO coordination

### System

Three DSOs $`j\in\{7,27,29\}`$ (agents, named by their TSO bus) are each an IEEE 33-bus feeder
(pandapower `case33bw`, 12.66 kV, slack at 1.05 p.u.). DSO $`j`$ scales the base loads $`(p_b,q_b)`$ by
$`\kappa_j`$. Each DSO has a dispatchable DG at bus 24, a battery at bus 12 and curtailable load. The
DSO imports $`P^{\mathrm{tie}}_{j,t}`$ (slack injection) are loads of the IEEE 30-bus transmission system
(`case30`, DC power flow); the transmission operator limits the total import.

### Action

Each DSO chooses $`a_{j,t}\in[0,1]^4`$:

| | Set-point |
|---|---|
| $`a^1`$ | DG active power $`P^{G}_{j,t}=a^1\,\bar{P}^{G}_j`$ |
| $`a^2`$ | DG reactive power $`Q^{G}_{j,t}=(2a^2-1)\,\bar{Q}^{G}_j`$ |
| $`a^3`$ | battery, $`\beta=2a^3-1`$: $`P^{\mathrm{ch}}_{j,t}=[\beta]_+\min\{\bar{P}^{B}_j,(\bar{E}_j-E_{j,t})/\eta\}`$, $`P^{\mathrm{dis}}_{j,t}=[-\beta]_+\min\{\bar{P}^{B}_j,(E_{j,t}-0.1\bar{E}_j)\,\eta\}`$ |
| $`a^4`$ | load curtailment ratio $`\rho_{j,t}=a^4\bar{\rho}_j`$ |

### Transition

The bus loads are $`\kappa_j(p_b,q_b)\,\ell_t\,(1-\rho_{j,t})`$. The AC power flow of each feeder gives
the voltages $`V_{j,b,t}`$ and the import $`P^{\mathrm{tie}}_{j,t}`$, and the battery energy follows
$`E_{j,t+1}=\min\{\max\{E_{j,t}+\eta P^{\mathrm{ch}}_{j,t}-P^{\mathrm{dis}}_{j,t}/\eta,\ 0.1\bar{E}_j\},\ \bar{E}_j\}`$,
$`E_{j,0}=0.5\bar{E}_j`$.

### Reward

All DSOs receive the same reward. With $`D_j=\kappa_j\sum_b p_b`$ the DSO's base demand and $`\pi_t`$ the import
tariff,

```math
r_t = 0.003\Big(\underbrace{\pi_t\sum_{j} D_j\ell_t}_{B_t}\;-\;\sum_{j}\underbrace{\big(c^{G}_jP^{G}_{j,t}+c^{C}_j\rho_{j,t}D_j\ell_t+c^{B}\big(P^{\mathrm{ch}}_{j,t}+P^{\mathrm{dis}}_{j,t}\big)+\pi_t\big[P^{\mathrm{tie}}_{j,t}\big]_+\big)}_{f_{j,t}}\Big).
```

### Costs

The local cost is the squared voltage violation over the 33 buses of DSO $`j`$; the global cost is the squared
excess over the import cap (MW²):

```math
c^{\ell}_{j,t}=100\sum_{b}\Big(\big[0.95-V_{j,b,t}\big]_+^2+\big[V_{j,b,t}-1.05\big]_+^2\Big),\qquad
c^{g}_{t}=\Big(\Big[\sum_{j}P^{\mathrm{tie}}_{j,t}-8\Big]_+\Big)^2 .
```

### Observation

Each DSO observes $`o_{j,t}\in\mathbb{R}^8`$: $`t/24`$; $`\ell_t/2`$; $`s^{\mathrm{PV}}_t`$; $`P^{\mathrm{tie}}_{j,t-1}/10`$;
$`\mathrm{clip}(\mu_{t}/50,-1,1)`$ with $`\mu_{t+1}=\mu_t+0.1\,c^{g}_t/3`$ (accumulated import-cap signal);
$`\mathrm{clip}\big(\sum_k P^{\mathrm{tie}}_{k,t-1}/20,-1,1\big)`$; $`E_{j,t}/\bar{E}_j`$; $`\rho_{j,t-1}`$
(previous-hour quantities are 0 at $`t=0`$).

### Parameters

Battery efficiency $`\eta=0.95`$, cycling cost $`c^{B}=12`$ $/MWh, import cap 8 MW.

| DSO (TSO bus) | $`\kappa_j`$ | $`\bar{P}^{G}`$ (MW) | $`\bar{Q}^{G}`$ (Mvar) | $`c^{G}`$ ($/MWh) | $`\bar{E}`$ (MWh) | $`\bar{P}^{B}`$ (MW) | $`\bar{\rho}`$ | $`c^{C}`$ ($/MWh) |
|---|---|---|---|---|---|---|---|---|
| 7 | 1.2 | 3.5 | 2.5 | 80 | 5.0 | 2.5 | 0.15 | 400 |
| 27 | 1.5 | 4.0 | 3.0 | 90 | 8.0 | 4.0 | 0.20 | 450 |
| 29 | 2.0 | 5.0 | 3.5 | 85 | 6.0 | 3.0 | 0.25 | 350 |

Tariff $`\pi_t`$ ($/MWh, $`t=0..23`$): 50 50 45 45 50 70 · 110 150 180 180 160 140 · 130 130 150 170 190 190 · 170 150 110 90 70 50.

### Operation problem

$`\min\sum_t\sum_j f_{j,t}`$ subject to the device limits and storage dynamics
above, the AC power flow of every feeder, $`0.95\le V_{j,b,t}\le1.05`$ (local) and
$`\sum_j P^{\mathrm{tie}}_{j,t}\le 8`$ MW (global).

## C2: microgrids in a distribution network

### System

The agents are five MGs $`i\in\{8,17,24,25,32\}`$, named by their bus in the IEEE 33-bus
distribution network (slack at 1.05 p.u.). Each MG is a 7-bus, 12.66 kV network with two branches
0-1-2-3 and 0-4-5-6, loads at buses 1 to 6 (total $`D=1.5`$ MW), a DG at bus 6, a battery at bus 3 and
curtailable load. The MG's point of common coupling (bus 0) is its slack bus, held at the network
voltage of its tie bus in the previous hour, $`V^{\mathrm{tie}}_{i,t-1}`$ ($`=1.05`$ at $`t=0`$). The MG
exchanges $`(P^{\mathrm{tie}}_{i,t},Q^{\mathrm{tie}}_{i,t})`$ appear as loads at bus $`i`$ of the network,
on top of its base loads $`(p_b,q_b)\ell_t`$. Each MG follows the shifted load shape
$`\ell^{i}_t=\ell_{(t-\sigma_i)\bmod 24}`$.

### Action

Each MG chooses $`a_{i,t}\in[0,1]^4`$:

| | Set-point |
|---|---|
| $`a^1`$ | DG active power $`P^{G}_{i,t}=a^1\bar{P}^{G}_i`$ (thermal MG) or $`a^1\bar{P}^{G}_i s^{\mathrm{PV}}_t`$ (PV MG) |
| $`a^2`$ | DG reactive power $`Q^{G}_{i,t}=(2a^2-1)\,\bar{Q}^{G}_i`$ |
| $`a^3`$ | battery power (positive = charging) $`P^{B}_{i,t}=\mathrm{clip}\big((2a^3-1)\bar{P}^{B}_i,\ -(\mathrm{SOC}_{i,t}-0.1)\bar{E}_i\eta,\ (0.9-\mathrm{SOC}_{i,t})\bar{E}_i/\eta\big)`$ |
| $`a^4`$ | load curtailment ratio $`\rho_{i,t}=a^4\bar{\rho}_i`$ |

### Transition

The MG loads are $`(p_b,q_b)\,\ell^{i}_t(1-\rho_{i,t})`$. The AC power flow of each MG gives
$`V_{i,b,t}`$ and $`(P^{\mathrm{tie}}_{i,t},Q^{\mathrm{tie}}_{i,t})`$; then the AC power flow of the network gives
$`V^{N}_{b,t}`$ and $`V^{\mathrm{tie}}_{i,t}=V^{N}_{i,t}`$. The state of charge follows
$`\mathrm{SOC}_{i,t+1}=\mathrm{clip}\big(\mathrm{SOC}_{i,t}+(\eta[P^{B}_{i,t}]_+-[-P^{B}_{i,t}]_+/\eta)/\bar{E}_i,\ 0.1,\ 0.9\big)`$,
$`\mathrm{SOC}_{i,0}=0.5`$.

### Reward

All MGs receive the same reward. With $`\tau=\max(t-1,0)`$,

```math
r_t = 0.008\Big(\pi_\tau\sum_{i}D\,\ell^{i}_\tau-\sum_{i}\underbrace{\big(c^{G}_iP^{G}_{i,t}+\pi_\tau\big[P^{\mathrm{tie}}_{i,t}\big]_+
+(c^{C}-c^{DR})\,\rho_{i,t}D\,\ell^{i}_t+c^{B}\big|P^{B}_{i,t}\big|\big)}_{f_{i,t}}\Big).
```

### Costs

The local cost covers the 7 buses of MG $`i`$, and the global cost the 33 network buses:

```math
c^{\ell}_{i,t}=100\sum_{b=0}^{6}\Big(\big[0.95-V_{i,b,t}\big]_+^2+\big[V_{i,b,t}-1.05\big]_+^2\Big),\qquad
c^{g}_{t}=100\sum_{b=0}^{32}\Big(\big[0.95-V^{N}_{b,t}\big]_+^2+\big[V^{N}_{b,t}-1.05\big]_+^2\Big).
```

### Observation

Each MG observes $`o_{i,t}\in\mathbb{R}^8`$: $`t/24`$; $`\ell_t/2`$; $`s^{\mathrm{PV}}_t`$; $`(V^{\mathrm{tie}}_{i,t-1}-0.9)/0.2`$;
$`|P^{\mathrm{tie}}_{i,t-1}|/5`$; $`\mathrm{clip}(\mu_t/50,-1,1)`$ with $`\mu_{t+1}=\mu_t+0.1\,c^{g}_t/5`$;
$`\mathrm{clip}(\delta V_{i,t}/50,-1,1)`$; $`\mathrm{SOC}_{i,t}`$.
The voltage-deviation signal is $`\delta V_{i,t}=V^{\mathrm{tie}}_{i,t-1}-1`$ for $`t>0`$;
at reset, $`\delta V_{i,0}=\mu_0=P^{\mathrm{tie}}_{i,-1}=0`$ and $`V^{\mathrm{tie}}_{i,-1}=1.05`$.

### Parameters

$`\eta=0.95`$, $`c^{B}=15`$ $/MWh, curtailment cost $`c^{C}=500`$ $/MWh less demand-response
revenue $`c^{DR}=50`$ $/MWh, $`\bar{Q}^{G}_i=1.4\,\bar{P}^{G}_i`$.

| MG (bus) | DG | $`\bar{P}^{G}`$ (MW) | $`c^{G}`$ ($/MWh) | $`\bar{E}`$ (MWh) | $`\bar{P}^{B}`$ (MW) | $`\bar{\rho}`$ | load shift $`\sigma_i`$ (h) |
|---|---|---|---|---|---|---|---|
| 8 | thermal | 1.5 | 85 | 0.5 | 0.25 | 0.20 | 0 |
| 17 | thermal | 2.0 | 90 | 1.0 | 0.50 | 0.15 | −2 |
| 24 | thermal | 2.0 | 80 | 0.8 | 0.40 | 0.25 | 4 |
| 25 | PV | 1.0 | 25 | 1.2 | 0.60 | 0.10 | −3 |
| 32 | PV | 2.0 | 30 | 0.6 | 0.30 | 0.30 | 2 |

MG lines (p.u. on 1 MVA): $`r`$ = 0.010, 0.015, 0.020 (lines 0-1, 1-2, 2-3) and 0.012, 0.015, 0.020 (lines 0-4, 4-5, 5-6);
$`x`$ = 0.005, 0.007, 0.010 and 0.006, 0.008, 0.010. MG loads (MW / Mvar) at buses 1 to 6: 0.20/0.05,
0.30/0.10, 0.25/0.05, 0.20/0.05, 0.30/0.10, 0.25/0.05.
Tariff $`\pi_t`$ ($/MWh): 60 60 55 55 60 80 · 120 160 200 200 180 160 · 140 140 160 180 200 200 · 180 160 120 100 80 60.

### Operation problem

$`\min\sum_t\sum_i f_{i,t}`$ subject to device limits, storage dynamics, the AC power
flow of every MG and of the network (coupled through the exchanges and the PCC voltage),
$`0.95\le V_{i,b,t}\le1.05`$ (local) and $`0.95\le V^{N}_{b,t}\le1.05`$ (global).

## C3: prosumer community with carbon caps

### System

The agents are twelve prosumers $`i=1,\dots,12`$, each with load, PV, a battery, a gas DG, grid import,
controllable flexible load that earns revenue, and an EV with a daily charging requirement. Emissions
come from grid imports (hourly carbon intensity $`\kappa_t`$) and gas (0.5 kg/kWh). Each prosumer has a
daily carbon cap $`\bar{M}_i`$; the community cap $`\bar{M}=11{,}354`$ kg is tighter than
$`\sum_i\bar{M}_i=12{,}105`$ kg. Prosumer $`i`$ has load $`L_{i,t}=\bar{L}_i\,\bar{\ell}_{(t-\sigma_i)\bmod 24}\,\xi^{i}_t`$ (with its own
noise), PV output $`\mathrm{PV}_{i,t}=\bar{P}^{PV}_i s^{\mathrm{PV}}_t`$ and flexible-load availability $`\bar{F}_i\phi_t`$.

### Action

Each prosumer chooses $`a_{i,t}\in[0,1]^5`$:

| | Set-point |
|---|---|
| $`a^1`$ | gas DG $`P^{\mathrm{gas}}_{i,t}=a^1\bar{P}^{\mathrm{gas}}_i`$ |
| $`a^2`$ | battery, $`\beta=2a^2-1`$: $`P^{\mathrm{ch}}=[\beta]_+\min\{\bar{E}_i/4,(\bar{E}_i-E_{i,t})/\eta\}`$, $`P^{\mathrm{dis}}=[-\beta]_+\min\{\bar{E}_i/4,(E_{i,t}-0.1\bar{E}_i)\eta\}`$ |
| $`a^3`$ | grid purchase $`a^3\max\{2L_{i,t},1\}`$; any remaining deficit is bought automatically |
| $`a^4`$ | flexible load served $`F_{i,t}=a^4\bar{F}_i\phi_t`$ |
| $`a^5`$ | EV charging $`P^{\mathrm{EV}}_{i,t}=a^5\min\{\bar{P}^{\mathrm{EV}}_i,(\bar{E}^{\mathrm{EV}}_i-E^{\mathrm{EV}}_{i,t})/\eta\}`$ |

### Transition

The grid import is
$`P^{\mathrm{grid}}_{i,t}=\max\big\{a^3\max\{2L_{i,t},1\},\ L_{i,t}+P^{\mathrm{ch}}+F_{i,t}+P^{\mathrm{EV}}-\mathrm{PV}_{i,t}-P^{\mathrm{gas}}-P^{\mathrm{dis}}\big\}`$;
surplus energy is spilled without compensation. The battery and EV energies follow
$`E_{i,t+1}=\min\{\max\{E_{i,t}+\eta P^{\mathrm{ch}}-P^{\mathrm{dis}}/\eta,0.1\bar{E}_i\},\bar{E}_i\}`$, $`E_{i,0}=0.5\bar{E}_i`$ and
$`E^{\mathrm{EV}}_{i,t+1}=\min\{E^{\mathrm{EV}}_{i,t}+\eta P^{\mathrm{EV}},\bar{E}^{\mathrm{EV}}_i\}`$, $`E^{\mathrm{EV}}_{i,0}=0.3\bar{E}^{\mathrm{EV}}_i`$.
The emissions are $`m_{i,t}=\kappa_tP^{\mathrm{grid}}_{i,t}+0.5P^{\mathrm{gas}}_{i,t}`$, with cumulative values
$`M_{i,t}=\sum_{\tau\le t}m_{i,\tau}`$ and $`M_t=\sum_iM_{i,t}`$.

### Reward

The reward combines the shared cost of all prosumers with an agent-specific terminal penalty for unmet
EV energy ($`N_i`$ is the daily EV need):

```math
r_{i,t}=-0.008\sum_{k}\underbrace{\big(\pi_tP^{\mathrm{grid}}_{k,t}+0.08P^{\mathrm{gas}}_{k,t}+0.02(P^{\mathrm{ch}}_{k,t}+P^{\mathrm{dis}}_{k,t})-0.20F_{k,t}\big)}_{f_{k,t}}
\;-\;2\cdot\mathbb{I}[t=23]\,\frac{\big[0.3\bar{E}^{\mathrm{EV}}_i+N_i-E^{\mathrm{EV}}_{i,24}\big]_+}{N_i}.
```

### Costs

Both costs are the relative excess of the cumulative emissions over the cap. Once a cap is exceeded, the
cost is incurred in every remaining hour:

```math
c^{\ell}_{i,t}=5\,\frac{\big[M_{i,t}-\bar{M}_i\big]_+}{\bar{M}_i},\qquad
c^{g}_{t}=5\,\frac{\big[M_t-\bar{M}\big]_+}{\bar{M}} .
```

### Observation

Each prosumer observes $`o_{i,t}\in\mathbb{R}^{11}`$: $`t/24`$; $`L_{i,t}/300`$; $`\mathrm{PV}_{i,t}/150`$;
$`E_{i,t}/\bar{E}_i`$; $`M_{i,t-1}/\bar{M}_i`$; $`\kappa_t`$; $`[1-M_{t-1}/\bar{M}]_+`$;
$`2[M_{t-1}/\bar{M}-0.5]_+`$ (carbon-price signal); $`\phi_t`$; $`E^{\mathrm{EV}}_{i,t}/\bar{E}^{\mathrm{EV}}_i`$;
EV urgency $`[0.3\bar{E}^{\mathrm{EV}}_i+N_i-E^{\mathrm{EV}}_{i,t}]_+/(\bar{P}^{\mathrm{EV}}_i\max\{1,24-t\}+10^{-6})`$.

### Parameters

$`\eta=0.95`$ (battery and EV). Prices: tariff $`\pi_t`$, gas 0.08 $/kWh, battery
cycling 0.02 $/kWh, flexible-load revenue 0.20 $/kWh.

| $`i`$ | $`\bar{L}`$ (kW) | $`\bar{P}^{PV}`$ (kW) | $`\bar{E}`$ (kWh) | $`\bar{P}^{\mathrm{gas}}`$ (kW) | $`\bar{M}_i`$ (kg) | $`\bar{F}`$ (kW) | $`\bar{E}^{\mathrm{EV}}`$ (kWh) | $`N_i`$ (kWh) | $`\bar{P}^{\mathrm{EV}}`$ (kW) | $`\sigma_i`$ (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 120 | 50 | 200 | 70 | 882 | 60 | 60 | 25 | 7 | 0 |
| 2 | 150 | 30 | 100 | 90 | 1136 | 80 | 50 | 20 | 11 | −2 |
| 3 | 90 | 80 | 350 | 50 | 597 | 40 | 70 | 30 | 7 | −4 |
| 4 | 180 | 15 | 30 | 110 | 1439 | 90 | 40 | 15 | 7 | 2 |
| 5 | 110 | 65 | 280 | 65 | 877 | 55 | 65 | 28 | 11 | −3 |
| 6 | 140 | 40 | 150 | 85 | 1028 | 70 | 55 | 22 | 7 | −1 |
| 7 | 80 | 100 | 400 | 45 | 465 | 35 | 80 | 35 | 11 | −6 |
| 8 | 170 | 20 | 50 | 100 | 1312 | 85 | 45 | 18 | 7 | 1 |
| 9 | 85 | 75 | 250 | 50 | 628 | 45 | 75 | 32 | 11 | −5 |
| 10 | 160 | 55 | 120 | 95 | 1272 | 75 | 50 | 20 | 7 | 3 |
| 11 | 100 | 90 | 320 | 55 | 732 | 50 | 70 | 30 | 11 | −7 |
| 12 | 200 | 10 | 35 | 120 | 1737 | 100 | 40 | 15 | 7 | 4 |

Hourly profiles ($`t=0..23`$):

| | |
|---|---|
| tariff $`\pi_t`$ ($/kWh) | 0.03 0.03 0.02 0.02 0.02 0.03 · 0.06 0.12 0.20 0.28 0.32 0.35 · 0.35 0.35 0.32 0.28 0.20 0.12 · 0.16 0.24 0.30 0.26 0.14 0.06 |
| carbon intensity $`\kappa_t`$ (kg/kWh) | 0.90 0.90 0.85 0.85 0.85 0.90 · 0.80 0.60 0.35 0.15 0.08 0.05 · 0.05 0.05 0.08 0.20 0.45 0.65 · 0.80 0.90 0.92 0.88 0.85 0.90 |
| flexible-load availability $`\phi_t`$ | 0 0 0 0 0 0.05 · 0.15 0.40 0.70 0.90 1.00 1.00 · 0.95 0.90 0.85 0.80 0.60 0.30 · 0.10 0.05 0 0 0 0 |

### Operation problem

$`\min\sum_t\sum_i f_{i,t}`$ subject to the energy balance, storage and EV dynamics,
$`E^{\mathrm{EV}}_{i,24}\ge0.3\bar{E}^{\mathrm{EV}}_i+N_i`$, $`M_{i,23}\le\bar{M}_i`$ (local) and $`M_{23}\le\bar{M}`$ (global).

## Shared profiles

| $`t`$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $`\bar{\ell}_t`$ | .45 | .40 | .38 | .36 | .36 | .38 | .50 | .70 | .85 | .90 | .90 | .88 | .87 | .87 | .85 | .85 | .88 | .95 | 1.00 | .97 | .90 | .80 | .65 | .52 |
| $`\bar{s}^{\mathrm{PV}}_t`$ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | .24 | .46 | .66 | .82 | .94 | .99 | .99 | .94 | .82 | .66 | .46 | .24 | 0 | 0 | 0 | 0 | 0 |
