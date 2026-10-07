"""Case C1: TSO-DSO coordination (three DSOs under one transmission system).

Agents: three DSOs, each an IEEE 33-bus feeder solved by AC power flow (pandapower),
attached to buses 7, 27 and 29 of the IEEE 30-bus transmission system (DC power flow).
  Local constraint  (per DSO): bus voltages within [0.95, 1.05] p.u.
  Global constraint (shared):  total TSO-to-DSO active-power import <= 8 MW.
Action per agent: [P_dg, Q_dg, P_bess, curtail] in [0, 1]^4.
Episode: 24 hourly steps (one day). See docs/environments.md for the full CMDP.
"""
import numpy as np
import pandapower as pp
import pandapower.networks as nw

from power_envs.profiles import get_load_profile, get_pv_profile


class TSODSOEnv:
    """TSO + 3 DSOs. Observation per agent (dim 8):
    [t/T, load factor, PV factor, P_tie/10, import-cap signal, total import/20, SOC, curtailment].
    """

    def __init__(self):
        # ---------- Profiles ----------
        self.base_load_profile = np.array(get_load_profile())
        self.base_pv_profile = np.array(get_pv_profile())
        self.current_load_profile = self.base_load_profile.copy()
        self.current_pv_profile = self.base_pv_profile.copy()

        # ---------- TSO network (IEEE 30-bus) ----------
        self.tso_net = nw.case30()
        self.tso_base_p = self.tso_net.load['p_mw'].values.copy()
        self.tso_base_q = self.tso_net.load['q_mvar'].values.copy()

        # ---------- DSO configs ----------
        self.dso_locations = [7, 27, 29]
        self.n_agents = 3
        self.dso_configs = {
            7:  {'load_scale': 1.2, 'P_max': 3.5, 'Q_max': 2.5, 'cost': 80,
                 'E_bess': 5.0, 'P_bess_max': 2.5, 'curtail_max': 0.15, 'curtail_cost': 400},
            27: {'load_scale': 1.5, 'P_max': 4.0, 'Q_max': 3.0, 'cost': 90,
                 'E_bess': 8.0, 'P_bess_max': 4.0, 'curtail_max': 0.20, 'curtail_cost': 450},
            29: {'load_scale': 2.0, 'P_max': 5.0, 'Q_max': 3.5, 'cost': 85,
                 'E_bess': 6.0, 'P_bess_max': 3.0, 'curtail_max': 0.25, 'curtail_cost': 350},
        }
        self.dg_bus = 24
        self.bess_bus = 12
        self.gen_cost = 20.0  # $/MWh, TSO generation cost (info only)
        self.bess_degradation = 12.0  # $/MWh BESS cycling cost
        self.eta_ch = 0.95
        self.eta_dis = 0.95

        # TOU import price for DSO grid purchase ($/MWh)
        self.grid_tou = np.array([
            50, 50, 45, 45, 50, 70,           # 0-5: night
            110, 150, 180, 180, 160, 140,      # 6-11: morning peak
            130, 130, 150, 170, 190, 190,      # 12-17: afternoon peak
            170, 150, 110, 90, 70, 50,         # 18-23: evening
        ], dtype=np.float64)

        # ---------- Create DSO pandapower networks ----------
        self.dso_nets = {}
        self.dso_base_p = {}
        self.dso_base_q = {}
        self.dso_dg_idx = {}
        self.dso_bess_idx = {}
        self.tso_dso_load_idx = {}

        for loc in self.dso_locations:
            net_dso = nw.case33bw()
            net_dso.ext_grid['vm_pu'] = 1.05
            scale = self.dso_configs[loc]['load_scale']
            self.dso_base_p[loc] = net_dso.load['p_mw'].values.copy() * scale
            self.dso_base_q[loc] = net_dso.load['q_mvar'].values.copy() * scale

            # DG at bus 24
            dg_idx = pp.create_sgen(net_dso, bus=self.dg_bus, p_mw=0.0, q_mvar=0.0)
            self.dso_dg_idx[loc] = dg_idx

            # BESS at bus 12 (modelled as sgen with net injection)
            bess_idx = pp.create_sgen(net_dso, bus=self.bess_bus, p_mw=0.0, q_mvar=0.0)
            self.dso_bess_idx[loc] = bess_idx

            self.dso_nets[loc] = net_dso

            # Add tie-power load to TSO
            load_idx = pp.create_load(self.tso_net, loc, p_mw=0.0, q_mvar=0.0)
            self.tso_dso_load_idx[loc] = load_idx

        # Compute total base load per DSO for curtailment cost
        self.dso_total_base_load = {}
        for loc in self.dso_locations:
            self.dso_total_base_load[loc] = float(np.sum(self.dso_base_p[loc]))

        # ---------- Params ----------
        self.T = 24
        self.V_min = 0.95
        self.V_max = 1.05
        self.P_cap_mw = 8.0

        # ---------- Spaces ----------
        self.obs_dim = 8
        self.act_dim = 4

        # ---------- State ----------
        self.t = 0
        self.P_tie = {loc: 0.0 for loc in self.dso_locations}
        self.lambda_price = {loc: 0.0 for loc in self.dso_locations}
        self.total_P_from_tso = 0.0
        self.soc = {loc: cfg['E_bess'] * 0.5 for loc, cfg in self.dso_configs.items()}
        self.curtail_prev = {loc: 0.0 for loc in self.dso_locations}

    def reset(self):
        self.t = 0
        self.P_tie = {loc: 0.0 for loc in self.dso_locations}
        self.lambda_price = {loc: 0.0 for loc in self.dso_locations}
        self.total_P_from_tso = 0.0
        self.soc = {loc: cfg['E_bess'] * 0.5 for loc, cfg in self.dso_configs.items()}
        self.curtail_prev = {loc: 0.0 for loc in self.dso_locations}

        noise_p = np.random.normal(1.0, 0.05, self.T)
        noise_l = np.random.normal(1.0, 0.05, self.T)
        self.current_pv_profile = np.clip(self.base_pv_profile * noise_p, 0.0, None)
        self.current_load_profile = np.clip(self.base_load_profile * noise_l, 0.0, None)

        return self._get_obs()

    def step(self, actions):
        t = self.t
        lf = self.current_load_profile[t]

        P_tie_all = {}
        dso_violations = {}
        dso_voltages = {}
        dso_costs = {}

        for loc in self.dso_locations:
            a = np.clip(actions[loc], 0, 1)
            cfg = self.dso_configs[loc]
            net_dso = self.dso_nets[loc]
            E_max = cfg['E_bess']

            # ---- Decode actions ----
            # a[0]: P_dg
            p_dg = a[0] * cfg['P_max']
            # a[1]: Q_dg
            q_dg = (a[1] - 0.5) * 2.0 * cfg['Q_max']
            # a[2]: P_bess (0=full discharge, 0.5=idle, 1=full charge)
            bess_raw = (a[2] - 0.5) * 2.0
            soc_kwh = self.soc[loc]
            if bess_raw >= 0:
                charge_room = max(0.0, E_max - soc_kwh)
                charge_cap = min(cfg['P_bess_max'], charge_room / self.eta_ch)
                P_ch = bess_raw * charge_cap
                P_dis = 0.0
            else:
                discharge_room = max(0.0, soc_kwh - E_max * 0.1)
                discharge_cap = min(cfg['P_bess_max'], discharge_room * self.eta_dis)
                P_ch = 0.0
                P_dis = -bess_raw * discharge_cap
            p_bess_net = P_dis - P_ch  # positive = injection into network

            # a[3]: curtailment fraction
            curtail_frac = a[3] * cfg['curtail_max']

            # ---- Apply to pandapower ----
            # Scale loads with curtailment
            net_dso.load.iloc[:len(self.dso_base_p[loc]),
                              net_dso.load.columns.get_loc('p_mw')] = \
                self.dso_base_p[loc] * lf * (1 - curtail_frac)
            net_dso.load.iloc[:len(self.dso_base_q[loc]),
                              net_dso.load.columns.get_loc('q_mvar')] = \
                self.dso_base_q[loc] * lf * (1 - curtail_frac)

            # DG at bus 24
            net_dso.sgen.at[self.dso_dg_idx[loc], 'p_mw'] = p_dg
            net_dso.sgen.at[self.dso_dg_idx[loc], 'q_mvar'] = q_dg
            # BESS at bus 12
            net_dso.sgen.at[self.dso_bess_idx[loc], 'p_mw'] = p_bess_net
            net_dso.sgen.at[self.dso_bess_idx[loc], 'q_mvar'] = 0.0

            # ---- Device costs ----
            dg_cost = cfg['cost'] * p_dg
            curtail_cost = cfg['curtail_cost'] * curtail_frac * self.dso_total_base_load[loc] * lf
            bess_cost = self.bess_degradation * (P_ch + P_dis)  # cycling cost
            dso_costs[loc] = dg_cost + curtail_cost + bess_cost

            # ---- Run AC PF ----
            try:
                pp.runpp(net_dso, numba=False)
            except Exception:
                net_dso.res_bus['vm_pu'] = 1.0
                net_dso.res_ext_grid['p_mw'] = 0.0
                dso_violations[loc] = 1.0
                dso_voltages[loc] = np.ones(len(net_dso.bus))
                P_tie_all[loc] = 0.0
                # SOC unchanged on PF failure
                self.curtail_prev[loc] = curtail_frac
                continue

            voltages = net_dso.res_bus['vm_pu'].values
            dso_voltages[loc] = voltages.copy()
            viol = (np.sum(np.maximum(self.V_min - voltages, 0) ** 2) +
                    np.sum(np.maximum(voltages - self.V_max, 0) ** 2))
            dso_violations[loc] = viol

            P_tie_all[loc] = float(net_dso.res_ext_grid['p_mw'].values[0])

            # ---- SOC update ----
            self.soc[loc] += P_ch * self.eta_ch - P_dis / self.eta_dis
            self.soc[loc] = np.clip(self.soc[loc], E_max * 0.1, E_max)
            self.curtail_prev[loc] = curtail_frac

        # ---- TSO DC power flow ----
        n_base = len(self.tso_base_p)
        self.tso_net.load.iloc[:n_base,
                               self.tso_net.load.columns.get_loc('p_mw')] = self.tso_base_p * lf
        self.tso_net.load.iloc[:n_base,
                               self.tso_net.load.columns.get_loc('q_mvar')] = self.tso_base_q * lf
        for loc in self.dso_locations:
            idx = self.tso_dso_load_idx[loc]
            self.tso_net.load.at[idx, 'p_mw'] = P_tie_all[loc]
        pp.rundcpp(self.tso_net, numba=False)

        # ---- Global constraint: P import cap ----
        total_P = sum(P_tie_all.values())
        P_over = max(0.0, total_P - self.P_cap_mw)
        overload_raw = P_over ** 2

        # TSO generation cost
        tso_gen_cost = float(np.sum(self.tso_net.res_gen['p_mw'].values) * self.gen_cost +
                             np.sum(self.tso_net.res_ext_grid['p_mw'].values) * self.gen_cost)

        # ---- Rewards (savings vs passive grid import) ----
        grid_price_t = self.grid_tou[min(t, 23)]
        norm_factor_cost = 0.003
        norm_factor_viol_local = 100.0

        # Baseline: cost if all DSO demand served from TSO grid (no DG/BESS/curtail)
        baseline_cost = 0.0
        for loc in self.dso_locations:
            demand_mw = self.dso_total_base_load[loc] * lf
            baseline_cost += grid_price_t * demand_mw

        # Actual system cost with agent dispatch
        total_system_cost = 0.0
        for loc in self.dso_locations:
            grid_import_cost = grid_price_t * max(0.0, P_tie_all[loc])
            total_system_cost += dso_costs[loc] + grid_import_cost

        # Savings: positive when agents reduce cost below passive import
        savings = baseline_cost - total_system_cost
        shared_reward = savings * norm_factor_cost
        rewards = {}
        for loc in self.dso_locations:
            rewards[loc] = float(shared_reward)
            dso_violations[loc] = dso_violations[loc] * norm_factor_viol_local

        norm_factor_viol_global = 1.0
        overload = float(overload_raw) * norm_factor_viol_global

        # ---- Update signals ----
        for loc in self.dso_locations:
            self.P_tie[loc] = float(P_tie_all[loc])
            self.lambda_price[loc] += 0.1 * overload / self.n_agents
        self.total_P_from_tso = float(total_P)

        # ---- Advance ----
        self.t += 1
        done = self.t >= self.T

        info = {
            'tso_gen_cost': float(tso_gen_cost),
            'total_P_from_tso': float(total_P),
            'line_overload': float(overload),
            'dso_voltage_violations': {loc: float(dso_violations[loc]) for loc in self.dso_locations},
            'dso_V_min': {loc: float(np.min(dso_voltages[loc])) for loc in self.dso_locations},
            'P_tie': {loc: float(P_tie_all[loc]) for loc in self.dso_locations},
            'soc': {loc: float(self.soc[loc]) for loc in self.dso_locations},
        }

        return self._get_obs(), rewards, done, info

    def _get_obs(self):
        t = self.t
        obs = {}
        for loc in self.dso_locations:
            cfg = self.dso_configs[loc]
            E_max = max(cfg['E_bess'], 1.0)
            obs[loc] = np.array([
                t / self.T,
                self.current_load_profile[min(t, self.T - 1)] / 2.0,
                self.current_pv_profile[min(t, self.T - 1)],
                self.P_tie[loc] / 10.0,
                np.clip(self.lambda_price[loc] / 50.0, -1, 1),
                np.clip(self.total_P_from_tso / 20.0, -1, 1),
                self.soc[loc] / E_max,
                self.curtail_prev[loc],
            ], dtype=np.float32)
        return obs

    @property
    def agent_ids(self):
        return self.dso_locations
