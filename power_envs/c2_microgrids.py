"""Case C2: five microgrids in a distribution network (voltage coordination).

Agents: five microgrids (7-bus networks) connected to buses 8, 17, 24, 25 and 32 of the
IEEE 33-bus distribution network; all power flows are AC (pandapower, Newton-Raphson).
  Local constraint  (per MG): MG bus voltages within [0.95, 1.05] p.u.
  Global constraint (shared): distribution-network bus voltages within [0.95, 1.05] p.u.
Action per agent: [P_dg, Q_dg, P_bess, load_curtail] in [0, 1]^4.
Episode: 24 hourly steps (one day). See docs/environments.md for the full CMDP.
"""
import numpy as np
import pandapower as pp
import pandapower.networks as nw

from power_envs.profiles import get_load_profile, get_pv_profile


def _create_mg_network(vn_kv=12.66):
    """7-bus microgrid: bus 0 (PCC), branches 1-2-3 and 4-5-6.

    Impedances in p.u. on a 1 MVA base, converted to ohms (x 160.2756).
    """
    net = pp.create_empty_network()
    for i in range(7):
        pp.create_bus(net, vn_kv=vn_kv, name=f"MG_Bus_{i}")
    # Y-fork topology: (0,1),(1,2),(2,3),(0,4),(4,5),(5,6)
    Z_base = vn_kv**2 / 1.0  # 1 MVA base → 160.2756 ohm
    r_pu = {(0,1): 0.01, (1,2): 0.015, (2,3): 0.02,
            (0,4): 0.012, (4,5): 0.015, (5,6): 0.02}
    x_pu = {(0,1): 0.005, (1,2): 0.007, (2,3): 0.01,
            (0,4): 0.006, (4,5): 0.008, (5,6): 0.01}
    for (f, t), r in r_pu.items():
        pp.create_line_from_parameters(net, f, t, length_km=1.0,
                                       r_ohm_per_km=r * Z_base,
                                       x_ohm_per_km=x_pu[(f,t)] * Z_base,
                                       c_nf_per_km=0, max_i_ka=1.0)
    # Loads on buses 1-6 (MW, Mvar)
    mg_loads = [(1, 0.20, 0.05), (2, 0.30, 0.10), (3, 0.25, 0.05),
                (4, 0.20, 0.05), (5, 0.30, 0.10), (6, 0.25, 0.05)]
    for bus, p, q in mg_loads:
        pp.create_load(net, bus, p_mw=p, q_mvar=q)
    return net


class MGVoltageEnv:
    """Distribution network + 5 MGs. Observation per agent (dim 8):
    [t/24, load factor, PV factor, V_tie, |P_tie|/5, voltage-violation signal, V_tie - 1, SOC].
    """

    def __init__(self):
        # ---------- Profiles ----------
        self.base_load_profile = np.array(get_load_profile())  # DNO load profile
        self.base_pv_profile = np.array(get_pv_profile())
        self.current_load_profile = self.base_load_profile.copy()
        self.current_pv_profile = self.base_pv_profile.copy()
        # Load-profile shift of each MG (h)
        self.mg_load_shift = {8: 0, 17: -2, 24: 4, 25: -3, 32: 2}

        # ---------- TOU grid price ($/MWh) ----------
        self.grid_tou = np.array([
            60, 60, 55, 55, 60, 80,           # 0-5: night/early morning
            120, 160, 200, 200, 180, 160,      # 6-11: morning ramp & peak
            140, 140, 160, 180, 200, 200,      # 12-17: afternoon peak
            180, 160, 120, 100, 80, 60,        # 18-23: evening decline
        ], dtype=np.float64)

        # ---------- DNO network (IEEE 33-bus) ----------
        self.dno_net = nw.case33bw()
        self.dno_net.ext_grid['vm_pu'] = 1.05
        self.dno_base_p = self.dno_net.load['p_mw'].values.copy()
        self.dno_base_q = self.dno_net.load['q_mvar'].values.copy()
        self.dno_load_buses = self.dno_net.load['bus'].values.copy()

        # ---------- MG configs (DG + BESS + curtailment) ----------
        self.mg_ids = [8, 17, 24, 25, 32]
        self.n_agents = 5
        self.mg_configs = {
            8:  {'P_max_mw': 1.5, 'Q_max_mvar': 2.1, 'Q_min_mvar': -2.1,
                 'cost': 85, 'is_pv': False,
                 'E_mwh': 0.5,  'P_bess_max': 0.25, 'curtail_max': 0.20},
            17: {'P_max_mw': 2.0, 'Q_max_mvar': 2.8, 'Q_min_mvar': -2.8,
                 'cost': 90, 'is_pv': False,
                 'E_mwh': 1.0,  'P_bess_max': 0.5,  'curtail_max': 0.15},
            24: {'P_max_mw': 2.0, 'Q_max_mvar': 2.8, 'Q_min_mvar': -2.8,
                 'cost': 80, 'is_pv': False,
                 'E_mwh': 0.8,  'P_bess_max': 0.4,  'curtail_max': 0.25},
            25: {'P_max_mw': 1.0, 'Q_max_mvar': 1.4, 'Q_min_mvar': -1.4,
                 'cost': 25, 'is_pv': True,
                 'E_mwh': 1.2,  'P_bess_max': 0.6,  'curtail_max': 0.10},
            32: {'P_max_mw': 2.0, 'Q_max_mvar': 2.8, 'Q_min_mvar': -2.8,
                 'cost': 30, 'is_pv': True,
                 'E_mwh': 0.6,  'P_bess_max': 0.3,  'curtail_max': 0.30},
        }
        self.curtail_cost = 500.0   # $/MWh curtailment cost
        self.dr_revenue = 50.0      # $/MWh demand-response revenue
        self.bess_degradation = 15.0  # $/MWh BESS cycling cost
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.eta_ch = 0.95
        self.eta_dis = 0.95

        # ---------- Create MG pandapower networks ----------
        self.mg_nets = {}
        self.mg_base_p = {}
        self.mg_base_q = {}
        self.mg_dg_idx = {}    # index of DG sgen in each MG net
        self.mg_bess_idx = {}  # index of BESS sgen in each MG net
        self.mg_load_idx = {}  # load at tie bus representing MG in DNO net
        for mg_id in self.mg_ids:
            net_mg = _create_mg_network()
            pp.create_ext_grid(net_mg, bus=0, vm_pu=1.05)
            # DG at bus 6
            dg_idx = pp.create_sgen(net_mg, bus=6, p_mw=0.0, q_mvar=0.0)
            self.mg_dg_idx[mg_id] = dg_idx
            # BESS at bus 3 (mid-feeder)
            bess_idx = pp.create_sgen(net_mg, bus=3, p_mw=0.0, q_mvar=0.0)
            self.mg_bess_idx[mg_id] = bess_idx
            self.mg_nets[mg_id] = net_mg
            self.mg_base_p[mg_id] = net_mg.load['p_mw'].values.copy()
            self.mg_base_q[mg_id] = net_mg.load['q_mvar'].values.copy()
            load_idx = pp.create_load(self.dno_net, mg_id, p_mw=0.0, q_mvar=0.0)
            self.mg_load_idx[mg_id] = load_idx

        # ---------- Params ----------
        self.T = 24
        self.V_min = 0.95
        self.V_max = 1.05

        # ---------- Spaces ----------
        self.obs_dim = 8
        self.act_dim = 4

        # ---------- State ----------
        self.t = 0
        self.V_tie = {mg: 1.05 for mg in self.mg_ids}
        self.P_tie = {mg: 0.0 for mg in self.mg_ids}
        self.lambda_P = {mg: 0.0 for mg in self.mg_ids}
        self.lambda_V = {mg: 0.0 for mg in self.mg_ids}
        self.soc = {mg: 0.5 for mg in self.mg_ids}

    def reset(self):
        self.t = 0
        self.V_tie = {mg: 1.05 for mg in self.mg_ids}
        self.P_tie = {mg: 0.0 for mg in self.mg_ids}
        self.lambda_P = {mg: 0.0 for mg in self.mg_ids}
        self.lambda_V = {mg: 0.0 for mg in self.mg_ids}
        self.soc = {mg: 0.5 for mg in self.mg_ids}
        
        # 5% multiplicative noise on the PV and load profiles
        noise_p = np.random.normal(1.0, 0.05, self.T)
        noise_l = np.random.normal(1.0, 0.05, self.T)
        self.current_pv_profile = np.clip(self.base_pv_profile * noise_p, 0.0, None)
        self.current_load_profile = np.clip(self.base_load_profile * noise_l, 0.0, None)
        # Per-MG shifted load profiles
        self.mg_load_profiles = {}
        for mg_id in self.mg_ids:
            shift = self.mg_load_shift[mg_id]
            shifted = np.roll(self.current_load_profile, shift)
            self.mg_load_profiles[mg_id] = shifted

        return self._get_obs()

    def step(self, actions):
        """
        Args:
            actions: dict {mg_id: np.array([P_dg, Q_dg, P_bess, curtail])} ∈ [0,1]^4
        Returns:
            obs, rewards, done, info
        """
        lf = self.current_load_profile[self.t]  # DNO load factor
        pf = self.current_pv_profile[self.t]

        # ---- 1. Decode agent actions & run MG power flows ----
        P_tie_all = {}
        Q_tie_all = {}
        mg_voltages = {}
        local_violations = {}
        dg_costs = {}
        curtail_costs = {}
        bess_costs = {}

        for mg_id in self.mg_ids:
            cfg = self.mg_configs[mg_id]
            a = np.clip(actions[mg_id], 0, 1)

            # a[0]: P_dg
            P_max_t = cfg['P_max_mw'] * pf if cfg['is_pv'] else cfg['P_max_mw']
            P_dg = a[0] * P_max_t
            # a[1]: Q_dg
            Q_dg = cfg['Q_min_mvar'] + a[1] * (cfg['Q_max_mvar'] - cfg['Q_min_mvar'])
            # a[2]: P_bess  (0=max discharge, 0.5=idle, 1=max charge)
            P_bess_raw = (a[2] - 0.5) * 2.0 * cfg['P_bess_max']  # MW, positive=charge
            # Clip by SOC bounds (with efficiency)
            soc_room_charge = (self.soc_max - self.soc[mg_id]) * cfg['E_mwh'] / self.eta_ch
            soc_room_discharge = (self.soc[mg_id] - self.soc_min) * cfg['E_mwh'] * self.eta_dis
            P_bess = np.clip(P_bess_raw, -soc_room_discharge, soc_room_charge)
            # a[3]: load curtailment fraction
            curtail_frac = a[3] * cfg['curtail_max']

            # Update MG network
            mg_lf = self.mg_load_profiles[mg_id][self.t]
            net_mg = self.mg_nets[mg_id]
            net_mg.load['p_mw'] = self.mg_base_p[mg_id] * mg_lf * (1.0 - curtail_frac)
            net_mg.load['q_mvar'] = self.mg_base_q[mg_id] * mg_lf * (1.0 - curtail_frac)
            net_mg.sgen.at[self.mg_dg_idx[mg_id], 'p_mw'] = P_dg
            net_mg.sgen.at[self.mg_dg_idx[mg_id], 'q_mvar'] = Q_dg
            # BESS: positive P_bess=charge=consuming power → sgen p_mw is negative
            net_mg.sgen.at[self.mg_bess_idx[mg_id], 'p_mw'] = -P_bess
            net_mg.ext_grid['vm_pu'] = self.V_tie[mg_id]

            # Run MG power flow
            try:
                pp.runpp(net_mg, numba=False)
            except Exception:
                net_mg.res_bus['vm_pu'] = 1.0
                net_mg.res_ext_grid['p_mw'] = 0.0
                net_mg.res_ext_grid['q_mvar'] = 0.0
                mg_voltages[mg_id] = np.ones(len(net_mg.bus))
                local_violations[mg_id] = 1.0
                P_tie_all[mg_id] = 0.0
                Q_tie_all[mg_id] = 0.0
                dg_costs[mg_id] = cfg['cost'] * P_dg
                curtail_costs[mg_id] = (self.curtail_cost - self.dr_revenue) * curtail_frac * sum(self.mg_base_p[mg_id]) * mg_lf
                bess_costs[mg_id] = self.bess_degradation * abs(P_bess)
                if P_bess >= 0:
                    self.soc[mg_id] += P_bess * self.eta_ch / cfg['E_mwh']
                else:
                    self.soc[mg_id] += P_bess / (self.eta_dis * cfg['E_mwh'])
                self.soc[mg_id] = np.clip(self.soc[mg_id], self.soc_min, self.soc_max)
                continue

            # Extract MG results
            mg_v = net_mg.res_bus['vm_pu'].values
            mg_voltages[mg_id] = mg_v.copy()
            viol = (np.sum(np.maximum(self.V_min - mg_v, 0) ** 2) +
                    np.sum(np.maximum(mg_v - self.V_max, 0) ** 2))
            local_violations[mg_id] = viol

            P_tie_all[mg_id] = float(net_mg.res_ext_grid['p_mw'].values[0])
            Q_tie_all[mg_id] = float(net_mg.res_ext_grid['q_mvar'].values[0])
            dg_costs[mg_id] = cfg['cost'] * P_dg
            curtail_costs[mg_id] = (self.curtail_cost - self.dr_revenue) * curtail_frac * sum(self.mg_base_p[mg_id]) * mg_lf
            bess_costs[mg_id] = self.bess_degradation * abs(P_bess)

            # Update SOC with efficiency
            if P_bess >= 0:
                self.soc[mg_id] += P_bess * self.eta_ch / cfg['E_mwh']
            else:
                self.soc[mg_id] += P_bess / (self.eta_dis * cfg['E_mwh'])
            self.soc[mg_id] = np.clip(self.soc[mg_id], self.soc_min, self.soc_max)

        # ---- 2. DNO power flow ----
        # Scale DNO base loads
        self.dno_net.load.iloc[:len(self.dno_base_p), self.dno_net.load.columns.get_loc('p_mw')] = self.dno_base_p * lf
        self.dno_net.load.iloc[:len(self.dno_base_q), self.dno_net.load.columns.get_loc('q_mvar')] = self.dno_base_q * lf

        # Set MG tie-power as loads at MG buses in DNO
        for mg_id in self.mg_ids:
            idx = self.mg_load_idx[mg_id]
            self.dno_net.load.at[idx, 'p_mw'] = P_tie_all[mg_id]
            self.dno_net.load.at[idx, 'q_mvar'] = Q_tie_all[mg_id]

        try:
            pp.runpp(self.dno_net, numba=False)
        except Exception:
            self.dno_net.res_bus['vm_pu'] = 1.0
            global_v_violation = 1.0
        else:
            # ---- 3. Global constraint: DNO voltages ----
            dno_v = self.dno_net.res_bus['vm_pu'].values
            global_v_violation = (np.sum(np.maximum(self.V_min - dno_v, 0) ** 2) +
                                  np.sum(np.maximum(dno_v - self.V_max, 0) ** 2))

        # ---- 4. Rewards (savings vs passive grid import) ----
        t = self.t - 1 if self.t > 0 else 0  # hour of the tariff and the baseline
        grid_price_t = self.grid_tou[min(t, 23)]
        norm_factor_cost = 0.008
        norm_factor_viol = 100.0

        # Baseline: cost if all demand served purely from grid (no DG/BESS/curtail)
        baseline_cost = 0.0
        for mg_id in self.mg_ids:
            mg_lf = self.mg_load_profiles[mg_id][min(t, 23)]
            demand_mw = sum(self.mg_base_p[mg_id]) * mg_lf
            baseline_cost += grid_price_t * demand_mw

        # Actual system cost with agent dispatch
        total_system_cost = 0.0
        for mg_id in self.mg_ids:
            grid_cost = grid_price_t * max(0.0, P_tie_all[mg_id])  # only pay for import
            agent_cost = dg_costs[mg_id] + grid_cost + curtail_costs[mg_id] + bess_costs[mg_id]
            total_system_cost += agent_cost

        # Savings: positive when agents reduce cost below passive import
        savings = baseline_cost - total_system_cost
        shared_reward = savings * norm_factor_cost
        rewards = {}
        for mg_id in self.mg_ids:
            rewards[mg_id] = float(shared_reward)
            local_violations[mg_id] = local_violations[mg_id] * norm_factor_viol

        global_v_violation = global_v_violation * norm_factor_viol

        # ---- 5. Update signals ----
        dno_v = self.dno_net.res_bus['vm_pu'].values
        for mg_id in self.mg_ids:
            self.V_tie[mg_id] = float(dno_v[mg_id])
            self.P_tie[mg_id] = float(P_tie_all[mg_id])
            self.lambda_P[mg_id] += 0.1 * (global_v_violation / self.n_agents)
            self.lambda_V[mg_id] = float(self.V_tie[mg_id] - 1.0)

        # ---- 6. Advance ----
        self.t += 1
        done = self.t >= self.T

        info = {
            'global_voltage_violation': float(global_v_violation),
            'local_voltage_violations': {mg: float(local_violations[mg]) for mg in self.mg_ids},
            'dno_V_min': float(np.min(dno_v)),
            'dno_V_max': float(np.max(dno_v)),
            'mg_voltages': {mg: mg_voltages[mg].tolist() for mg in self.mg_ids},
            'P_tie': {mg: float(P_tie_all[mg]) for mg in self.mg_ids},
        }

        return self._get_obs(), rewards, done, info

    def _get_obs(self):
        t = self.t
        obs = {}
        for mg_id in self.mg_ids:
            obs[mg_id] = np.array([
                t / self.T,
                self.current_load_profile[min(t, self.T - 1)] / 2.0,
                self.current_pv_profile[min(t, self.T - 1)],
                (self.V_tie[mg_id] - 0.9) / 0.2,
                abs(self.P_tie[mg_id]) / 5.0,
                np.clip(self.lambda_P[mg_id] / 50.0, -1, 1),
                np.clip(self.lambda_V[mg_id] / 50.0, -1, 1),
                self.soc[mg_id],  # SOC ∈ [0.1, 0.9]
            ], dtype=np.float32)
        return obs

    @property
    def agent_ids(self):
        return self.mg_ids
