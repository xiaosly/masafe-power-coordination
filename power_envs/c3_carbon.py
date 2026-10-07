"""Case C3: 12-prosumer energy community with daily carbon caps.

Global constraint (shared): cumulative daily carbon emission of the community <= cap.
Local constraint (per prosumer): cumulative daily carbon emission of the prosumer <= its cap.
Agents: 12 prosumers with PV, battery, gas DG, grid import, flexible load and an EV.

Action per agent: [P_gas, battery, P_grid, flexible load, EV charging] in [0, 1]^5.
Episode: 24 hourly steps (one day). See docs/environments.md for the full CMDP.
"""
import numpy as np

from power_envs.profiles import get_load_profile, get_pv_profile, get_carbon_intensity_profile


def get_flex_load_profile():
    """Hourly availability of the flexible load."""
    return np.array([
        0.0,  0.0,  0.0,  0.0,  0.0,  0.05,
        0.15, 0.40, 0.70, 0.90, 1.00, 1.00,
        0.95, 0.90, 0.85, 0.80, 0.60, 0.30,
        0.10, 0.05, 0.0,  0.0,  0.0,  0.0
    ])


class CarbonMarketEnv:
    """Community of 12 prosumers. Observation per agent (dim 11):
    [t/T, load, PV, SOC, own emissions / cap, carbon intensity, remaining community budget,
    carbon-price signal, flexible-load availability, EV SOC, EV urgency].
    """

    def __init__(self):
        self.base_load_profile = np.array(get_load_profile())
        self.base_pv_profile = np.array(get_pv_profile())
        self.current_pv_profile = self.base_pv_profile.copy()
        self.carbon_intensity = np.array(get_carbon_intensity_profile())
        self.flex_profile = get_flex_load_profile()

        self.prosumer_ids = list(range(12))
        self.n_agents = 12
        self.gas_cost = 0.08  # $/kWh
        self.gas_ci = 0.50
        self.flex_price = 0.20  # $/kWh revenue of the flexible load
        self.curt_cost = 0.01
        self.bess_degradation = 0.02  # $/kWh per cycle

        self.configs = {
            0:  {'load_scale': 120, 'pv_scale': 50,  'bess_cap': 200, 'gas_cap': 70,  'cap_local': 882,
                 'flex_scale': 60,  'ev_cap': 60,  'ev_daily_need': 25, 'ev_charge_max': 7},
            1:  {'load_scale': 150, 'pv_scale': 30,  'bess_cap': 100, 'gas_cap': 90,  'cap_local': 1136,
                 'flex_scale': 80,  'ev_cap': 50,  'ev_daily_need': 20, 'ev_charge_max': 11},
            2:  {'load_scale': 90,  'pv_scale': 80,  'bess_cap': 350, 'gas_cap': 50,  'cap_local': 597,
                 'flex_scale': 40,  'ev_cap': 70,  'ev_daily_need': 30, 'ev_charge_max': 7},
            3:  {'load_scale': 180, 'pv_scale': 15,  'bess_cap': 30,  'gas_cap': 110, 'cap_local': 1439,
                 'flex_scale': 90,  'ev_cap': 40,  'ev_daily_need': 15, 'ev_charge_max': 7},
            4:  {'load_scale': 110, 'pv_scale': 65,  'bess_cap': 280, 'gas_cap': 65,  'cap_local': 877,
                 'flex_scale': 55,  'ev_cap': 65,  'ev_daily_need': 28, 'ev_charge_max': 11},
            5:  {'load_scale': 140, 'pv_scale': 40,  'bess_cap': 150, 'gas_cap': 85,  'cap_local': 1028,
                 'flex_scale': 70,  'ev_cap': 55,  'ev_daily_need': 22, 'ev_charge_max': 7},
            6:  {'load_scale': 80,  'pv_scale': 100, 'bess_cap': 400, 'gas_cap': 45,  'cap_local': 465,
                 'flex_scale': 35,  'ev_cap': 80,  'ev_daily_need': 35, 'ev_charge_max': 11},
            7:  {'load_scale': 170, 'pv_scale': 20,  'bess_cap': 50,  'gas_cap': 100, 'cap_local': 1312,
                 'flex_scale': 85,  'ev_cap': 45,  'ev_daily_need': 18, 'ev_charge_max': 7},
            8:  {'load_scale': 85,  'pv_scale': 75,  'bess_cap': 250, 'gas_cap': 50,  'cap_local': 628,
                 'flex_scale': 45,  'ev_cap': 75,  'ev_daily_need': 32, 'ev_charge_max': 11},
            9:  {'load_scale': 160, 'pv_scale': 55,  'bess_cap': 120, 'gas_cap': 95,  'cap_local': 1272,
                 'flex_scale': 75,  'ev_cap': 50,  'ev_daily_need': 20, 'ev_charge_max': 7},
            10: {'load_scale': 100, 'pv_scale': 90,  'bess_cap': 320, 'gas_cap': 55,  'cap_local': 732,
                 'flex_scale': 50,  'ev_cap': 70,  'ev_daily_need': 30, 'ev_charge_max': 11},
            11: {'load_scale': 200, 'pv_scale': 10,  'bess_cap': 35,  'gas_cap': 120, 'cap_local': 1737,
                 'flex_scale': 100, 'ev_cap': 40,  'ev_daily_need': 15, 'ev_charge_max': 7},
        }

        self.load_shift = {0:0, 1:-2, 2:-4, 3:2, 4:-3, 5:-1, 6:-6, 7:1, 8:-5, 9:3, 10:-7, 11:4}
        self.tou_price = np.array([
            0.03, 0.03, 0.02, 0.02, 0.02, 0.03, 0.06, 0.12, 0.20, 0.28,
            0.32, 0.35, 0.35, 0.35, 0.32, 0.28, 0.20, 0.12, 0.16, 0.24,
            0.30, 0.26, 0.14, 0.06
        ])
        self.ci_cap_total = 11354.0

        self.prosumer_load_profiles = {
            i: np.roll(self.base_load_profile, self.load_shift[i])
            for i in self.prosumer_ids
        }
        self.eta_ch = 0.95
        self.eta_dis = 0.95
        self.ev_eta = 0.95

        self.T = 24

        self.obs_dim = 11
        self.act_dim = 5

        self.t = 0
        self.soc = {}
        self.ev_soc = {}
        self.cumul_carbon = {}
        self.global_cumul_carbon = 0.0
        self.carbon_price = 0.0

    def reset(self):
        self.t = 0
        self.soc = {i: cfg['bess_cap'] * 0.5 for i, cfg in self.configs.items()}
        self.ev_soc = {i: cfg['ev_cap'] * 0.3 for i, cfg in self.configs.items()}
        self.cumul_carbon = {i: 0.0 for i in self.prosumer_ids}
        self.global_cumul_carbon = 0.0
        self.carbon_price = 0.0

        noise_p = np.random.normal(1.0, 0.05, self.T)
        self.current_pv_profile = np.clip(self.base_pv_profile * noise_p, 0.0, None)
        self.current_load_profiles = {}
        for i in self.prosumer_ids:
            noise_l = np.random.normal(1.0, 0.05, self.T)
            self.current_load_profiles[i] = np.clip(
                self.prosumer_load_profiles[i] * noise_l, 0.0, None)

        return self._get_obs()

    def step(self, actions):
        """
        actions: dict {pid: np.array([P_gas_norm, batt_norm, P_grid_norm, flex_accept, ev_charge_norm])}
        All actions in [0, 1].
        """
        t = self.t
        pf = self.current_pv_profile[t]
        ci = self.carbon_intensity[t]
        flex_t = self.flex_profile[t]

        rewards = {}
        step_carbon = {}
        step_cost = {}

        for i in self.prosumer_ids:
            cfg = self.configs[i]
            a = np.clip(actions[i], 0, 1)
            E_max = cfg['bess_cap']

            load_t = cfg['load_scale'] * self.current_load_profiles[i][t]
            pv_t = cfg['pv_scale'] * pf

            # --- Decode 5 actions ---
            # a[0]: P_gas_norm → [0, gas_cap]
            P_gas = a[0] * cfg['gas_cap']

            # a[1]: batt_norm → 0=full discharge, 0.5=idle, 1=full charge
            if E_max > 0:
                p_chg_max = E_max / 4.0
                p_dis_max = E_max / 4.0
                charge_room = max(0.0, E_max - self.soc[i])
                discharge_room = max(0.0, self.soc[i] - E_max * 0.1)
                charge_cap = min(p_chg_max, charge_room / self.eta_ch)
                discharge_cap = min(p_dis_max, discharge_room * self.eta_dis)
                batt_raw = (a[1] - 0.5) * 2.0
                if batt_raw >= 0:
                    P_ch = batt_raw * charge_cap
                    P_dis = 0
                else:
                    P_ch = 0
                    P_dis = -batt_raw * discharge_cap
            else:
                P_ch = 0; P_dis = 0

            # a[2]: P_grid_norm → [0, max(2 load_t, 1)]
            max_grid = max(load_t * 2, 1.0)
            P_grid = a[2] * max_grid

            # a[3]: flex_accept → fraction of available flex load to serve
            flex_avail = cfg['flex_scale'] * flex_t
            flex_served = a[3] * flex_avail

            # a[4]: ev_charge_norm → [0, ev_charge_max], limited by EV SOC room
            ev_cap_i = cfg['ev_cap']
            ev_room = max(0.0, ev_cap_i - self.ev_soc[i])
            ev_charge_limit = min(cfg['ev_charge_max'], ev_room / self.ev_eta)
            ev_charge = a[4] * ev_charge_limit

            # --- Power balance ---
            # supply = pv + grid + gas + bess_discharge
            # demand = load + bess_charge + flex_served + ev_charge
            supply = pv_t + P_grid + P_gas + P_dis
            demand = load_t + P_ch + flex_served + ev_charge
            deficit = demand - supply

            if deficit > 0:
                P_grid += deficit  # auto-purchase from grid to balance

            # --- SOC updates ---
            if E_max > 0:
                self.soc[i] += P_ch * self.eta_ch - P_dis / self.eta_dis
                self.soc[i] = np.clip(self.soc[i], E_max * 0.1, E_max)

            self.ev_soc[i] += ev_charge * self.ev_eta
            self.ev_soc[i] = min(self.ev_soc[i], ev_cap_i)

            # --- Carbon accounting ---
            carbon_grid = P_grid * ci
            carbon_gas = P_gas * self.gas_ci
            carbon_t = carbon_grid + carbon_gas
            self.cumul_carbon[i] += carbon_t
            step_carbon[i] = carbon_t

            # --- Cost: grid TOU + gas + BESS degradation - flex revenue ---
            bess_cycle_cost = self.bess_degradation * (P_ch + P_dis)
            cost = P_grid * self.tou_price[t] + P_gas * self.gas_cost + bess_cycle_cost - flex_served * self.flex_price
            step_cost[i] = cost

        # --- Global carbon ---
        self.global_cumul_carbon = sum(self.cumul_carbon.values())

        # --- Constraint violations ---
        norm_factor_viol = 5.0
        local_violations = {}
        for i in self.prosumer_ids:
            excess = max(0, self.cumul_carbon[i] - self.configs[i]['cap_local'])
            local_violations[i] = excess / max(self.configs[i]['cap_local'], 1.0) * norm_factor_viol

        global_excess = max(0, self.global_cumul_carbon - self.ci_cap_total)
        global_violation = global_excess / self.ci_cap_total * norm_factor_viol

        # --- Rewards (shared total cost across all prosumers) ---
        total_system_cost = sum(step_cost.values())
        shared_reward = -total_system_cost * 0.008
        for i in self.prosumer_ids:
            rewards[i] = float(shared_reward)

        # --- Carbon price signal ---
        utilization = self.global_cumul_carbon / self.ci_cap_total
        self.carbon_price = max(0, utilization - 0.5) * 2

        # --- EV penalty at end of day ---
        self.t += 1
        done = self.t >= self.T

        # At end of episode, check if EV daily need was met
        ev_penalty = {}
        if done:
            for i in self.prosumer_ids:
                cfg = self.configs[i]
                target = cfg['ev_cap'] * 0.3 + cfg['ev_daily_need']
                shortfall = max(0, target - self.ev_soc[i])
                ev_penalty[i] = shortfall / max(cfg['ev_daily_need'], 1.0)
                rewards[i] -= ev_penalty[i] * 2.0  # penalize unmet EV charging

        info = {
            'global_cumul_carbon': float(self.global_cumul_carbon),
            'global_cap': self.ci_cap_total,
            'global_carbon_violation': float(global_violation),
            'global_utilization': float(utilization),
            'carbon_price': float(self.carbon_price),
            'local_cumul_carbon': {i: float(self.cumul_carbon[i]) for i in self.prosumer_ids},
            'local_violations': {i: float(local_violations[i]) for i in self.prosumer_ids},
            'soc': {i: float(self.soc[i]) for i in self.prosumer_ids},
            'ev_soc': {i: float(self.ev_soc[i]) for i in self.prosumer_ids},
        }

        return self._get_obs(), rewards, done, info

    def _get_obs(self):
        t = self.t
        obs = {}
        for i in self.prosumer_ids:
            cfg = self.configs[i]
            E_max = max(cfg['bess_cap'], 1.0)
            ev_cap_i = max(cfg['ev_cap'], 1.0)
            flex_avail = cfg['flex_scale'] * self.flex_profile[min(t, self.T - 1)]

            ev_need_remaining = max(0, cfg['ev_cap'] * 0.3 + cfg['ev_daily_need'] - self.ev_soc[i])
            hours_left = max(1, self.T - t)

            obs[i] = np.array([
                t / self.T,                                                         # 0: time
                cfg['load_scale'] * self.current_load_profiles[i][min(t, self.T-1)] / 300.0,  # 1: load
                cfg['pv_scale'] * self.current_pv_profile[min(t, self.T-1)] / 150.0,          # 2: pv
                self.soc[i] / E_max,                                                # 3: bess soc
                self.cumul_carbon[i] / max(cfg['cap_local'], 1.0),                  # 4: local carbon frac
                self.carbon_intensity[min(t, self.T-1)],                            # 5: carbon intensity
                max(0.0, 1.0 - self.global_cumul_carbon / self.ci_cap_total),       # 6: global carbon remaining
                self.carbon_price,                                                  # 7: carbon price signal
                flex_avail / max(cfg['flex_scale'], 1.0),                           # 8: flex available (norm)
                self.ev_soc[i] / ev_cap_i,                                         # 9: ev soc
                ev_need_remaining / (cfg['ev_charge_max'] * hours_left + 1e-6),     # 10: ev urgency
            ], dtype=np.float32)
        return obs

    @property
    def agent_ids(self):
        return self.prosumer_ids
