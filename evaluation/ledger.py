"""Evaluation of a controller on one test day: reward, operating cost, local and global costs."""
import time

import numpy as np

from power_envs import CASES


def grid_parameters(env, case, i):
    c = env.mg_configs[i] if case == "C2" else env.dso_configs[i]
    if case == "C2":
        pmax = c["P_max_mw"] * (env.current_pv_profile[env.t] if c["is_pv"] else 1)
        qlo, qhi = c["Q_min_mvar"], c["Q_max_mvar"]
        energy, soc = c["E_mwh"], env.soc[i] * c["E_mwh"]
        maximum = .9 * energy
        load = float(sum(env.mg_base_p[i]) * env.mg_load_profiles[i][env.t])
        curt_cost = env.curtail_cost - env.dr_revenue
    else:
        pmax, qlo, qhi = c["P_max"], -c["Q_max"], c["Q_max"]
        energy, soc, maximum = c["E_bess"], env.soc[i], c["E_bess"]
        load = env.dso_total_base_load[i] * env.current_load_profile[env.t]
        curt_cost = c["curtail_cost"]
    ch = min(c["P_bess_max"], max(0, maximum - soc) / env.eta_ch)
    dis = min(c["P_bess_max"], max(0, soc - .1 * energy) * env.eta_dis)
    return dict(c=c, pmax=pmax, qlo=qlo, qhi=qhi, ch=ch, dis=dis,
                bmax=c["P_bess_max"], load=load, curt_cost=curt_cost)


def canonical_action(env, case, i, action):
    """Action with the battery entry rescaled to power / rated power."""
    x = np.asarray(action, dtype=float).copy()
    p = grid_parameters(env, case, i)
    if case == "C2":
        power = np.clip((x[2] - .5) * 2 * p["bmax"], -p["dis"], p["ch"])
    else:
        power = (x[2] - .5) * 2 * (p["ch"] if x[2] >= .5 else p["dis"])
    x[2] = .5 + power / (2 * p["bmax"])
    return x


def grid_cost_before_step(env, case, actions):
    """DG, battery and curtailment costs of all agents for the current hour."""
    total = 0.
    for i in env.agent_ids:
        p = grid_parameters(env, case, i)
        x = canonical_action(env, case, i, actions[i])
        total += p["c"]["cost"] * p["pmax"] * x[0]
        total += env.bess_degradation * abs((x[2] - .5) * 2 * p["bmax"])
        total += p["curt_cost"] * p["load"] * p["c"]["curtail_max"] * x[3]
    return total


def carbon_cost_before_step(env, actions):
    """Operating cost of all prosumers for the current hour."""
    total = 0.
    t = env.t
    for i in env.agent_ids:
        c, a = env.configs[i], np.asarray(actions[i])
        load = c["load_scale"] * env.current_load_profiles[i][t]
        pv = c["pv_scale"] * env.current_pv_profile[t]
        gas = c["gas_cap"] * a[0]
        chcap = min(c["bess_cap"] / 4, max(0., c["bess_cap"] - env.soc[i]) / env.eta_ch)
        discap = min(c["bess_cap"] / 4, max(0., env.soc[i] - .1 * c["bess_cap"]) * env.eta_dis)
        raw = 2 * (a[1] - .5)
        ch, dis = max(raw, 0.) * chcap, max(-raw, 0.) * discap
        available_flex = c["flex_scale"] * env.flex_profile[t]
        flex = a[3] * available_flex
        ev = a[4] * min(c["ev_charge_max"], max(0., c["ev_cap"] - env.ev_soc[i]) / env.ev_eta)
        imports = max(a[2] * max(2 * load, 1.), load + ch + flex + ev - pv - gas - dis)
        total += imports * env.tou_price[t] + gas * env.gas_cost + (ch + dis) * env.bess_degradation - flex * env.flex_price
    return float(total)


def evaluate(case, seed, method, *, controller, env=None, obs=None):
    """Run one test day with controller(env, obs) -> (actions, diagnostics).

    The load/PV realization of the day is drawn with np.random.seed(seed).
    """
    if env is None:
        np.random.seed(seed)
        env = CASES[case]()
        obs = env.reset()
    totals = dict(cost=0., reward=0., local_violation_score=0., global_violation_score=0.,
                  control_seconds=0., simulation_seconds=0., communication_rounds=0.,
                  local_violation_steps=0., global_violation_steps=0.,
                  local_voltage_excess_puh=0., global_voltage_excess_puh=0., import_excess_mwh=0.)
    steps, min_v, max_v, max_import, failures = [], 2., 0., 0., 0
    for t in range(env.T):
        start = time.perf_counter()
        actions, diagnostics = controller(env, obs)
        elapsed = time.perf_counter() - start
        totals["control_seconds"] += elapsed
        devices = grid_cost_before_step(env, case, actions) if case != "C3" else carbon_cost_before_step(env, actions)
        start = time.perf_counter()
        obs, rewards, done, info = env.step(actions)
        totals["simulation_seconds"] += time.perf_counter() - start
        totals["reward"] += float(np.mean(list(rewards.values())))
        if case == "C2":
            price_t = max(0, t - 1)
            cost = devices + env.grid_tou[price_t] * sum(max(0, p) for p in info["P_tie"].values())
            lc, gc = sum(info["local_voltage_violations"].values()), info["global_voltage_violation"]
            local_v = np.concatenate([info["mg_voltages"][i] for i in env.agent_ids])
            global_v = env.dno_net.res_bus.vm_pu.to_numpy()
            vs = np.r_[local_v, global_v]
            nets = [env.dno_net, *env.mg_nets.values()]
            totals["local_voltage_excess_puh"] += float(np.maximum(.95 - local_v, 0).sum() + np.maximum(local_v - 1.05, 0).sum())
            totals["global_voltage_excess_puh"] += float(np.maximum(.95 - global_v, 0).sum() + np.maximum(global_v - 1.05, 0).sum())
        elif case == "C1":
            cost = devices + env.grid_tou[t] * sum(max(0, p) for p in info["P_tie"].values())
            lc, gc = sum(info["dso_voltage_violations"].values()), info["line_overload"]
            vs = np.concatenate([n.res_bus.vm_pu.to_numpy() for n in env.dso_nets.values()])
            nets = [env.tso_net, *env.dso_nets.values()]
            max_import = max(max_import, info["total_P_from_tso"])
            totals["local_voltage_excess_puh"] += float(np.maximum(.95 - vs, 0).sum() + np.maximum(vs - 1.05, 0).sum())
            totals["import_excess_mwh"] += max(0, info["total_P_from_tso"] - env.P_cap_mw)
        else:
            cost = devices
            lc, gc = sum(info["local_violations"].values()), info["global_carbon_violation"]
        if case != "C3":
            min_v, max_v = min(min_v, float(min(vs))), max(max_v, float(max(vs)))
            failures += sum(not bool(n.converged) for n in nets)
            local_bad = bool(np.min(local_v if case == "C2" else vs) < .95 - 1e-5 or np.max(local_v if case == "C2" else vs) > 1.05 + 1e-5)
            global_bad = bool(np.min(global_v) < .95 - 1e-5 or np.max(global_v) > 1.05 + 1e-5) if case == "C2" else info["total_P_from_tso"] > env.P_cap_mw + 1e-5
        else:
            local_bad, global_bad = lc > 1e-8, gc > 1e-8
        totals["cost"] += float(cost)
        totals["local_violation_score"] += float(lc)
        totals["global_violation_score"] += float(gc)
        totals["local_violation_steps"] += int(local_bad)
        totals["global_violation_steps"] += int(global_bad)
        totals["communication_rounds"] += diagnostics.get("communication_rounds", 0)
        steps.append(dict(t=t, actions={str(i): np.asarray(a).tolist() for i, a in actions.items()},
                          cost=float(cost), local_score=float(lc), global_score=float(gc),
                          control_seconds=elapsed, diagnostics=diagnostics))
    result = dict(case=case, seed=seed, method=method, pf_failures=failures, steps=steps, **totals)
    if case == "C3":
        result.update(carbon_kg=float(env.global_cumul_carbon), carbon_cap_kg=env.ci_cap_total,
                      carbon_excess_kg=max(0., float(env.global_cumul_carbon - env.ci_cap_total)),
                      local_carbon_excess_kg=float(sum(max(0, env.cumul_carbon[i] - env.configs[i]["cap_local"]) for i in env.agent_ids)),
                      ev_shortfall_kwh=float(sum(max(0, .3 * env.configs[i]["ev_cap"] + env.configs[i]["ev_daily_need"] - env.ev_soc[i]) for i in env.agent_ids)))
        result["feasible"] = all(result[k] < 1e-4 for k in ("carbon_excess_kg", "local_carbon_excess_kg", "ev_shortfall_kwh"))
    else:
        result.update(v_min_pu=min_v, v_max_pu=max_v, max_import_mw=max_import if case == "C1" else None)
        result["feasible"] = bool(min_v >= .95 - 1e-5 and max_v <= 1.05 + 1e-5 and max_import <= 8 + 1e-5 and failures == 0)
    return result
