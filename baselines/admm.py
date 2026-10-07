"""Distributed MPC (ADMM) and decentralized MPC (max_iter=0) on the per-agent models.

C1: sharing ADMM on the hourly import cap. C2: consensus ADMM between the microgrids and the
network operator on the power exchange and the root voltage. C3: sharing ADMM on the community
carbon cap. Each ADMM iteration is one communication round between the agents and a coordinator.
"""
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from baselines.model import battery, circuit, forecasts, new_model, plan_action

PENALTY = 1e5  # weight of the constraint slacks
MU, TAU = 10.0, 2.0  # residual balancing (Boyd et al., 2011)


def local_model(name):
    m = new_model(name, limit=10)
    m.Params.Threads = 1
    m.Params.MIPGap = 1e-4
    return m


class Local:
    """One agent's model for the current hour, kept across ADMM iterations."""

    def __init__(self, m, cost, coupling, plan):
        self.m, self.cost, self.coupling, self.plan = m, cost, coupling, plan
        self.seconds = 0.0

    def solve(self, targets=None, rho=None):
        obj = self.cost
        if targets is not None:
            for g, xs in self.coupling.items():
                obj = obj + rho[g] / 2 * gp.quicksum((x - float(a)) * (x - float(a)) for x, a in zip(xs, targets[g]))
        self.m.setObjective(obj)
        start = time.perf_counter()
        self.m.optimize()
        self.seconds = time.perf_counter() - start
        return {g: np.array([x.X for x in xs]) for g, xs in self.coupling.items()}


# --------------------------------------------------------------------------- C3
def carbon_local(env, i, hours, load, pv):
    c = env.configs[i]
    m = local_model(f"C3_admm_{i}")
    bc = {i: {"pmax": c["bess_cap"] / 4, "emin": c["bess_cap"] * .1, "emax": c["bess_cap"]}}
    ch, dis, _ = battery(m, [i], hours, bc, env.soc, env.eta_ch, env.eta_dis)
    grid = m.addVars(hours, lb=0, name="grid")
    gas = m.addVars(hours, lb=0, ub=c["gas_cap"], name="gas")
    flex = m.addVars(hours, lb=0, name="flex")
    ev = m.addVars(hours, lb=0, ub=c["ev_charge_max"], name="ev")
    esoc = m.addVars(hours, lb=0, ub=c["ev_cap"], name="ev_soc")
    for k, t in enumerate(hours):
        flex[t].UB = c["flex_scale"] * env.flex_profile[t]
        m.addConstr(esoc[t] == (esoc[hours[k - 1]] if k else env.ev_soc[i]) + env.ev_eta * ev[t])
        m.addConstr(grid[t] + gas[t] + dis[i, t] + c["pv_scale"] * pv[t]
                    >= c["load_scale"] * load[i, t] + ch[i, t] + flex[t] + ev[t])
    m.addConstr(esoc[hours[-1]] >= c["ev_cap"] * .3 + c["ev_daily_need"])
    emissions = gp.quicksum(grid[t] * env.carbon_intensity[t] + gas[t] * env.gas_ci for t in hours)
    local_slack = m.addVar(lb=0, name="local_carbon_slack")
    m.addConstr(env.cumul_carbon[i] + emissions <= c["cap_local"] + local_slack)
    future = m.addVar(lb=-GRB.INFINITY, name="future_emissions")
    m.addConstr(future == emissions)
    economic = gp.quicksum(grid[t] * env.tou_price[t] + gas[t] * env.gas_cost
                           + (ch[i, t] + dis[i, t]) * env.bess_degradation - flex[t] * env.flex_price for t in hours)
    cost = economic + PENALTY * local_slack / c["cap_local"]

    def plan(t):
        return {"gas": gas[t].X, "ch": ch[i, t].X, "dis": dis[i, t].X, "grid": grid[t].X,
                "flex": flex[t].X, "ev": ev[t].X}

    local = Local(m, cost, {"share": [future]}, plan)
    local.economic = economic
    return local


# --------------------------------------------------------------------------- C2 / C1
def grid_local(env, case, i, hours, load, pv, loss_weight, rootv_fixed, relax=False):
    """One MG (C2) or DSO (C1) with its own branch-flow circuit; relax=True makes the battery mode continuous."""
    c = (env.mg_configs if case == "C2" else env.dso_configs)[i]
    net = (env.mg_nets if case == "C2" else env.dso_nets)[i]
    basep = (env.mg_base_p if case == "C2" else env.dso_base_p)[i]
    baseq = (env.mg_base_q if case == "C2" else env.dso_base_q)[i]
    E = c["E_mwh"] if case == "C2" else c["E_bess"]
    m = local_model(f"{case}_admm_{i}")
    bc = {i: {"pmax": c["P_bess_max"], "emin": E * .1, "emax": E * (.9 if case == "C2" else 1)}}
    init = {i: env.soc[i] * E if case == "C2" else env.soc[i]}
    ch, dis, _ = battery(m, [i], hours, bc, init, env.eta_ch, env.eta_dis)
    if relax:
        m.update()
        for var in m.getVars():
            if var.VarName.startswith("bmode"):
                var.VType = GRB.CONTINUOUS
    dg = m.addVars(hours, lb=0, name="dg")
    qg = m.addVars(hours, lb=-GRB.INFINITY, name="qg")
    curt = m.addVars(hours, lb=0, ub=c["curtail_max"], name="curt")
    imports = m.addVars(hours, lb=0, name="import")
    # C2: the root voltage of hour t is the network voltage of hour t-1 (coupling variable)
    rootv = {t: m.addVar(lb=.5**2, ub=1.5**2, name=f"root_{t}") for t in hours}
    lp, lq, pg, qgen = {}, {}, {}, {}
    for t in hours:
        dg[t].UB = c["P_max_mw"] * (pv[t] if c["is_pv"] else 1) if case == "C2" else c["P_max"]
        qg[t].LB, qg[t].UB = (c["Q_min_mvar"], c["Q_max_mvar"]) if case == "C2" else (-c["Q_max"], c["Q_max"])
        for j, row in enumerate(net.load.itertuples()):
            b = int(row.bus)
            lp[b, t] = lp.get((b, t), 0) + float(basep[j]) * load[i, t] * (1 - curt[t])
            lq[b, t] = lq.get((b, t), 0) + float(baseq[j]) * load[i, t] * (1 - curt[t])
        dgb, bb = (6, 3) if case == "C2" else (env.dg_bus, env.bess_bus)
        pg[dgb, t], qgen[dgb, t] = dg[t], qg[t]
        pg[bb, t] = dis[i, t] - ch[i, t]
        if t in rootv_fixed:
            rootv[t].LB = rootv[t].UB = rootv_fixed[t]
    cir = circuit(m, f"local{i}_", net, hours, lp, lq, pg, qgen, rootv, soft=True)
    for t in hours:
        m.addConstr(imports[t] >= cir["pin"][t])
    price = {t: env.grid_tou[max(t - 1, 0) if case == "C2" else t] for t in hours}
    curt_price = (env.curtail_cost - env.dr_revenue) if case == "C2" else c["curtail_cost"]
    economic = gp.quicksum(price[t] * imports[t] + c["cost"] * dg[t] + env.bess_degradation * (ch[i, t] + dis[i, t])
                           + curt_price * float(sum(basep)) * load[i, t] * curt[t] for t in hours)
    losses = gp.quicksum(cir["r"][e] * cir["ell"][e[0], e[1], t] for e in cir["edges"] for t in hours)
    cost = economic + loss_weight * losses + PENALTY * cir["slack"].sum()

    def plan(t):
        return {"dg": dg[t].X, "qg": qg[t].X, "ch": ch[i, t].X, "dis": dis[i, t].X, "curt": curt[t].X}

    if case == "C2":
        coupling = {"p": [cir["pin"][t] for t in hours], "q": [cir["qin"][t] for t in hours],
                    "v": [rootv[t] for t in hours[1:]]}
    else:
        coupling = {"share": [cir["pin"][t] for t in hours]}
    local = Local(m, cost, coupling, plan)
    local.economic = economic
    local.rootv = rootv
    return local


def dno_local(env, hours, load, loss_weight):
    """Network operator of C2: its circuit with copies of the MG exchanges."""
    m = local_model("C2_admm_dno")
    P = {(i, t): m.addVar(lb=-GRB.INFINITY, name=f"P_{i}_{t}") for i in env.agent_ids for t in hours}
    Q = {(i, t): m.addVar(lb=-GRB.INFINITY, name=f"Q_{i}_{t}") for i in env.agent_ids for t in hours}
    lp, lq = {}, {}
    for t in hours:
        for j, row in enumerate(env.dno_net.load.iloc[:len(env.dno_base_p)].itertuples()):
            b = int(row.bus)
            lp[b, t] = lp.get((b, t), 0) + float(env.dno_base_p[j]) * load["dno", t]
            lq[b, t] = lq.get((b, t), 0) + float(env.dno_base_q[j]) * load["dno", t]
        for i in env.agent_ids:
            lp[i, t] = lp.get((i, t), 0) + P[i, t]
            lq[i, t] = lq.get((i, t), 0) + Q[i, t]
    root = float(env.dno_net.ext_grid.vm_pu.iloc[0]) ** 2
    cir = circuit(m, "dno_", env.dno_net, hours, lp, lq, {}, {}, {t: root for t in hours}, soft=True)
    losses = gp.quicksum(cir["r"][e] * cir["ell"][e[0], e[1], t] for e in cir["edges"] for t in hours)
    cost = loss_weight * losses + PENALTY * cir["slack"].sum()
    coupling = {}
    for i in env.agent_ids:
        coupling[i] = {"p": [P[i, t] for t in hours], "q": [Q[i, t] for t in hours],
                       "v": [cir["v"][i, t] for t in hours[:-1]]}
    flat = {g: [x for i in env.agent_ids for x in coupling[i][g]] for g in ("p", "q", "v")}
    local = Local(m, cost, flat, None)
    local.per_agent = coupling
    return local


# --------------------------------------------------------------------------- coordinator
def project_share(v, cap, penalty, rho):
    """argmin_z sum (rho/2)(z_i - v_i)^2 + penalty * max(0, sum z - cap), per column."""
    v = np.atleast_2d(v)
    excess = v.sum(axis=0) - cap
    delta = np.where(excess > 0, np.minimum(excess / v.shape[0], penalty / rho), 0.0)
    return v - delta


def balance(rho, u, r, s):
    if r > MU * s:
        return rho * TAU, u / TAU
    if s > MU * r:
        return rho / TAU, u * TAU
    return rho, u


class ADMMPlanner:
    """Controller for evaluation.ledger: distributed MPC, or decentralized MPC with max_iter=0."""

    def __init__(self, case, max_iter=300, eps_abs=None, eps_rel=1e-3, loss_weight=None,
                 relax=False, adapt_iters=50, alpha=1.0, rho0=None, record=False):
        self.case, self.max_iter, self.eps_rel = case, max_iter, eps_rel
        self.relax, self.adapt_iters, self.alpha, self.rho0 = relax, adapt_iters, alpha, rho0
        self.record, self.history = record, []
        self.eps_abs = eps_abs if eps_abs is not None else {"C2": 1e-4, "C1": 1e-4, "C3": 1e-2}[case]
        self.loss_weight = loss_weight if loss_weight is not None else {"C2": 30., "C1": .01, "C3": 0.}[case]
        self.warm = None

    def __call__(self, env, obs):
        hours = list(range(env.t, env.T))
        load, pv = forecasts(env, self.case, hours)
        if self.case == "C2":
            plan, diag = self._consensus(env, hours, load, pv)
        else:
            plan, diag = self._sharing(env, hours, load, pv)
        return plan_action(env, self.case, plan), diag

    # sharing ADMM (C1, C3): sum_i x_i <= cap
    def _sharing(self, env, hours, load, pv):
        ids = list(env.agent_ids)
        locals_, build_times = {}, []
        for i in ids:
            start = time.perf_counter()
            if self.case == "C3":
                locals_[i] = carbon_local(env, i, hours, load, pv)
            else:
                root = float(env.dso_nets[i].ext_grid.vm_pu.iloc[0]) ** 2
                locals_[i] = grid_local(env, "C1", i, hours, load, pv, self.loss_weight, {t: root for t in hours},
                                        relax=self.relax)
            build_times.append(time.perf_counter() - start)
        if self.case == "C3":
            cap = np.array([env.ci_cap_total - env.global_cumul_carbon])
            penalty = PENALTY / env.ci_cap_total
            rho = self.warm["rho"] if self.warm else 1e-3
        else:
            cap = np.full(len(hours), env.P_cap_mw)
            penalty = PENALTY / 8
            rho = self.warm["rho"] if self.warm else 100.0
        build_s, build_par = sum(build_times), max(build_times)
        seq = par = 0.0
        # iteration 0: uncoordinated local optimum
        x = np.array([locals_[i].solve()["share"] for i in ids])
        times = [locals_[i].seconds for i in ids]
        seq, par = seq + sum(times), par + max(times)
        u = np.zeros_like(x)
        if self.warm is not None and self.warm["u"].shape[1] >= x.shape[1]:
            u = self.warm["u"][:, -x.shape[1]:] if self.case == "C1" else self.warm["u"]
        z = project_share(x + u, cap, penalty, rho)
        converged, k, r, s = False, 0, 0.0, 0.0
        for k in range(1, self.max_iter + 1):
            x = np.array([locals_[i].solve({"share": z[n] - u[n]}, {"share": rho})["share"] for n, i in enumerate(ids)])
            times = [locals_[i].seconds for i in ids]
            start = time.perf_counter()
            z_old = z
            z = project_share(x + u, cap, penalty, rho)
            u = u + x - z
            coord = time.perf_counter() - start
            seq, par = seq + sum(times) + coord, par + max(times) + coord
            r = np.linalg.norm(x - z)
            s = rho * np.linalg.norm(z - z_old)
            eps_pri = np.sqrt(x.size) * self.eps_abs + self.eps_rel * max(np.linalg.norm(x), np.linalg.norm(z))
            eps_dual = np.sqrt(x.size) * self.eps_abs + self.eps_rel * rho * np.linalg.norm(u)
            if r <= eps_pri and s <= eps_dual:
                converged = True
                break
            if k <= self.adapt_iters:
                rho, u = balance(rho, u, r, s)
        # warm start for the next hour
        self.warm = {"rho": rho, "u": u[:, 1:] if self.case == "C1" else u}
        t = env.t
        plan = {i: locals_[i].plan(t) for i in ids}
        economic = sum(locals_[i].economic.getValue() for i in ids)
        excess = float(np.maximum(x.sum(axis=0) - cap, 0).max())
        for i in ids:
            locals_[i].m.dispose()
        return plan, dict(admm_iterations=k, admm_converged=converged, communication_rounds=k,
                          admm_primal_residual=float(r), admm_dual_residual=float(s),
                          admm_rho=float(rho), admm_sequential_s=seq + build_s, admm_parallel_s=par + build_par,
                          admm_build_s=build_s, admm_planned_coupling_excess=excess,
                          admm_planned_economic_cost=float(economic))

    # consensus ADMM (C2): MG side x = (P, Q, root voltage), network side z = copies
    def _consensus(self, env, hours, load, pv):
        ids = list(env.agent_ids)
        groups = {"pq": ("p", "q"), "v": ("v",)}
        eps_abs = {"pq": self.eps_abs, "v": self.eps_abs / 10}
        locals_, build_times = {}, []
        for i in ids:
            start = time.perf_counter()
            locals_[i] = grid_local(env, "C2", i, hours, load, pv, self.loss_weight,
                                    {hours[0]: float(env.V_tie[i]) ** 2}, relax=self.relax)
            build_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        dno = dno_local(env, hours, load, self.loss_weight)
        dno_build = time.perf_counter() - start
        build_s, build_par = sum(build_times) + dno_build, max(build_times) + dno_build

        def keys(i, g):
            return [(g, i, t) for t in (hours if g in ("p", "q") else hours[:-1])]

        def flatten(per_agent):
            return {g: np.concatenate([per_agent[i][g] for i in ids]) for g in ("p", "q", "v")}

        def unflatten(flat):
            out, pos = {i: {} for i in ids}, {g: 0 for g in flat}
            for i in ids:
                for g in flat:
                    n = len(keys(i, g))
                    out[i][g] = flat[g][pos[g]:pos[g] + n]
                    pos[g] += n
            return out

        seq = par = 0.0
        if self.max_iter == 0:
            for i in ids:
                for t in hours[1:]:
                    locals_[i].rootv[t].LB = locals_[i].rootv[t].UB = float(env.V_tie[i]) ** 2
            for i in ids:
                locals_[i].solve()
            times = [locals_[i].seconds for i in ids]
            plan = {i: locals_[i].plan(env.t) for i in ids}
            economic = sum(locals_[i].economic.getValue() for i in ids)
            for i in ids:
                locals_[i].m.dispose()
            dno.m.dispose()
            return plan, dict(admm_iterations=0, admm_converged=False, communication_rounds=0,
                              admm_sequential_s=sum(times) + sum(build_times),
                              admm_parallel_s=max(times) + max(build_times),
                              admm_build_s=sum(build_times), admm_planned_economic_cost=float(economic))
        if self.warm is not None:
            z = {g: np.array([self.warm["z"][key] for i in ids for key in keys(i, g)]) for g in ("p", "q", "v")}
            u = {g: np.array([self.warm["u"][key] for i in ids for key in keys(i, g)]) for g in ("p", "q", "v")}
            rho = dict(self.warm["rho"])
        else:
            # iteration 0: MGs plan alone with the measured root voltage held all day
            for i in ids:
                for t in hours[1:]:
                    locals_[i].rootv[t].LB = locals_[i].rootv[t].UB = float(env.V_tie[i]) ** 2
            x = flatten({i: locals_[i].solve() for i in ids})
            times = [locals_[i].seconds for i in ids]
            for i in ids:
                for t in hours[1:]:
                    locals_[i].rootv[t].LB, locals_[i].rootv[t].UB = .5**2, 1.5**2
            rho = dict(self.rho0 or {"p": 100.0, "q": 100.0, "v": 1e4})
            u = {g: np.zeros_like(x[g]) for g in x}
            z = dno.solve({g: x[g] + u[g] for g in x}, rho)
            u = {g: u[g] + x[g] - z[g] for g in x}
            seq, par = seq + sum(times) + dno.seconds, par + max(times) + dno.seconds
        converged, k, r, s = False, 0, {}, {}
        for k in range(1, self.max_iter + 1):
            targets = unflatten({g: z[g] - u[g] for g in z})
            x = flatten({i: locals_[i].solve(targets[i], rho) for i in ids})
            times = [locals_[i].seconds for i in ids]
            z_old = z
            xr = {g: self.alpha * x[g] + (1 - self.alpha) * z_old[g] for g in x}  # over-relaxation
            z = dno.solve({g: xr[g] + u[g] for g in x}, rho)
            u = {g: u[g] + xr[g] - z[g] for g in x}
            seq, par = seq + sum(times) + dno.seconds, par + max(times) + dno.seconds
            done = True
            for name, gs in groups.items():
                xv = np.concatenate([x[g] for g in gs])
                zv = np.concatenate([z[g] for g in gs])
                uv = np.concatenate([u[g] for g in gs])
                r[name] = np.linalg.norm(xv - zv)
                s[name] = rho[gs[0]] * np.linalg.norm(zv - np.concatenate([z_old[g] for g in gs]))
                eps_pri = np.sqrt(xv.size) * eps_abs[name] + self.eps_rel * max(np.linalg.norm(xv), np.linalg.norm(zv))
                eps_dual = np.sqrt(xv.size) * eps_abs[name] + self.eps_rel * rho[gs[0]] * np.linalg.norm(uv)
                done &= bool(r[name] <= eps_pri and s[name] <= eps_dual)
            if self.record:
                self.history.append(dict(k=k, r=dict(r), s=dict(s), rho=dict(rho),
                                         max_mismatch={g: float(np.abs(x[g] - z[g]).max()) for g in x},
                                         cost=float(sum(locals_[i].economic.getValue() for i in ids))))
            if done:
                converged = True
                break
            for name, gs in groups.items():
                if k > self.adapt_iters:
                    break
                new_rho, _ = balance(rho[gs[0]], 0.0, r[name], s[name])
                for g in gs:
                    u[g] = u[g] * rho[g] / new_rho
                    rho[g] = new_rho
        self.warm = {"rho": rho,
                     "z": {key: val for g in z for key, val in zip([kk for i in ids for kk in keys(i, g)], z[g])},
                     "u": {key: val for g in u for key, val in zip([kk for i in ids for kk in keys(i, g)], u[g])}}
        plan = {i: locals_[i].plan(env.t) for i in ids}
        economic = sum(locals_[i].economic.getValue() for i in ids)
        for i in ids:
            locals_[i].m.dispose()
        dno.m.dispose()
        return plan, dict(admm_iterations=k, admm_converged=converged, communication_rounds=k,
                          admm_primal_residual={n: float(v) for n, v in r.items()},
                          admm_dual_residual={n: float(v) for n, v in s.items()},
                          admm_rho={g: float(v) for g, v in rho.items()}, admm_sequential_s=seq + build_s,
                          admm_parallel_s=par + build_par, admm_build_s=build_s,
                          admm_planned_economic_cost=float(economic))
