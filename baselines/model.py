"""Optimization model of the three cases used by the MPC baselines.

C1, C2: radial branch-flow (DistFlow) model with second-order-cone relaxation and battery
charge/discharge binaries. C3: energy and carbon balance with battery binaries.
"""
import gurobipy as gp
import numpy as np
from gurobipy import GRB


def new_model(name, limit=30):
    m = gp.Model(name)
    m.Params.OutputFlag = 0
    m.Params.Threads = 2
    m.Params.TimeLimit = limit
    m.Params.MIPGap = 1e-5
    m.Params.FeasibilityTol = 1e-8
    m.Params.OptimalityTol = 1e-8
    m.Params.BarQCPConvTol = 1e-8
    return m



def battery(m, ids, hours, configs, init, eta_ch, eta_dis, prefix="b"):
    ch = m.addVars(ids, hours, lb=0, name=prefix + "ch")
    dis = m.addVars(ids, hours, lb=0, name=prefix + "dis")
    soc = m.addVars(ids, hours, lb=0, name=prefix + "soc")
    mode = m.addVars(ids, hours, vtype=GRB.BINARY, name=prefix + "mode")
    for i in ids:
        c = configs[i]
        for k, t in enumerate(hours):
            m.addConstr(ch[i, t] <= c["pmax"] * mode[i, t])
            m.addConstr(dis[i, t] <= c["pmax"] * (1 - mode[i, t]))
            prev = soc[i, hours[k - 1]] if k else init[i]
            m.addConstr(soc[i, t] == prev + eta_ch * ch[i, t] - dis[i, t] / eta_dis)
            soc[i, t].LB, soc[i, t].UB = c["emin"], c["emax"]
    return ch, dis, soc



def topology(net):
    """Radial circuit of a pandapower network: lines oriented away from the slack, p.u. on 1 MVA."""
    buses = list(net.bus.index)
    lines = net.line[net.line.in_service]
    edges, r, x = [], {}, {}
    for _, row in lines.iterrows():
        e = (int(row.from_bus), int(row.to_bus))
        edges.append(e)
        zbase = float(net.bus.at[e[0], "vn_kv"]) ** 2
        r[e] = float(row.r_ohm_per_km * row.length_km) / zbase
        x[e] = float(row.x_ohm_per_km * row.length_km) / zbase
    root = int(net.ext_grid.bus.iloc[0])
    parent = {j: (i, j) for i, j in edges}
    children = {b: [e for e in edges if e[0] == b] for b in buses}
    return buses, edges, r, x, root, parent, children


def circuit(m, name, net, hours, loadp, loadq, genp, genq, rootv,
            soft=False, vmin=.95, vmax=1.05):
    """Branch-flow model with the second-order-cone relaxation; voltage limits softened if soft."""
    buses, edges, r, x, root, parent, children = topology(net)
    v = m.addVars(buses, hours, lb=.5**2, ub=1.5**2, name=name + "v")
    p = m.addVars(edges, hours, lb=-GRB.INFINITY, name=name + "p")
    q = m.addVars(edges, hours, lb=-GRB.INFINITY, name=name + "q")
    ell = m.addVars(edges, hours, lb=0, ub=2000, name=name + "ell")
    pin = m.addVars(hours, lb=-GRB.INFINITY, name=name + "pin")
    qin = m.addVars(hours, lb=-GRB.INFINITY, name=name + "qin")
    slack = m.addVars(buses, hours, lb=0, ub=GRB.INFINITY if soft else 0, name=name + "vslack")
    for t in hours:
        m.addConstr(v[root, t] == rootv[t])
        for b in buses:
            m.addConstr(v[b, t] >= vmin**2 - slack[b, t])
            m.addConstr(v[b, t] <= vmax**2 + slack[b, t])
            outp = gp.quicksum(p[i, j, t] for i, j in children[b])
            outq = gp.quicksum(q[i, j, t] for i, j in children[b])
            if b == root:
                inp, inq = pin[t], qin[t]
            else:
                i, j = parent[b]
                inp = p[i, j, t] - r[i, j] * ell[i, j, t]
                inq = q[i, j, t] - x[i, j] * ell[i, j, t]
            m.addConstr(inp + genp.get((b, t), 0) == outp + loadp.get((b, t), 0))
            m.addConstr(inq + genq.get((b, t), 0) == outq + loadq.get((b, t), 0))
        for i, j in edges:
            m.addConstr(v[j, t] == v[i, t] - 2 * (r[i, j] * p[i, j, t] + x[i, j] * q[i, j, t])
                        + (r[i, j]**2 + x[i, j]**2) * ell[i, j, t])
            m.addQConstr(p[i, j, t]**2 + q[i, j, t]**2 <= v[i, t] * ell[i, j, t])
    return {"v": v, "p": p, "q": q, "ell": ell, "pin": pin, "qin": qin,
            "slack": slack, "edges": edges, "r": r}



def forecasts(env, case, hours):
    """Realized load and PV for the current hour, base profiles for the later hours."""
    load, pv = {}, {}
    for t in hours:
        pv[t] = float(env.current_pv_profile[t] if t == env.t else env.base_pv_profile[t])
        if case == "C2":
            for i in env.agent_ids:
                load[i, t] = float(env.mg_load_profiles[i][t] if t == env.t
                                   else np.roll(env.base_load_profile, env.mg_load_shift[i])[t])
            load["dno", t] = float(env.current_load_profile[t] if t == env.t else env.base_load_profile[t])
        elif case == "C1":
            for i in env.agent_ids:
                load[i, t] = float(env.current_load_profile[t] if t == env.t else env.base_load_profile[t])
        else:
            for i in env.agent_ids:
                load[i, t] = float(env.current_load_profiles[i][t] if t == env.t else env.prosumer_load_profiles[i][t])
    return load, pv


def plan_action(env, case, plan):
    actions = {}
    for i, p in plan.items():
        if case == "C3":
            c = env.configs[i]
            chcap = min(c["bess_cap"] / 4, max(0, c["bess_cap"] - env.soc[i]) / env.eta_ch)
            discap = min(c["bess_cap"] / 4, max(0, env.soc[i] - c["bess_cap"] * .1) * env.eta_dis)
            bn = .5 + .5 * (p["ch"] / max(chcap, 1e-10) if p["ch"] > 1e-7 else -p["dis"] / max(discap, 1e-10))
            evcap = min(c["ev_charge_max"], max(0, c["ev_cap"] - env.ev_soc[i]) / env.ev_eta)
            avail = c["flex_scale"] * env.flex_profile[env.t]
            # grid purchase action 0: the environment buys the planned deficit
            a = [p["gas"] / c["gas_cap"], bn, 0, p["flex"] / max(avail, 1e-10), p["ev"] / max(evcap, 1e-10)]
        else:
            c = env.mg_configs[i] if case == "C2" else env.dso_configs[i]
            pmax = c["P_max_mw"] * (env.current_pv_profile[env.t] if c["is_pv"] else 1) if case == "C2" else c["P_max"]
            qlo, qhi = (c["Q_min_mvar"], c["Q_max_mvar"]) if case == "C2" else (-c["Q_max"], c["Q_max"])
            if case == "C2":
                bn = .5 + .5 * (p["ch"] - p["dis"]) / c["P_bess_max"]
            else:
                chcap = min(c["P_bess_max"], max(0, c["E_bess"] - env.soc[i]) / env.eta_ch)
                discap = min(c["P_bess_max"], max(0, env.soc[i] - c["E_bess"] * .1) * env.eta_dis)
                bn = .5 + .5 * (p["ch"] / max(chcap, 1e-10) if p["ch"] > 1e-7 else -p["dis"] / max(discap, 1e-10))
            a = [p["dg"] / max(pmax, 1e-10), (p["qg"] - qlo) / (qhi - qlo), bn, p["curt"] / c["curtail_max"]]
        actions[i] = np.clip(a, 0, 1)
    return actions
