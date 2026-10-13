import numpy as np
import gurobipy as gp
from gurobipy import GRB

def build_beso_dro(
    rho, p_OA, c, d, Q, b, M,
    K, L, seed=None, fixed_x=None
):
    """
    Build the BESO-DRO model without solving it.

    Parameters
    ----------
    rho : array (N, n)
        Nonnegative integer demands rho[s, j].
        Each row is one joint demand scenario across all stores.
    p_OA : array (N,)
        Objective-aware probabilities, in the same order as rho.
    c : array (m, n)
        Fixed assignment costs c[i, j].
    d : array (m,)
        Unit storage costs d[i].
    Q : array (m,)
        Fixed DC capacities, including the external DC.
    b : array (n,)
        Exact number of DC assignments per store.
    M : float
        Positive penalty per pallet of unmet demand.
    K : positive integer
        Number of resampled distributions.
    L : positive integer
        Number of scenario draws per distribution.
    seed : int, optional
        Random seed for reproducibility.
    fixed_x : dict, optional
        Prescribed assignments {(i, j): 0 or 1}.
        All indices are zero-based.

    Returns
    -------
    model, x, q, eps, z, w
        x[i,j]   : common binary assignments.
        q[i,j,s] : integer deliveries in scenario s.
        eps[j,s] : integer shortages in scenario s.
        z        : worst expected recourse cost.
        w[k,s]   : scenario probabilities in distribution k.
    """
    # ---------- Validate inputs ----------
    rho, p, c, d, Q, b = [
        np.asarray(v, dtype=float)
        for v in (rho, p_OA, c, d, Q, b)
    ]

    if rho.ndim != 2 or min(rho.shape) == 0:
        raise ValueError("rho must have shape (N, n), with N,n > 0.")
    N, n = rho.shape

    if c.ndim != 2 or c.shape[1] != n or c.shape[0] == 0:
        raise ValueError("c must have shape (m, n), with m > 0.")
    m = c.shape[0]

    if (p.shape != (N,) or d.shape != (m,)
            or Q.shape != (m,) or b.shape != (n,)):
        raise ValueError("Inconsistent dimensions of p_OA, d, Q, or b.")

    if not all(np.all(np.isfinite(v)) for v in (rho, p, c, d, Q, b)):
        raise ValueError("All input arrays must contain finite values.")

    if np.any(rho < 0) or np.any(Q < 0):
        raise ValueError("Demands and capacities must be nonnegative.")

    if np.any(rho != np.floor(rho)):
        raise ValueError("rho must contain integer pallet demands.")

    if np.any(b != np.floor(b)) or np.any((b < 1) | (b > m)):
        raise ValueError("b must contain integers between 1 and m.")

    if np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("p_OA must be nonnegative and sum to one.")

    if not np.isfinite(M) or M <= 0:
        raise ValueError("M must be finite and positive.")

    for name, value in (("K", K), ("L", L)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0):
            raise ValueError(f"{name} must be a positive integer.")

    I, J, S = range(m), range(n), range(N)

    if fixed_x is not None:
        for (i, j), value in fixed_x.items():
            if i not in I or j not in J or value not in (0, 1):
                raise ValueError("Invalid fixed assignment index or value.")

    # ---------- Generate ambiguity-set vertices ----------
    p = p / p.sum()
    rng = np.random.default_rng(seed)

    # Equivalent to L independent draws with replacement per vertex.
    counts = rng.multinomial(L, p, size=K)
    w = counts / float(L)

    # ---------- Create model and variables ----------
    model = gp.Model("BESO_DRO")

    # Common first-stage assignments.
    x = model.addVars(I, J, vtype=GRB.BINARY, name="x")

    # Scenario-dependent recourse.
    q = model.addVars(I, J, S, lb=0, vtype=GRB.INTEGER, name="q")
    eps = model.addVars(J, S, lb=0, vtype=GRB.INTEGER, name="eps")

    z = model.addVar(
        lb=-GRB.INFINITY,
        name="worst_expected_recourse"
    )

    # ---------- Feasibility constraints ----------
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in I) == b[j]
         for j in J),
        name="assignments"
    )

    model.addConstrs(
        (gp.quicksum(q[i, j, s] for i in I) + eps[j, s]
         == rho[s, j]
         for j in J for s in S),
        name="demand"
    )

    model.addConstrs(
        (gp.quicksum(q[i, j, s] for j in J) <= Q[i]
         for i in I for s in S),
        name="capacity"
    )

    model.addConstrs(
        (q[i, j, s] <= rho[s, j] * x[i, j]
         for i in I for j in J for s in S),
        name="link"
    )

    if fixed_x is not None:
        for (i, j), value in fixed_x.items():
            x[i, j].LB = value
            x[i, j].UB = value

    # ---------- Worst expected recourse ----------
    recourse = {
        s: (
            gp.quicksum(
                d[i] * q[i, j, s] for i in I for j in J
            )
            + M * gp.quicksum(eps[j, s] for j in J)
        )
        for s in S
    }

    for k in range(K):
        support = np.flatnonzero(w[k])
        model.addConstr(
            z >= gp.quicksum(
                float(w[k, s]) * recourse[int(s)]
                for s in support
            ),
            name=f"distribution[{k}]"
        )

    # ---------- Objective ----------
    assignment_cost = gp.quicksum(
        c[i, j] * x[i, j] for i in I for j in J
    )

    model.setObjective(
        assignment_cost + z,
        GRB.MINIMIZE
    )
    model.update()

    return model, x, q, eps, z, w

model, x, q, eps, z, w = build_beso_dro(
    rho=rho,
    p_OA=p_OA,
    c=c,
    d=d,
    Q=Q,
    b=b,
    M=M,
    K=100,       # Illustrative
    L=125,       # Illustrative
    seed=42
)

model.Params.MIPGap = 0.01
model.optimize()

if model.SolCount > 0:
    assignments = [(i, j) for i, j in x if x[i, j].X > 0.5]
    print("Objective:", model.ObjVal)
    print("Worst expected recourse bound:", z.X)
    print("Assignments:", assignments)
else:
    print("No feasible solution found. Gurobi status:", model.Status)