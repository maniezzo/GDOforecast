import numpy as np
import pandas as pd
from scipy.optimize import linprog
import math, random

# ================================================================
#  Approximate Wasserstein DRO vs SAA example for stochastic LP
# ================================================================
# Problem:
#   min sum_{i,j} (c_ij x_ij) + E[d_i q_ij]
#   s.t.   sum_i q_ij = rho_j      (balance)
#          sum_j q_ij <= Q_i       (capacity)
#          sum_i x_ij = 1          (assignment)
#          q_ij <= rho_j x_ij      (link)
#          x_ij in [0,1], q_ij >= 0
# ================================================================

# random seed
random.seed(995)
np.random.seed(995)

# problem dimensions
I, J = 2, 3
n_train, n_test = 40, 200

# costs and capacities
c = np.abs(np.random.randn(I,J)) * 5
d = np.abs(np.random.randn(I)) * 3
Q = np.array([100.0, 100.0])

# random demands
rho_mean    = np.array([8.0, 10.0, 6.0])
rho_scale   = np.array([2.0, 3.0, 1.5])
rho_samples = np.maximum(0.1, np.random.randn(n_train, J) * rho_scale + rho_mean)
rho_test    = np.maximum(0.1, np.random.randn(n_test, J) * rho_scale + rho_mean)

# indexing helpers
def idx_x(i,j):
   return i*J + j
def idx_q(s,i,j,n_x,n_q_block):
   return n_x + s*n_q_block + i*J + j

# 1. SAA solver (continuous relaxation)
def solve_saa_lp_linprog(rho_samples, c, d, Q, I, J, eps_penalty=0.0, lip_constant=0.0):
    S = len(rho_samples)
    n_x = I * J
    n_q_block = I * J
    Nvars = n_x + S*n_q_block

    cvec = np.zeros(Nvars)
    for i in range(I):
        for j in range(J):
            cvec[idx_x(i,j)] = c[i,j]
    for s in range(S):
        for i in range(I):
            for j in range(J):
                cvec[idx_q(s,i,j,n_x,n_q_block)] = (1.0/S) * d[i]

    A_eq, b_eq, A_ub, b_ub = [], [], [], []

    # assignment: sum_i x_ij = 1
    for j in range(J):
        row = np.zeros(Nvars)
        for i in range(I):
            row[idx_x(i,j)] = 1.0
        A_eq.append(row); b_eq.append(1.0)

    # balance: sum_i q_ij^s = rho_j^s
    for s in range(S):
        rho_s = rho_samples[s]
        for j in range(J):
            row = np.zeros(Nvars)
            for i in range(I):
                row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
            A_eq.append(row); b_eq.append(float(rho_s[j]))

    # capacity: sum_j q_ij^s <= Q_i
    for s in range(S):
        for i in range(I):
            row = np.zeros(Nvars)
            for j in range(J):
                row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
            A_ub.append(row); b_ub.append(float(Q[i]))

    # linking: q_ij^s <= rho_j^s x_ij
    for s in range(S):
        rho_s = rho_samples[s]
        for i in range(I):
            for j in range(J):
                row = np.zeros(Nvars)
                row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
                row[idx_x(i,j)] = -float(rho_s[j])
                A_ub.append(row); b_ub.append(0.0)

    bounds = []
    for v in range(Nvars):
        if v < n_x:
            bounds.append((0.0, 1.0))
        else:
            bounds.append((0.0, None))

    res = linprog(c=cvec, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method='highs')

    if not res.success:
        return {"status":"fail", "message":res.message}
    sol = res.x
    x_val = sol[:I*J].reshape((I,J))
    q_val = sol[I*J:].reshape((S,I,J))
    obj = res.fun + eps_penalty * lip_constant
    return {"status":"ok", "x":x_val, "q":q_val, "obj":obj}

# 2. Recourse evaluation (for fixed x, given rho)
def compute_recourse_linprog(x_val, rho_s, d, Q, I, J):
    Nq = I*J
    c_q = np.array([d[i] for i in range(I) for j in range(J)])

    # balance constraints
    A_eq = np.zeros((J, Nq))
    b_eq = np.zeros(J)
    for j in range(J):
        for i in range(I):
            A_eq[j, i*J+j] = 1.0
        b_eq[j] = float(rho_s[j])

    # capacity
    A_ub = np.zeros((I, Nq))
    b_ub = np.zeros(I)
    for i in range(I):
        for j in range(J):
            A_ub[i, i*J+j] = 1.0
        b_ub[i] = float(Q[i])

    # bounds
    bounds = []
    for i in range(I):
        for j in range(J):
            bounds.append((0.0, float(rho_s[j]*x_val[i,j])))

    res = linprog(c=c_q, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        return 1e6, None
    qsol = res.x.reshape((I,J))
    return float(res.fun), qsol

# 3. Lipschitz estimation (finite differences)
def estimate_lipschitz_numeric(x_val, rho_samples, d, Q, I, J, delta=1e-3):
    diffs = []
    for s in range(len(rho_samples)):
        rho_s = rho_samples[s]
        base, _ = compute_recourse_linprog(x_val, rho_s, d, Q, I, J)
        if base > 1e5: continue
        for j in range(J):
            rho_pert = rho_s.copy(); rho_pert[j] += delta
            pert, _ = compute_recourse_linprog(x_val, rho_pert, d, Q, I, J)
            if pert > 1e5: continue
            diffs.append(abs(pert - base) / delta)
    if len(diffs)==0: return 1e6
    return max(diffs)

# 4. Approximate W-DRO solver (iterative penalty approach)
def solve_approx_wdro_linprog(rho_samples, c, d, Q, I, J, eps, n_iter=5):
    res = solve_saa_lp_linprog(rho_samples, c, d, Q, I, J)
    x_cur = res["x"]
    lip_cur = estimate_lipschitz_numeric(x_cur, rho_samples, d, Q, I, J)
    history = []
    for it in range(n_iter):
        res = solve_saa_lp_linprog(rho_samples, c, d, Q, I, J,
                                   eps_penalty=eps, lip_constant=lip_cur)
        if res["status"] != "ok": break
        x_new = res["x"]
        lip_new = estimate_lipschitz_numeric(x_new, rho_samples, d, Q, I, J)
        history.append({"it":it, "obj":res["obj"], "lip":lip_new})
        if np.linalg.norm(x_new - x_cur) < 1e-4 and abs(lip_new - lip_cur) < 1e-2:
            x_cur, lip_cur = x_new, lip_new
            break
        x_cur, lip_cur = x_new, lip_new
    return {"x":x_cur, "lip":lip_cur, "history":history, "final_obj":res.get("obj",None)}

# 5. Evaluation routine
def evaluate_solution_linprog(x_val, rho_test, d, Q, I, J):
    Ls = []
    infeas = 0
    for s in range(len(rho_test)):
        val, _ = compute_recourse_linprog(x_val, rho_test[s], d, Q, I, J)
        if val > 1e5:
            infeas += 1
        else:
            Ls.append(val)
    return {"avg_recourse": np.mean(Ls) if len(Ls)>0 else np.nan,
            "infeas": infeas,
            "valid": len(Ls)}

# Run SAA and W-DRO comparison
if __name__ == "__main__":
   saa_res = solve_saa_lp_linprog(rho_samples, c, d, Q, I, J)
   x_saa   = saa_res["x"]
   lip_saa = estimate_lipschitz_numeric(x_saa, rho_samples, d, Q, I, J)
   
   eps = 2.0 # coefficient for wasserstein ragius
   wdro_res = solve_approx_wdro_linprog(rho_samples, c, d, Q, I, J, eps=eps, n_iter=6)
   x_wdro = wdro_res["x"]
   
   # evaluate
   eval_saa  = evaluate_solution_linprog(x_saa, rho_test, d, Q, I, J)
   eval_wdro = evaluate_solution_linprog(x_wdro, rho_test, d, Q, I, J)
   
   df = pd.DataFrame({
       "method": ["SAA", "W-DRO (approx)"],
       "first_stage_cost": [np.sum(c * x_saa), np.sum(c * x_wdro)],
       "train_avg_recourse": [
           np.mean([compute_recourse_linprog(x_saa, rho_samples[s], d, Q, I, J)[0] for s in range(n_train)]),
           np.mean([compute_recourse_linprog(x_wdro, rho_samples[s], d, Q, I, J)[0] for s in range(n_train)])
       ],
       "approx_lip": [lip_saa, wdro_res["lip"]],
       "test_avg_recourse": [eval_saa["avg_recourse"], eval_wdro["avg_recourse"]],
       "test_infeas": [eval_saa["infeas"], eval_wdro["infeas"]]
   })
   
   print("\n=== SAA vs Approximate W-DRO results ===")
   print(df.to_string(index=False))
