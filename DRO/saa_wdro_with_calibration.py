'''
What it does
- Calibrates a proxy Wasserstein radius ε by bootstrapping the empirical ρ samples and computing coordinate-wise 1-D Wasserstein distances (aggregated by max across coordinates).
The calibrated ε is the chosen percentile (default 90th) of these bootstrap distances.
- Uses that ε in the approximate W-DRO solver (heuristic alternating scheme using finite-difference Lipschitz estimates) to compute a robustified decision.
- Evaluates both SAA and W-DRO decisions on a held-out test set and prints a summary table.
'''
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
import random, json, time

def read_instance(filePath,boostSize=75):
    with open(filePath, 'r', encoding='utf-8') as file:
        inst = json.load(file)
    name = inst['name']
    n    = inst['n']
    m    = inst['m']
    req  = inst['req']
    cap  = inst['cap']
    qcost= inst['qcost']

    filePath = filePath.replace("inst", "costMatrix")
    filePath = filePath.replace("json", "csv")
    cost     = np.transpose(np.loadtxt(filePath, delimiter=','))

    boostReq = np.loadtxt(f"ETSboosts_{boostSize}.csv", delimiter=',')

    return name,n,m,req,cap,qcost,cost,boostReq

# sequenzializzazione indici delle x
def idx_x(i,j):
   return i*n + j
# sequenzializzazione indici delle q
def idx_q(s,i,j,n_x,n_q_block):
   return n_x + s*n_q_block + i*n + j

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
    for j in range(J):
        row = np.zeros(Nvars)
        for i in range(I):
            row[idx_x(i,j)] = 1.0
        A_eq.append(row); b_eq.append(1.0)
    for s in range(S):
        rho_s = rho_samples[s]
        for j in range(J):
            row = np.zeros(Nvars)
            for i in range(I):
                row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
            A_eq.append(row); b_eq.append(float(rho_s[j]))
    for s in range(S):
        for i in range(I):
            row = np.zeros(Nvars)
            for j in range(J):
                row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
            A_ub.append(row); b_ub.append(float(Q[i]))
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
        if v < n_x: bounds.append((0.0, 1.0))
        else: bounds.append((0.0, None))
    res = linprog(c=cvec, A_ub=np.array(A_ub) if len(A_ub)>0 else None, b_ub=np.array(b_ub) if len(b_ub)>0 else None,
                  A_eq=np.array(A_eq) if len(A_eq)>0 else None, b_eq=np.array(b_eq) if len(b_eq)>0 else None,
                  bounds=bounds, method='highs')
    if not res.success:
        return {"status":"fail", "message": res.message}
    sol = res.x
    x_val = sol[:n_x].reshape((I,J))
    q_val = sol[n_x:].reshape((len(rho_samples), I, J))
    obj = res.fun + eps_penalty * lip_constant
    return {"status":"ok", "x": x_val, "q": q_val, "obj": obj}

def compute_recourse_linprog(x_val, rho_s, d, Q, I, J):
    Nq = I*J
    c_q = np.array([d[i] for i in range(I) for j in range(J)])
    A_eq = np.zeros((J, Nq)); b_eq = np.zeros(J)
    for j in range(J):
        for i in range(I):
            A_eq[j, i*J + j] = 1.0
        b_eq[j] = float(rho_s[j])
    A_ub = np.zeros((I, Nq)); b_ub = np.zeros(I)
    for i in range(I):
        for j in range(J):
            A_ub[i, i*J + j] = 1.0
        b_ub[i] = float(Q[i])
    bounds = []
    for i in range(I):
        for j in range(J):
            bounds.append((0.0, float(rho_s[j] * x_val[i,j])))
    res = linprog(c=c_q, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        return 1e6, None
    qsol = res.x.reshape((I,J))
    return float(res.fun), qsol

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

def solve_approx_wdro_linprog(rho_samples, eps, c, d, Q, I, J, n_iter=5):
    res = solve_saa_lp_linprog(rho_samples, c, d, Q, I, J)
    if res["status"] != "ok":
        raise RuntimeError("SAA failed in W-DRO solver")
    x_cur = res["x"]
    lip_cur = estimate_lipschitz_numeric(x_cur, rho_samples, d, Q, I, J)
    history = []
    for it in range(n_iter):
        res = solve_saa_lp_linprog(rho_samples, c, d, Q, I, J, eps_penalty=eps, lip_constant=lip_cur)
        if res["status"] != "ok":
            break
        x_new = res["x"]
        lip_new = estimate_lipschitz_numeric(x_new, rho_samples, d, Q, I, J)
        history.append({"it": it, "obj": res["obj"], "lip": lip_new})
        if np.linalg.norm(x_new - x_cur) < 1e-4 and abs(lip_new - lip_cur) < 1e-2:
            x_cur, lip_cur = x_new, lip_new; break
        x_cur, lip_cur = x_new, lip_new
    return {"x": x_cur, "lip": lip_cur, "history": history, "final_obj": res.get("obj", None)}

def multivariate_wasserstein_proxy(X, Y, mode="max"):
    dists = []
    for j in range(X.shape[1]):
        d = wasserstein_distance(X[:, j], Y[:, j])
        dists.append(d)
    if mode == "max":
        return max(dists)
    elif mode == "l1":
        return sum(dists)
    else:
        return max(dists)

def calibrate_epsilon_bootstrap(rho_samples, B=200, percentile=0.9, mode="max"):
    N = len(rho_samples)
    distances = []
    for b in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        rho_boot = rho_samples[idx]
        w = multivariate_wasserstein_proxy(rho_boot, rho_samples, mode=mode)
        distances.append(w)
    distances = np.array(distances)
    eps = np.percentile(distances, percentile*100)
    return eps, distances

def evaluate_solution(x_val, rho_test):
    Ls = []
    infeas = 0
    for s in range(len(rho_test)):
        val, _ = compute_recourse_linprog(x_val, rho_test[s], qcost, cap, m, n)
        if val > 1e5:
            infeas += 1
        else:
            Ls.append(val)
    return {"avg_recourse": np.mean(Ls) if len(Ls)>0 else np.nan, "infeas": infeas}

if __name__ == "__main__":
   np.random.seed(995)
   random.seed(995)

   tStart = time.process_time()

   name,n,m,req,cap,qcost,cost,boostReq = read_instance("inst_52_4_0_0.json",75)
   rho_samples = np.loadtxt("ETSboosts_75.csv", delimiter=',')
   n_train = len(rho_samples)

   rho_test = np.loadtxt("datiVeri.csv", delimiter=',')
   n_test = len(rho_test)

   fTest = False
   if fTest:
      # Costs and capacities
      cost  = np.array([[3.379103, 8.952563, 0.75332], [3.444593, 3.355126, 7.57820]])
      qcost = np.array([2.027462, 5.371538])
      cap   = np.array([100.0, 100.0])
      m = cost.shape[0] # righe, num server
      n = cost.shape[1] # colonne, num client
      
      rho_samples = np.loadtxt('rho_samples.csv', delimiter=',')
      rho_test    = np.loadtxt('rho_test.csv', delimiter=',')
      n_train = len(rho_samples)
      n_test  = len(rho_test)
      '''
      rho_mean    = np.array([8.0, 10.0, 6.0])
      rho_scale   = np.array([2.0, 3.0, 1.5])
      rho_samples = np.maximum(0.1, np.random.randn(n_train, n) * rho_scale + rho_mean)
      rho_test    = np.maximum(0.1, np.random.randn(n_test, n) * rho_scale + rho_mean)
      '''
   
   saa_res = solve_saa_lp_linprog(rho_samples, cost, qcost, cap, m, n)
   if saa_res["status"] != "ok":
      raise RuntimeError("SAA failed")
   x_saa = saa_res["x"]
   
   eps_calibrated, boot_dists = calibrate_epsilon_bootstrap(rho_samples, B=200, percentile=0.9, mode="max")
   print("Calibrated epsilon (90th pct of bootstrap proxy distances):", eps_calibrated)
   wdro_res = solve_approx_wdro_linprog(rho_samples, eps=eps_calibrated, c=cost, d=qcost, Q=cap, I=m, J=n, n_iter=6)
   x_wdro = wdro_res["x"]
   print("SAA x (rounded):\\n", np.round(x_saa,3))
   print("W-DRO x (rounded):\\n", np.round(x_wdro,3))
   
   eval_saa  = evaluate_solution(x_saa, rho_test)
   eval_wdro = evaluate_solution(x_wdro, rho_test)

   tEnd = time.process_time()
   cpu_time = tEnd - tStart

   df = pd.DataFrame({
      "method": ["SAA", "W-DRO (calibrated)"],
      "first_stage_cost": [np.sum(cost * x_saa), np.sum(cost * x_wdro)],
      "approx_lip": [estimate_lipschitz_numeric(x_saa, rho_samples, qcost, cap, m, n), wdro_res["lip"]],
      "test_avg_recourse": [eval_saa["avg_recourse"], eval_wdro["avg_recourse"]],
      "test_infeas": [eval_saa["infeas"], eval_wdro["infeas"]]
   })
   print("\\n=== Results ===")
   print(df.to_string(index=False))
   print(f"CPU time used: {cpu_time:.4f} seconds")
