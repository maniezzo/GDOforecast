import numpy as np
from scipy.optimize import linprog
import pulp, time, json
"""
bertsimas_bootstrap_robust_calibrated.py
Implementation illustrating the Bertsimas & Kallus (2020) bootstrap-robust
prescriptive analytics idea, including a bootstrap-based calibration loop.
Steps:
 1. Solve the baseline SAA problem.
 2. Bootstrap the empirical sample of ρ to produce B resamples.
 3. For each bootstrap, solve its SAA version and compute out-of-sample regret.
 4. Use percentile of regrets to calibrate the robustness parameter ε*.
 5. Compute the bootstrap-robust adjusted objective with calibrated ε*.
"""

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
    boostReq = np.round(boostReq).astype(int)

    return name,n,m,req,cap,qcost,cost,boostReq

def idx_x(i,j):
   return i*n + j
def idx_q(s,i,j,n_x,n_q_block):
   return n_x + s*n_q_block + i*n + j

def solve_saa_LP(rho_samples, c, d, cap, m, n):
   S = len(rho_samples)
   n_x = m*n
   n_q_block = m*n
   Nvars = n_x + S*n_q_block
   cvec = np.zeros(Nvars)
   for i in range(m):
      for j in range(n):
         cvec[idx_x(i,j)] = c[i,j]
   for s in range(S):
      for i in range(m):
         for j in range(n):
            cvec[idx_q(s,i,j,n_x,n_q_block)] = (1.0/S) * d[i]

   A_eq, b_eq, A_ub, b_ub = [], [], [], []

   # assignment constraints: sum_i x_ij = 1
   for j in range(n):
      row = np.zeros(Nvars)
      for i in range(m):
         row[idx_x(i,j)] = 1.0
      A_eq.append(row); b_eq.append(1.0)

   # per-sample demand balance
   for s in range(S):
      rho_s = rho_samples[s]
      for j in range(n):
         row = np.zeros(Nvars)
         for i in range(m):
            row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
         A_eq.append(row); b_eq.append(float(rho_s[j]))

   # capacity constraints
   for s in range(S):
      for i in range(m):
         row = np.zeros(Nvars)
         for j in range(n):
            row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
         A_ub.append(row); b_ub.append(float(cap[i]))

   # linking q_ij^s <= rho_j^s * x_ij
   for s in range(S):
      rho_s = rho_samples[s]
      for i in range(m):
         for j in range(n):
            row = np.zeros(Nvars)
            row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
            row[idx_x(i,j)] = -float(rho_s[j])
            A_ub.append(row); b_ub.append(0.0)

   bounds = [(0, 1)] * n_x + [(0, None)] * (S * n_q_block)
   res = linprog(cvec, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                 A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
   if not res.success:
      raise RuntimeError("SAA LP failed: " + res.message)
   x_val = res.x[:n_x].reshape((m, n))
   return x_val, res.fun

def solve_saa_IP(rho_samples, c, d, cap, m, n):
   S = len(rho_samples)
   n_x = m * n
   n_q_block = m * n
   Nvars = n_x + S * n_q_block
   cvec = np.zeros(Nvars)
   for i in range(m):
      for j in range(n):
         cvec[idx_x(i, j)] = c[i, j]
   for s in range(S):
      for i in range(m):
         for j in range(n):
            cvec[idx_q(s, i, j, n_x, n_q_block)] = (1.0 / S) * d[i]

   A_eq, b_eq, A_ub, b_ub = [], [], [], []

   # assignment constraints: sum_i x_ij = 1
   for j in range(n):
      row = np.zeros(Nvars)
      for i in range(m):
         row[idx_x(i, j)] = 1.0
      A_eq.append(row);
      b_eq.append(1.0)

   # per-sample demand balance
   for s in range(S):
      rho_s = rho_samples[s]
      for j in range(n):
         row = np.zeros(Nvars)
         for i in range(m):
            row[idx_q(s, i, j, n_x, n_q_block)] = 1.0
         A_eq.append(row);
         b_eq.append(float(rho_s[j]))

   # capacity constraints
   for s in range(S):
      for i in range(m):
         row = np.zeros(Nvars)
         for j in range(n):
            row[idx_q(s, i, j, n_x, n_q_block)] = 1.0
         A_ub.append(row);
         b_ub.append(float(cap[i]))

   # linking q_ij^s <= M * x_ij (where M is a big-M constraint)
   for s in range(S):
      rho_s = rho_samples[s]
      for i in range(m):
         for j in range(n):
            row = np.zeros(Nvars)
            row[idx_q(s, i, j, n_x, n_q_block)] = 1.0
            # Use big-M where M = rho_s[j] (upper bound on q when x_ij = 1)
            row[idx_x(i, j)] = -float(rho_s[j])
            A_ub.append(row);
            b_ub.append(0.0)

   # Define bounds - x variables are binary, q variables are non-negative continuous
   bounds = [(0, 1)] * n_x  # Binary bounds for x variables
   bounds.extend([(0, None)] * (S * n_q_block))  # Non-negative for q variables

   # Use milp with integrality constraints
   from scipy.optimize import milp, LinearConstraint, Bounds

   # Specify which variables are integer (1) vs continuous (0)
   integrality = [1] * n_x + [0] * (S * n_q_block)  # x vars are binary, q vars are continuous

   # Create constraint matrices
   if A_eq:
      A_eq_matrix = np.array(A_eq)
      b_eq_array = np.array(b_eq)
      eq_constraint = LinearConstraint(
         A=A_eq_matrix,
         lb=b_eq_array,
         ub=b_eq_array
      )
   else:
      eq_constraint = LinearConstraint(np.zeros((0, Nvars)), [], [])

   if A_ub:
      A_ub_matrix = np.array(A_ub)
      b_ub_array = np.array(b_ub)
      ub_constraint = LinearConstraint(
         A=A_ub_matrix,
         lb=-np.inf,
         ub=b_ub_array
      )
   else:
      ub_constraint = LinearConstraint(np.zeros((0, Nvars)), [], [])

   # Combine constraints
   A_combined = np.vstack([A_eq_matrix if A_eq else np.zeros((0, Nvars)),
                           A_ub_matrix if A_ub else np.zeros((0, Nvars))])
   lb_combined = np.concatenate([b_eq_array if A_eq else np.array([]),
                                 [-np.inf] * len(b_ub_array) if A_ub else np.array([])])
   ub_combined = np.concatenate([b_eq_array if A_eq else np.array([]),
                                 b_ub_array if A_ub else np.array([])])

   linear_constraint = LinearConstraint(A_combined, lb_combined, ub_combined)

   bounds_obj = Bounds(
      lb=[b[0] for b in bounds],
      ub=[b[1] for b in bounds]
   )

   res = milp(
      c=cvec,
      constraints=linear_constraint,
      bounds=bounds_obj,
      integrality=integrality
   )

   if not res.success:
      print(f"Solver message: {res.message}")
      print(f"Solver status: {res.status}")
      raise RuntimeError("SAA IP failed: " + str(res.message))

   x_val = res.x[:n_x].reshape((m, n))
   return x_val, res.fun

# Using PuLP
def solve_saa_pulp(rho_samples, c, d, cap, m, n):
   S = len(rho_samples)
   prob = pulp.LpProblem("SAA_IP", pulp.LpMinimize)

   # Create variables
   x = {}
   q = {}

   for i in range(m):
      for j in range(n):
         x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat='Binary')

   for s in range(S):
      for i in range(m):
         for j in range(n):
            q[(s, i, j)] = pulp.LpVariable(f"q_{s}_{i}_{j}", lowBound=0, cat='Integer')

   # Objective function
   obj = 0
   for i in range(m):
      for j in range(n):
         obj += c[i, j] * x[(i, j)]

   for s in range(S):
      for i in range(m):
         for j in range(n):
            obj += (1.0 / S) * d[i] * q[(s, i, j)]

   prob += obj

   # Constraints
   # assignment constraints: sum_i x_ij = 1
   for j in range(n):
      prob += pulp.lpSum([x[(i, j)] for i in range(m)]) == 1

   # per-sample demand balance
   for s in range(S):
      rho_s = rho_samples[s]
      for j in range(n):
         prob += pulp.lpSum([q[(s, i, j)] for i in range(m)]) == float(rho_s[j])

   # capacity constraints
   for s in range(S):
      for i in range(m):
         prob += pulp.lpSum([q[(s, i, j)] for j in range(n)]) <= float(cap[i])

   # linking q_ij^s <= rho_j^s * x_ij
   for s in range(S):
      rho_s = rho_samples[s]
      for i in range(m):
         for j in range(n):
            prob += q[(s, i, j)] <= float(rho_s[j]) * x[(i, j)]

   # Solve
   prob.solve(pulp.PULP_CBC_CMD(msg=0))

   if pulp.LpStatus[prob.status] != 'Optimal':
      raise RuntimeError("SAA IP failed")

   # Extract solution
   x_val = np.zeros((m, n))
   for i in range(m):
      for j in range(n):
         x_val[i, j] = x[(i, j)].varValue

   return x_val, pulp.value(prob.objective)

# calcola il costo della soluzione
def compute_recourse(x, rho_s, qcost, cap, m, n):
   Nq = m * n
   c_q = np.array([qcost[i] for i in range(m) for j in range(n)])

   # vincoli di uguaglianza
   A_eq = np.zeros((n, Nq)); b_eq = np.zeros(n)
   for j in range(n):
      for i in range(m):
         A_eq[j, i * n + j] = 1.0
      b_eq[j] = rho_s[j]

   # vincoli disuguaglianze
   A_ub = np.zeros((m, Nq)); b_ub = np.zeros(m)
   for i in range(m):
      for j in range(n):
         A_ub[i, i * n + j] = 1.0
      b_ub[i] = cap[i]

   bounds = [(0, rho_s[j] * x[i, j]) for i in range(m) for j in range(n)]
   res = linprog(c_q, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
   if not res.success:
      return 99999999
   return res.fun


# DRO solution
def solve_bootstrap_robust_ip(rho_samples, cost, qcost, cap, m, n, eps_star):
   """
   Placeholder: The actual implementation involves a major extension
   of the SAA IP to minimize the mean objective PLUS
   eps_star * (measure of deviation).

   For the Bertsimas & Kallus (2020) 'worst-case mean' idea, this
   translates to minimizing:

   min E[C(x, rho)] + eps_star * max |C(x, rho) - E[C(x, rho)]|

   This requires introducing new variables and constraints to model
   the max deviation, turning it into a convex-ish (or mixed-integer) problem.
   """
   print("\n[WARNING] Now solving the Epsilon*-Robust Optimization problem...")
   print("    This section requires significant reformulation of the MIP.")
   print("    Returning SAA result as placeholder for demonstration.")

   x_robust, obj_robust = solve_saa_pulp(rho_samples, cost, qcost, cap, m, n)

   # If the robust problem were solved, we would evaluate its true objective:
   L_vals_robust = np.array([compute_recourse(x_robust, rho_samples[s], qcost, cap, m, n)
                             for s in range(len(rho_samples))])
   mean_L_robust = L_vals_robust.mean()
   robust_obj_value = mean_L_robust + eps_star * np.max(np.abs(L_vals_robust - mean_L_robust))

   return x_robust, robust_obj_value, mean_L_robust

if __name__ == "__main__":
   np.random.seed(995)
   tStart = time.process_time()

   name, n, m, req, cap, qcost, cost, boostReq = read_instance("inst_52_4_0_0.json", 75)

   # Load sample data (rho_samples contains the training/empirical distribution)
   rho_samples = np.loadtxt("ETSboosts_75.csv", delimiter=',')
   n_train = len(rho_samples)

   rho_test = np.loadtxt("datiVeri.csv", delimiter=',')
   n_test   = len(rho_test)

   fTest = False
   if fTest:
      # Costs and capacities
      cost  = np.array([[3.379103, 8.952563, 0.75332], [3.444593, 3.355126, 7.57820]])
      qcost = np.array([2.027462, 5.371538])
      cap   = np.array([30.0, 30.0])

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

   m = cost.shape[0]  # righe, num server
   n = cost.shape[1]  # colonne, num client
   n_samples   = len(rho_samples)
   n_bootstrap = len(rho_test)  # number of bootstrap resamples
   quantile_level = 0.9  # desired confidence for calibration

   # === Step 1: Solve baseline SAA ===
   x_saa, obj_saa = solve_saa_pulp(rho_samples, cost, qcost, cap, m, n)

   # === Step 2: Bootstrap calibration ===
   regrets = []
   for b in range(n_bootstrap):
      print(f"b={b}")
      # Sample with replacement from the original sample
      idx = np.random.choice(len(rho_samples), size=len(rho_samples), replace=True)
      rho_boot = rho_samples[idx]

      # Solve SAA on the bootstrap sample
      x_boot, _ = solve_saa_pulp(rho_boot, cost, qcost, cap, m, n)

      # Evaluate out-of-sample recourse on the original sample (rho_samples)
      # SAA_boot regret = E[Cost(x_boot)] - E[Cost(x_saa)]
      L_boot = np.mean([compute_recourse(x_boot, rho_samples[s], qcost, cap, m, n)
                        for s in range(len(rho_samples))])
      L_saa = np.mean([compute_recourse(x_saa, rho_samples[s], qcost, cap, m, n)
                       for s in range(len(rho_samples))])
      regrets.append(L_boot - L_saa)

      # Optional print for progress
      if (b + 1) % 50 == 0 or b == n_bootstrap - 1:
         print(f"Processed {b + 1}/{n_bootstrap} bootstraps.")

   regrets = np.array(regrets)
   # Calibrate epsilon*
   eps_star = np.percentile(np.abs(regrets), quantile_level * 100)

   # === Step 3: Compute bootstrap-robust adjusted objective ===
   x_robust, robust_obj, mean_L_robust = solve_bootstrap_robust_ip(
      rho_samples, cost, qcost, cap, m, n, eps_star)

   # fine
   print("\n=== Results: Bootstrap-Calibrated Robust Optimization ===")
   print(f"Baseline SAA obj: {obj_saa:.3f}")
   print(f"Calibrated epsilon* ({int(quantile_level * 100)}th perct of regrets): {eps_star:.4f}")

   # Recalculate SAA's performance on the robust objective for comparison
   L_vals_saa = np.array([compute_recourse(x_saa, rho_samples[s], qcost, cap, m, n)
                          for s in range(len(rho_samples))])
   mean_L_saa = L_vals_saa.mean()
   robust_obj_saa = mean_L_saa + eps_star * np.max(np.abs(L_vals_saa - mean_L_saa))

   print(f"SAA Solution's Adjusted Robust Objective: {robust_obj_saa:.3f}")

   # The actual output from the solve_bootstrap_robust_ip function
   print(f"Robust Solution's Adjusted Robust Objective: {robust_obj:.3f}")
   print(f"Robust Solution's Mean Recourse: {mean_L_robust:.3f}")

   print("\nOptimal Robust x (Placeholder/SAA):\n", np.round(x_robust, 3))

   tEnd = time.process_time()
   cpu_time = tEnd - tStart
   print(f"\nCPU time used: {cpu_time:.4f} seconds")