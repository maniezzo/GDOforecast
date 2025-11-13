import numpy as np, pandas as pd
from scipy.optimize import linprog  # Used only for recourse calculation and Lipschitz estimate
from scipy.stats import wasserstein_distance
import pulp
import random, json, time
from pyomo.environ import *

# --- Global Definitions (for helper functions) ---
n = 0
m = 0

def read_instance(filePath, boostSize=75):
   global n, m
   with open(filePath, 'r', encoding='utf-8') as file:
      inst = json.load(file)
   name = inst['name']
   n = inst['n']
   m = inst['m']
   req = inst['req']
   cap = inst['cap']
   qcost = inst['qcost']

   filePath = filePath.replace("inst", "costMatrix")
   filePath = filePath.replace("json", "csv")
   cost = np.transpose(np.loadtxt(filePath, delimiter=','))

   boostReq = np.loadtxt(f"ETSboosts_{boostSize}.csv", delimiter=',')
   boostReq = np.round(boostReq).astype(int)

   return name, n, m, req, cap, qcost, cost, boostReq

def old_solve_saa_pulp(rho_samples, c, d, cap, I, J, eps_penalty=0.0, lip_constant=0.0):
   """
   Solves the SAA (or the objective part of the W-DRO approx).
   The solution 'x' is forced to be Binary.
   """
   S = len(rho_samples)
   prob = pulp.LpProblem("SAA_IP", pulp.LpMinimize)

   # 1. Variables
   x = {}  # Binary variable (First-stage decision)
   q = {}  # Integer variable (Second-stage recourse)

   for i in range(I):
      for j in range(J):
         x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat='Binary')

   for s in range(S):
      for i in range(I):
         for j in range(J):
            q[(s, i, j)] = pulp.LpVariable(f"q_{s}_{i}_{j}", lowBound=0, cat='Integer')

   # 2. Objective function: E[Cost] + Fixed Penalty Term
   # Note: The fixed penalty term (eps * L) is added outside the sum
   # because it does not depend on the decision variable x or q.
   # It ensures the overall reported objective is correct for the W-DRO heuristic.
   obj = pulp.lpSum(c[i, j] * x[(i, j)] for i in range(I) for j in range(J))
   obj += pulp.lpSum((1.0 / S) * d[i] * q[(s, i, j)]
                     for s in range(S) for i in range(I) for j in range(J))
   prob += obj

   # 3. Constraints
   # assignment constraints: sum_i x_ij = 1
   for j in range(J):
      prob += pulp.lpSum([x[(i, j)] for i in range(I)]) == 1, f"Assignment_Client_{j}"

   # per-sample demand balance
   for s in range(S):
      rho_s = rho_samples[s]
      for j in range(J):
         prob += pulp.lpSum([q[(s, i, j)] for i in range(I)]) == int(rho_s[j]), f"Demand_Sample_{s}_Client_{j}"

   # capacity constraints
   for s in range(S):
      for i in range(I):
         prob += pulp.lpSum([q[(s, i, j)] for j in range(J)]) <= int(cap[i]), f"Capacity_Sample_{s}_Server_{i}"

   # linking q_ij^s <= rho_j^s * x_ij
   # Uses M = rho_j^s to create a big-M style constraint: q <= M*x
   for s in range(S):
      rho_s = rho_samples[s]
      for i in range(I):
         for j in range(J):
            # The big-M is simply the demand rho_s[j] (integer)
            prob += q[(s, i, j)] <= int(rho_s[j]) * x[(i, j)], f"Link_Sample_{s}_S{i}_C{j}"

   # 4. Solve
   prob.solve(pulp.PULP_CBC_CMD(msg=0))

   if pulp.LpStatus[prob.status] != 'Optimal':
      return {"status": "fail", "message": pulp.LpStatus[prob.status]}

   # 5. Extract solution and Objective
   x_val = np.zeros((I, J))
   for i in range(I):
      for j in range(J):
         x_val[i, j] = x[(i, j)].varValue

   # Add penalty term for final W-DRO objective calculation
   final_obj = pulp.value(prob.objective) + eps_penalty * lip_constant

   return {"status": "ok", "x": x_val, "obj": final_obj}

def solve_saa_pyomo(rho_samples, c, d, cap, I, J, eps_penalty=0.0, lip_constant=0.0):
   """
   Solves the SAA (or the objective part of the W-DRO approx) using Pyomo/Gurobi.
   The solution 'x' is forced to be Binary.

   Args:
       rho_samples (np.array): Demand scenarios (S x J).
       c (np.array): First-stage costs (I x J).
       d (np.array): Second-stage recourse costs (I).
       cap (list/np.array): Server capacities (I).
       I (int): Number of servers/rows.
       J (int): Number of clients/columns.
       eps_penalty (float): Epsilon (W-DRO radius).
       lip_constant (float): Lipschitz constant L(x_k).

   Returns:
       dict: Status, x_val (I x J array), and objective value.
   """
   S = len(rho_samples)

   # 1. Model Initialization
   model = ConcreteModel()

   # 2. Sets (Indices)
   model.Servers = RangeSet(0, I - 1)
   model.Clients = RangeSet(0, J - 1)
   model.Scenarios = RangeSet(0, S - 1)

   # 3. Variables
   # x: Binary allocation decision (First-stage)
   model.x = Var(model.Servers, model.Clients, domain=Binary)

   # q: Integer recourse quantity (Second-stage)
   model.q = Var(model.Scenarios, model.Servers, model.Clients, domain=NonNegativeIntegers)

   # 4. Objective Function (SAA Cost + Fixed W-DRO Penalty)
   def objective_rule(model):
      # First-stage cost: sum(c_ij * x_ij)
      first_stage_cost = sum(c[i, j] * model.x[i, j] for i in model.Servers for j in model.Clients)

      # Expected second-stage cost: (1/S) * sum(d_i * q_s,i,j)
      second_stage_cost = (1.0 / S) * sum(d[i] * model.q[s, i, j]
                                          for s in model.Scenarios
                                          for i in model.Servers
                                          for j in model.Clients)

      # W-DRO Penalty Term (fixed value added to the objective)
      fixed_penalty = eps_penalty * lip_constant

      return first_stage_cost + second_stage_cost + fixed_penalty

   model.objective = Objective(rule=objective_rule, sense=minimize)

   # 5. Constraints

   # a. Assignment constraints: sum_i x_ij = 1 (Each client is assigned to one server)
   def assignment_rule(model, j):
      return sum(model.x[i, j] for i in model.Servers) == 1

   model.Assignment = Constraint(model.Clients, rule=assignment_rule)

   # b. Demand balance: sum_i q_s,i,j = rho_s,j (Demand must be met for each scenario)
   def demand_rule(model, s, j):
      return sum(model.q[s, i, j] for i in model.Servers) == int(rho_samples[s, j])

   model.Demand = Constraint(model.Scenarios, model.Clients, rule=demand_rule)

   # c. Capacity constraints: sum_j q_s,i,j <= cap_i (Total allocated quantity cannot exceed capacity)
   def capacity_rule(model, s, i):
      return sum(model.q[s, i, j] for j in model.Clients) <= int(cap[i])

   model.Capacity = Constraint(model.Scenarios, model.Servers, rule=capacity_rule)

   # d. Linking constraints: q_s,i,j <= rho_s,j * x_i,j
   # (Recourse only possible if client is assigned to server, using rho as Big-M)
   def linking_rule(model, s, i, j):
      # rho_samples[s, j] is the Big-M here.
      return model.q[s, i, j] <= int(rho_samples[s, j]) * model.x[i, j]

   model.Linking = Constraint(model.Scenarios, model.Servers, model.Clients, rule=linking_rule)

   model.write("model.lp", io_options={'symbolic_solver_labels':True})

   # 6. Solve the Model
   solver = SolverFactory('gurobi')
   results = solver.solve(model, tee=False)  # tee=True shows solver output

   # 7. Extract Solution
   if results.solver.termination_condition != TerminationCondition.optimal:
      # Handle cases where Gurobi fails or finds no optimal solution
      return {"status": "fail", "message": str(results.solver.termination_condition)}

   x_val = np.zeros((I, J))
   for i in range(I):
      for j in range(J):
         # Pyomo's value() function is used to get the solution value
         x_val[i, j] = value(model.x[i, j])

   final_obj = value(model.objective)

   # The objective value returned by Pyomo *includes* the fixed penalty term
   return {"status": "ok", "x": x_val, "obj": final_obj}

# compute_recourse_linprog is fine as it solves a continuous LP for a fixed x.
def old_compute_recourse_linprog(x_val, rho_s, d, Q, I, J):
   # This function is unchanged and correctly uses linprog for the second stage.
   Nq = I * J
   c_q = np.array([d[i] for i in range(I) for j in range(J)])
   A_eq = np.zeros((J, Nq));
   b_eq = np.zeros(J)
   for j in range(J):
      for i in range(I):
         A_eq[j, i * J + j] = 1.0
      b_eq[j] = float(rho_s[j])
   A_ub = np.zeros((I, Nq));
   b_ub = np.zeros(I)
   for i in range(I):
      for j in range(J):
         A_ub[i, i * J + j] = 1.0
      b_ub[i] = float(Q[i])
   bounds = []
   for i in range(I):
      for j in range(J):
         # x_val is binary/integer here, rho_s is integer. Bound is integer.
         bounds.append((0.0, float(rho_s[j] * x_val[i, j])))
   res = linprog(c=c_q, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
   if not res.success:
      return 1e6, None
   qsol = res.x.reshape((I, J))
   return float(res.fun), qsol

def compute_recourse_pyomo(x_val, rho_s, d, cap, I, J):
   """
   Computes the minimum recourse cost for a fixed first-stage decision x_val
   and a specific demand scenario rho_s, using Pyomo.

   Args:
       x_val (np.array): Fixed first-stage allocation (I x J).
       rho_s (np.array): Single demand scenario (J).
       d (np.array): Second-stage recourse costs (I).
       cap (list/np.array): Server capacities (I).
       I (int): Number of servers/rows.
       J (int): Number of clients/columns.

   Returns:
       tuple: (Minimum recourse cost (float), Recourse solution q (np.array or None))
   """
   # 1. Model Initialization
   model = ConcreteModel()

   # 2. Sets (Indices)
   model.Servers = RangeSet(0, I - 1)
   model.Clients = RangeSet(0, J - 1)

   # 3. Variables
   # q: Recourse quantity (Continuous LP variable)
   model.q = Var(model.Servers, model.Clients, domain=NonNegativeReals)

   # 4. Objective Function (Minimize Recourse Cost)
   def objective_rule(model):
      return sum(d[i] * model.q[i, j] for i in model.Servers for j in model.Clients)

   model.objective = Objective(rule=objective_rule, sense=minimize)

   # 5. Constraints

   # a. Demand balance: sum_i q_i,j = rho_s,j (Demand must be met)
   def demand_rule(model, j):
      # rho_s must be integer here for consistency with the MIP problem
      return sum(model.q[i, j] for i in model.Servers) == int(rho_s[j])

   model.Demand = Constraint(model.Clients, rule=demand_rule)

   # b. Capacity constraints: sum_j q_i,j <= cap_i (Total allocated quantity cannot exceed capacity)
   def capacity_rule(model, i):
      return sum(model.q[i, j] for j in model.Clients) <= int(cap[i])

   model.Capacity = Constraint(model.Servers, rule=capacity_rule)

   # c. Allocation/Linking constraints: q_i,j <= rho_s,j * x_val_i,j
   # Recourse is only possible if the client is assigned to the server in x_val.
   def linking_rule(model, i, j):
      # The upper bound is the maximum amount server 'i' can supply to client 'j'
      # given the fixed x_val and the demand rho_s.
      upper_bound = float(rho_s[j] * x_val[i, j])
      return model.q[i, j] <= upper_bound

   model.Linking = Constraint(model.Servers, model.Clients, rule=linking_rule)

   # 6. Solve the Model
   # Use 'gurobi' or 'glpk'/'cbc' since it's an LP. 'glpk' is often readily available.
   solver = SolverFactory('gurobi')
   results = solver.solve(model, tee=False)

   # 7. Extract Solution
   if results.solver.termination_condition != TerminationCondition.optimal:
      # Return a large penalty cost for infeasible solutions
      return 1e6, None

   recourse_cost = value(model.objective)

   # Extract q solution (optional, but helpful for debugging/analysis)
   q_sol = np.zeros((I, J))
   for i in model.Servers:
      for j in model.Clients:
         q_sol[i, j] = value(model.q[i, j])

   return recourse_cost, q_sol

# estimate_lipschitz_numeric is a heuristic.
def estimate_lipschitz_numeric(x_val, rho_samples, d, Q, I, J, delta=1e-3):
   diffs = []
   for s in range(len(rho_samples)):
      rho_s = rho_samples[s]
      base, _ = compute_recourse_pyomo(x_val, rho_s, d, Q, I, J)
      if base > 1e5: continue
      for j in range(J):
         rho_pert = rho_s.copy();
         # Note: Perturbation might make rho_pert[j] non-integer. This is allowed
         # for Lipschitz estimate if the recourse function is Lipschitz on R^n.
         rho_pert[j] += delta
         pert, _ = compute_recourse_pyomo(x_val, rho_pert, d, Q, I, J)
         if pert > 1e5: continue
         diffs.append(abs(pert - base) / delta)
   if len(diffs) == 0: return 1e6
   return max(diffs)

def solve_approx_wdro_pulp(rho_samples, eps, c, d, Q, I, J, n_iter=6):
   """
   Alternating scheme using solve_saa_pulp
   """
   # 1. Initialize with SAA solution (eps=0, lip=0)
   res = solve_saa_pyomo(rho_samples, c, d, Q, I, J, eps_penalty=0.0, lip_constant=0.0)
   if res["status"] != "ok":
      raise RuntimeError("Initial SAA failed in W-DRO solver")

   x_cur = res["x"]
   lip_cur = estimate_lipschitz_numeric(x_cur, rho_samples, d, Q, I, J)
   history = []

   # 2. Alternating Optimization
   for it in range(n_iter):
      # Step A: Solve for x using the current fixed Lipschitz constant (lip_cur)
      res = solve_saa_pyomo(rho_samples, c, d, Q, I, J, eps_penalty=eps, lip_constant=lip_cur)
      if res["status"] != "ok":
         print(f"W-DRO iteration {it} failed: {res['message']}")
         break

      x_new = res["x"]

      # Step B: Estimate the new Lipschitz constant based on x_new
      lip_new = estimate_lipschitz_numeric(x_new, rho_samples, d, Q, I, J)

      history.append({"it": it, "obj": res["obj"], "lip": lip_new})

      # Check convergence: tolerance check on x and L
      x_diff = np.linalg.norm(x_new - x_cur)
      lip_diff = abs(lip_new - lip_cur)

      if x_diff < 1e-4 and lip_diff < 1e-2:
         print(f"W-DRO converged at iteration {it}.")
         x_cur, lip_cur = x_new, lip_new;
         break

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
   eps = np.percentile(distances, percentile * 100)
   return eps, distances


def evaluate_solution(x_val, rho_test):
   global m, n, qcost, cap
   Ls = []
   infeas = 0
   for s in range(len(rho_test)):
      rho_s = np.round(rho_test[s]).astype(int)
      val, _ = compute_recourse_pyomo(x_val, rho_s, qcost, cap, m, n)
      if val > 1e5:
         infeas += 1
      else:
         Ls.append(val)
   return {"avg_recourse": np.mean(Ls) if len(Ls) > 0 else np.nan, "infeas": infeas}


if __name__ == "__main__":
   np.random.seed(995)
   random.seed(995)

   tStart = time.process_time()

   instance = "inst_52_4_0_0.json" # "inst_8_2_0_0.json" "inst_52_4_0_0.json"
   datiVeri = "datiVeri.csv" # "datiVeriTest.csv"  "datiVeri.csv"
   boostSet = 75 # "15_test"  75
   # 1. Load and Prepare Data (rho is integer)
   name, n, m, req, cap, qcost, cost, boostReq = read_instance(instance, boostSet)
   rho_samples = boostReq

   # Ensure test data is also integer for evaluation consistency
   rho_test_raw = np.loadtxt(datiVeri, delimiter=',')
   rho_test = np.round(rho_test_raw).astype(int)
   # Check if the array is 1D
   if rho_test.ndim == 1:
      # Reshape to 2D with one row
      rho_test = rho_test.reshape(1, -1)
   n_test   = len(rho_test)

   n_train = len(rho_samples)
   n_test = len(rho_test)

   # 2. Solve SAA Baseline (Now a proper MIP)
   print("=== Solving SAA MIP Baseline ===")
   saa_res = solve_saa_pyomo(rho_samples, cost, qcost, cap, m, n)
   if saa_res["status"] != "ok":
      raise RuntimeError(f"SAA failed: {saa_res['message']}")
   x_saa = saa_res["x"]

   # 3. Calibrate Epsilon
   print("\n=== Calibrating Wasserstein Radius ===")
   eps_calibrated, boot_dists = calibrate_epsilon_bootstrap(rho_samples, B=200, percentile=0.9, mode="max")
   #eps_calibrated = 20000.0
   print(f"Calibrated epsilon (90th pct of max 1D distances): {eps_calibrated:.4f}")

   # 4. Solve Approximate W-DRO (Alternating MIP/LP)
   print("\n=== Solving Approximate W-DRO ===")
   wdro_res = solve_approx_wdro_pulp(rho_samples, eps=eps_calibrated, c=cost, d=qcost, Q=cap, I=m, J=n, n_iter=6)
   x_wdro = wdro_res["x"]

   print("\nSAA x (rounded):", np.round(x_saa, 0).flatten())
   print("W-DRO x (rounded):", np.round(x_wdro, 0).flatten())

   # 5. Evaluate Solutions
   print("\n=== Evaluating Solutions on Test Set ===")
   eval_saa = evaluate_solution(x_saa, rho_test)
   eval_wdro = evaluate_solution(x_wdro, rho_test)

   tEnd = time.process_time()
   cpu_time = tEnd - tStart

   # 6. Summary Output
   df = pd.DataFrame({
      "method": ["SAA", "W-DRO (calibrated)"],
      "first_stage_cost": [np.sum(cost * x_saa), np.sum(cost * x_wdro)],
      "final_robust_obj": [saa_res["obj"], wdro_res["final_obj"]],  # Note: SAA obj is approx
      "approx_lip": [estimate_lipschitz_numeric(x_saa, rho_samples, qcost, cap, m, n), wdro_res["lip"]],
      "test_avg_recourse": [eval_saa["avg_recourse"], eval_wdro["avg_recourse"]],
      "test_infeas": [eval_saa["infeas"], eval_wdro["infeas"]]
   })

   print("\n=== Final Results ===")
   print(df.to_string(index=False, float_format="%.3f"))
   print(f"\nCPU time used: {cpu_time:.4f} seconds")