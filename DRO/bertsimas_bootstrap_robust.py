import numpy as np
from scipy.optimize import linprog

"""
bertsimas_bootstrap_robust.py

Toy implementation illustrating a simple Bertsimas-style bootstrap-robust
adjustment for the two-stage stochastic assignment/flow problem.

This script:
 - builds a small random problem (continuous relaxations for x and q)
 - solves the SAA LP exactly (using scipy.linprog)
 - evaluates per-sample recourse costs L_s(x)
 - computes a simple bootstrap-robust adjusted objective:
       robust_obj = mean(L_s) + epsilon * max(abs(L_s - mean))
   which approximates the effect of worst-case reweighting within an l1-ball.
 - prints results and a small summary table

Notes:
 - This is a didactic prototype. A full implementation of the Bertsimas entropic
   approach with exact KL-DRO reformulation would require solving a mixed-integer
   convex program (or using the entropic dual) and bootstrap calibration loops.
 - The present code keeps things LP-friendly by relaxing integer variables.
"""

np.random.seed(123)

# Problem dimensions
I, J = 2, 3
n_samples = 40
n_test = 200

# Random costs and capacities (toy)
c = np.abs(np.random.randn(I, J)) * 5.0
d = np.abs(np.random.randn(I)) * 3.0
Q = np.array([30.0, 30.0])

# Generate random rho samples (positive demands)
rho_mean = np.array([8.0, 10.0, 6.0])
rho_scale = np.array([2.0, 3.0, 1.5])
rho_samples = np.maximum(0.1, np.random.randn(n_samples, J) * rho_scale + rho_mean)
rho_test = np.maximum(0.1, np.random.randn(n_test, J) * rho_scale + rho_mean)

# Helper indices
def idx_x(i,j): return i*J + j
def idx_q(s,i,j,n_x,n_q_block): return n_x + s*n_q_block + i*J + j

# Solve SAA LP (continuous x in [0,1], continuous q)
def solve_saa(rho_samples, c, d, Q, I, J):
   S = len(rho_samples)
   n_x = I*J
   n_q_block = I*J
   Nvars = n_x + S*n_q_block

   # objective vector
   cvec = np.zeros(Nvars)
   for i in range(I):
      for j in range(J):
         cvec[idx_x(i,j)] = c[i,j]
   for s in range(S):
      for i in range(I):
         for j in range(J):
            cvec[idx_q(s,i,j,n_x,n_q_block)] = (1.0/S) * d[i]

   # constraints
   A_eq = []
   b_eq = []
   A_ub = []
   b_ub = []

   # assignment sum_i x_ij = 1
   for j in range(J):
      row = np.zeros(Nvars)
      for i in range(I):
         row[idx_x(i,j)] = 1.0
      A_eq.append(row); b_eq.append(1.0)

   # per-sample demand balance: sum_i q_ij^s = rho_j^s
   for s in range(S):
      rho_s = rho_samples[s]
      for j in range(J):
         row = np.zeros(Nvars)
         for i in range(I):
            row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
         A_eq.append(row); b_eq.append(float(rho_s[j]))

   # capacity per i: sum_j q_ij^s <= Q_i
   for s in range(S):
      for i in range(I):
         row = np.zeros(Nvars)
         for j in range(J):
            row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
         A_ub.append(row); b_ub.append(float(Q[i]))

   # linking q_ij^s <= rho_j^s * x_ij
   for s in range(S):
      rho_s = rho_samples[s]
      for i in range(I):
         for j in range(J):
            row = np.zeros(Nvars)
            row[idx_q(s,i,j,n_x,n_q_block)] = 1.0
            row[idx_x(i,j)] = -float(rho_s[j])
            A_ub.append(row); b_ub.append(0.0)

   # bounds
   bounds = []
   for v in range(Nvars):
      if v < n_x:
         bounds.append((0.0, 1.0))
      else:
         bounds.append((0.0, None))

   # solve LP
   res = linprog(c=cvec, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                 A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method='highs')
   if not res.success:
      raise RuntimeError("SAA LP failed: " + str(res.message))
   sol = res.x
   x_val = sol[:n_x].reshape((I,J))
   q_val = sol[n_x:].reshape((S, I, J))
   obj = res.fun
   return {"x": x_val, "q": q_val, "obj": obj}

# compute recourse L(x, rho_s) for fixed x and a single sample
def compute_recourse(x_val, rho_s, d, Q, I, J):
   Nq = I*J
   c_q = np.array([d[i] for i in range(I) for j in range(J)])

   # equality: sum_i q_ij = rho_j
   A_eq = np.zeros((J, Nq)); b_eq = np.zeros(J)
   for j in range(J):
      for i in range(I):
         A_eq[j, i*J + j] = 1.0
      b_eq[j] = float(rho_s[j])

   # capacity: sum_j q_ij <= Q_i
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

# Main execution
if __name__ == "__main__":
   saa = solve_saa(rho_samples, c, d, Q, I, J)
   x_saa = saa["x"]
   obj_saa = saa["obj"]

   # evaluate recourse per sample
   Ls = []
   for s in range(len(rho_samples)):
      val, _ = compute_recourse(x_saa, rho_samples[s], d, Q, I, J)
      Ls.append(val)
   Ls = np.array(Ls)
   mean_L = Ls.mean()

   # bootstrap-robust adjustment (heuristic): mean + epsilon * max deviation
   epsilon = 0.1
   robust_obj = mean_L + epsilon * np.max(np.abs(Ls - mean_L))

   print("\\n=== Bertsimas-style bootstrap-robust prototype ===")
   print("SAA objective (train):", obj_saa)
   print("Average recourse L mean:", mean_L)
   print("Max deviation:", np.max(np.abs(Ls - mean_L)))
   print("Robust adjusted objective (heuristic):", robust_obj)
   print("\\nSAA x (rounded):\\n", np.round(x_saa, 3))
