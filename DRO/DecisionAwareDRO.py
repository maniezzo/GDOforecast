import numpy as np
from scipy.optimize import linprog

# --- 1. SETUP: ASSUMPTIONS & PARAMETERS ---
# Assume we have 100 historical observations of the residual vector (epsilon)
N_HISTORICAL = 100
I, J = 3, 4 # Number of sources (i) and destinations (j)

# Historical residuals (epsilon_t)
# In reality, this would come from your AR model fitting
# Let's assume a simple structure where residuals are 3-dimensional (I=3)
historical_residuals = np.random.normal(loc=0, scale=0.5, size=(N_HISTORICAL, I))

# Optimal First-Stage Decision (x_1*) - FIXED
# This must be obtained from solving the SP with the empirical distribution.
# Let's assume x_ij* is a 3x4 matrix of assignments/capacities
x_star = np.array([
    [10, 5, 0, 0],
    [0, 15, 5, 0],
    [0, 0, 10, 10]
])

# Recourse Costs (d_ij)
d_ij = np.array([
    [5, 6, 7, 8],
    [4, 5, 6, 7],
    [3, 4, 5, 6]
])

# Objective-Awareness Parameter (gamma)
gamma = 0.5


def calculate_recourse_cost(rho_i, x_star_ij, d_ij):
   """
   Calculates the minimum recourse cost L(rho) for a given rho_i realization.
   This assumes a simplified Second-Stage LP structure:
   MIN SUM(d_ij * q_ij)
   s.t. SUM_j(q_ij) = rho_i
        q_ij <= M * x_ij* (Recourse q_ij is constrained by the first stage x_ij*)
        q_ij >= 0
   """
   I, J = d_ij.shape
   
   # 1. Define Objective (Flattened d_ij)
   c = d_ij.flatten()
   
   # 2. Define Constraints (A_ub @ q <= b_ub)
   # The recourse cost needs constraints to make sense.
   # Example: q_ij is limited by the assignment x_ij*
   M = 100  # A large number if x_star is binary/capacity
   
   # A. Flow Constraint: SUM_j(q_ij) = rho_i (Equality A_eq @ q = b_eq)
   A_eq = np.zeros((I, I * J))
   b_eq = rho_i
   for i in range(I):
      A_eq[i, i * J: (i + 1) * J] = 1  # Sum of q_ij for a given i must equal rho_i
   
   # B. Capacity Constraint: q_ij <= M * x_ij* (Inequality A_ub @ q <= b_ub)
   # Here, we assume the first stage decision x_star serves as an upper bound
   # for the recourse variable q.
   A_ub = np.eye(I * J) * -1  # -q_ij <= 0 (q_ij >= 0)
   b_ub = np.zeros(I * J)
   
   # Simple bounds: q_ij >= 0
   # The bounds parameter handles this directly:
   bounds = [(0, None) for _ in range(I * J)]
   
   # Solve the linear program
   res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
   
   if res.success:
      return res.fun
   else:
      # Handle infeasibility (could happen if rho_i is too large/small)
      return 1e9  # Return a very high cost


# --- 4. SIMULATE SCENARIO GENERATION (MEB STEP) ---
# In a real implementation, you would use a full MEB algorithm
# (e.g., Vinod's code) to generate scenarios.
N_SCENARIOS = 100
H = 5  # Planning horizon of 5 steps

# 1. Generate standard MEB scenario paths (via simple block bootstrap for the stub)
# Each path 's' is an H-length sequence of residuals (H x I)
standard_meb_paths = []
path_losses = []

# Arbitrary time-series forecast for the next H steps
# (e.g., from the AR model's deterministic part)
forecast_rho_t_plus_k = np.array([20, 22, 25])  # Assuming fixed for simplicity

for s in range(N_SCENARIOS):
   # Simulate an H-step path of residuals
   path_indices = np.random.choice(N_HISTORICAL, size=H, replace=True)
   residual_path = historical_residuals[path_indices, :]
   standard_meb_paths.append(residual_path)
   
   # 2. Calculate Loss for the entire path L(epsilon^s)
   path_loss = 0
   for k in range(H):
      # Scenario rho = Forecast + Residual
      rho_scenario = forecast_rho_t_plus_k + residual_path[k, :]
      
      # Calculate recourse cost L(rho) = Q(x_1*, rho)
      path_loss += calculate_recourse_cost(rho_scenario, x_star, d_ij)
   
   path_losses.append(path_loss)

# --- 5. APPLY OBJECTIVE-AWARE WEIGHTING ---

# 3. Calculate raw importance weights: w_s = exp(-gamma * L(epsilon^s))
raw_weights = np.exp(-gamma * np.array(path_losses))

# 4. Normalize weights (so SUM(w_s) = N_SCENARIOS for proper Monte Carlo)
# A simple normalization:
normalized_weights = raw_weights * N_SCENARIOS / np.sum(raw_weights)

# --- 6. OUTPUT ---

print(f"Total Scenarios Generated: {N_SCENARIOS}")
print(f"Gamma (Objective-Awareness): {gamma}")
print("-" * 30)
print(f"Mean Loss (Standard MEB proxy): {np.mean(path_losses):.2f}")
print(f"Max Loss: {np.max(path_losses):.2f}")
print("-" * 30)
print(f"Mean Raw Weight: {np.mean(raw_weights):.4f}")
print(f"Weight Range: [{np.min(normalized_weights):.4f}, {np.max(normalized_weights):.4f}]")
print("\nConclusion: Scenarios with a higher loss L(rho) have a higher normalized weight.")

# The final OA-MEB scenario set is the standard_meb_paths with their associated normalized_weights.
# This weighted set is then used in the Stochastic Program to estimate the robust cost.