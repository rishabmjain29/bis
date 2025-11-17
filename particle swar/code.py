import numpy as np

# ---------------------------------------------------------------
# 1. INPUT: Historical returns for crypto assets (daily or weekly)
# ---------------------------------------------------------------
# Example: pretend daily returns for 4 assets (BTC, ETH, SOL, XRP)
returns = np.array([
    [0.012, -0.004, 0.021, 0.015],
    [0.010,  0.002, 0.018, 0.011],
    [-0.005, 0.006, 0.017, 0.012],
    [0.014, 0.003, 0.023, 0.009],
    [0.009, -0.002, 0.012, 0.010]
])

# ----------------------------------------------
# 2. FINANCE FUNCTIONS: Return, Risk, Sharpe-like
# ----------------------------------------------
def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def objective_function(weights, mean_returns, cov_matrix):
    ret = portfolio_return(weights, mean_returns)
    risk = portfolio_risk(weights, cov_matrix)
    return -(ret / risk)   # Negative Sharpe ratio → minimize

# ---------------------------------------------------
# 3. PSO Algorithm Setup
# ---------------------------------------------------
num_particles = 40
num_dimensions = returns.shape[1]  # number of cryptos
iterations = 200

# Calculate mean returns & covariance
mean_returns = np.mean(returns, axis=0)
cov_matrix = np.cov(returns.T)

# Random initial weights (normalized)
particles = np.random.rand(num_particles, num_dimensions)
particles = particles / particles.sum(axis=1).reshape(-1, 1)

velocities = np.zeros((num_particles, num_dimensions))

# Personal bests
pbest_positions = particles.copy()
pbest_values = np.array([
    objective_function(w, mean_returns, cov_matrix) for w in particles
])

# Global best
gbest_index = np.argmin(pbest_values)
gbest_position = pbest_positions[gbest_index]
gbest_value = pbest_values[gbest_index]

# PSO Hyperparameters
w = 0.7    # inertia
c1 = 1.6   # cognitive
c2 = 1.6   # social

# ---------------------------------------------------
# 4. RUN PSO
# ---------------------------------------------------
for iteration in range(iterations):
    for i in range(num_particles):

        # Update velocity
        velocities[i] = (
            w * velocities[i] +
            c1 * np.random.rand() * (pbest_positions[i] - particles[i]) +
            c2 * np.random.rand() * (gbest_position - particles[i])
        )

        # Update position
        particles[i] = particles[i] + velocities[i]

        # Enforce weights >= 0
        particles[i] = np.clip(particles[i], 0, 1)

        # Normalize weights to sum = 1
        particles[i] = particles[i] / particles[i].sum()

        # Evaluate new fitness
        score = objective_function(particles[i], mean_returns, cov_matrix)

        # Update personal best
        if score < pbest_values[i]:
            pbest_positions[i] = particles[i]
            pbest_values[i] = score

            # Update global best
            if score < gbest_value:
                gbest_position = particles[i]
                gbest_value = score

# ----------------------------------------
# 5. OUTPUT RESULT
# ----------------------------------------
optimal_weights = gbest_position
optimal_return = portfolio_return(optimal_weights, mean_returns)
optimal_risk = portfolio_risk(optimal_weights, cov_matrix)
optimal_sharpe = optimal_return / optimal_risk

print("\n=== OPTIMAL CRYPTO PORTFOLIO (PSO) ===")
print("Weights (Allocation %):")
for i, w in enumerate(optimal_weights):
    print(f"Asset {i+1}: {w*100:.2f}%")

print(f"\nExpected Portfolio Return: {optimal_return:.4f}")
print(f"Portfolio Risk (Std Dev): {optimal_risk:.4f}")
print(f"Sharpe-like Ratio:        {optimal_sharpe:.4f}")
