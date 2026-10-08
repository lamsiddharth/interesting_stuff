import numpy as np

# Design matrix X — shape: (m, n)
# Each row is one training example; first column is all 1s (bias trick)
# so θ[0] acts as the intercept without a separate variable.
# X = [ 1  x₁ ]     m = 3 samples
#     [ 1  x₂ ]     n = 2 features (bias + one real feature)
#     [ 1  x₃ ]
X = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])
print(f"X.shape: {X.shape}")  # (m, n) = (3, 2)

# Target vector y — shape: (m, 1)
# y = [ y₁ ]
#     [ y₂ ]
#     [ y₃ ]
y = np.array([
    [3],
    [6],
    [7]
])
print(f"y.shape: {y.shape}")  # (m, 1) = (3, 1)

# Parameter vector θ — shape: (n, 1)
# θ = [ θ₀ ]   <- bias / intercept
#     [ θ₁ ]   <- weight for feature x
# Hypothesis: h = Xθ  ->  each prediction = θ₀ + θ₁·xᵢ
theta = np.array([
    [1],
    [2]
])
print(f"theta.shape: {theta.shape}")  # (n, 1) = (2, 1)

# ── Iterative method: Gradient Descent ──────────────────────────────────────
# Cost function (MSE):
#   J(θ) = (1/m) · Σ (hᵢ - yᵢ)²   where  h = Xθ
#
# Gradient:
#   ∂J/∂θ = (1/m) · Xᵀ(Xθ - y)     shape: (n, 1)
#
# Update rule (step opposite the gradient):
#   θ := θ - α · (1/m) · Xᵀ(Xθ - y)
#
# Repeat until MSE < convergence threshold.
# alpha = 0.1
# error = 1
# for _ in range(100):
#     h = np.dot(X, theta)                              # h = Xθ,  (3,2)@(2,1) → (3,1)
#     error = h - y                                     # residuals (hᵢ - yᵢ),  (3,1)
#     theta = theta - alpha * (1/len(y)) * np.dot(X.T, error)  # θ update
#     error = np.mean(np.square(error))                 # scalar MSE
#     print(f"Error: {error}")

# ── Closed-form method: Normal Equation ─────────────────────────────────────
# At the cost minimum, the gradient is zero:
#   ∂J/∂θ = 0
#   (1/m) · Xᵀ(Xθ - y) = 0
#   XᵀXθ = Xᵀy
#   θ = (XᵀX)⁻¹ Xᵀy      <- closed-form, no iterations needed
#
# np.linalg.solve(A, b) solves Aθ = b — more stable than inv(A)@b
#   A = XᵀX   shape: (n,m)@(m,n) → (n,n) = (2,2)
#   b = Xᵀy   shape: (n,m)@(m,1) → (n,1) = (2,1)
#   θ          shape: (n,1) = (2,1)
theta_nomal_eq = np.linalg.solve(
    X.T @ X,
    X.T @ y
)
print(f"Optimal theta (normal equation, no penalty/regulariztion):\n{theta_nomal_eq}")
training_error = np.mean(np.square(X @ theta_nomal_eq - y))
print(f"Training error (normal equation, no penalty/regulariztion): {training_error}")
# ── Regularized closed-form: Ridge Regression (L2) ──────────────────────────
# Problem: large θ values overfit; XᵀX can also be singular.
# Fix: add penalty term λ‖θ‖² to the cost to shrink coefficients toward zero.
#
# Modified cost:
#   J(θ) = (1/m) · Σ(hᵢ - yᵢ)² + λ · Σθⱼ²
#
# Gradient set to zero:
#   (1/m) · Xᵀ(Xθ - y) + 2λθ = 0
#
# Factor out θ  (since θ = I·θ, so 2λθ = 2λI·θ):
#   XᵀXθ + 2λIθ = Xᵀy
#   (XᵀX + 2λI) θ = Xᵀy
#   θ = (XᵀX + 2λI)⁻¹ Xᵀy
#
# Adding 2λI to the diagonal makes the system well-conditioned (invertible).
#   XᵀX + 2λI   shape: (n,n) = (2,2),   λ = 0.1

#smaller ridge penalty
theta_ridge = np.linalg.solve(
    X.T @ X + 2 * 0.1 * np.eye(X.shape[1]),
    X.T @ y
)
print(f"Optimal theta (ridge regression, smaller penalty):\n{theta_ridge}")
training_error = np.mean(np.square(X @ theta_ridge - y))
print(f"Training error (ridge regression, smaller penalty): {training_error}")
#larger ridge penalty
theta_ridge_large = np.linalg.solve(
    X.T @ X + 2 * 10 * np.eye(X.shape[1]),
    X.T @ y
)
training_error_large = np.mean(np.square(X @ theta_ridge_large - y))
print(f"Training error (ridge regression, large penalty): {training_error_large}")
print(f"Optimal theta (ridge regression, large penalty):\n{theta_ridge_large}")