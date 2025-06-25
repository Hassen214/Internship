
# 1. Forward variance curve interpolator
"""
def make_fv_curve_interpolator(T_array_nodes, fv_nodes, spline_order=3):
    spline_sqrt = splrep(T_array_nodes, np.sqrt(fv_nodes), k=spline_order)
    def fv(t):
        t = np.asarray(t)
        sqrt_interp = splev(t, spline_sqrt, der=0)
        return np.square(sqrt_interp)
    return fv
# 2. Simulate OU process
def simulate_OU_X(T, n_steps, eps, H, N_sims):
    dt = T / n_steps
    kappa = (0.5 - H) / eps
    sigma = eps ** (H - 0.5)
    X = np.zeros((N_sims, n_steps))
    Y = np.random.randn(N_sims, n_steps - 1)
    for i in range(1, n_steps):
        X[:, i] = X[:, i-1] - kappa * X[:, i-1] * dt + sigma * np.sqrt(dt) * Y[:, i-1]
    return Y, X

def simulate_OU_X(T, n_steps, eps=1/52, H=0.1, N_sims=10000):
    
    Simulates paths of the Ornstein-Uhlenbeck (OU) process and stores the trajectories and noise.

    Args:
        T (float): Total time duration for the simulation (e.g., 1 year).
        n_steps (int): Number of time steps in the simulation.
        eps (float, optional): Time scaling parameter (default is 1/52, assuming weekly steps).
        H (float, optional): Hurst parameter (default is 0.1).
        N_sims (int, optional): Number of simulation paths (default is 10000).

    Returns:
        Y_storage (ndarray): Array of shape (N_sims, n_steps) containing the standard normal random variables used as noise at each time step.
        trajectories (ndarray): Array of shape (N_sims, n_steps) containing the simulated Ornstein-Uhlenbeck process trajectories, scaled by the rough volatility term.
    

    dt = T / n_steps
    X = np.zeros((N_sims, n_steps))
    X[:, 0] = np.random.normal(loc=1, scale=np.sqrt(2), size=N_sims)

    trajectories = np.zeros((N_sims, n_steps))
    Y_storage = np.zeros((N_sims, n_steps))
    
    kappa = (0.5 - H) / eps
    beta = np.sqrt(eps**(2 * H) / (1 - 2 * H))

    for i in range(1, n_steps):
        Z = np.random.randn(N_sims)  # N(0,1) noise
        dX = beta * (np.exp(kappa * i * dt / 2) - np.exp(kappa * (i - 1) * dt)) * Z
        X[:, i] = X[:, i - 1] + dX

        alpha = np.exp((0.5 - H) * (-i * dt) / eps)
        trajectories[:, i] = alpha * X[:, i]
        Y_storage[:, i] = Z

    return Y_storage, trajectories



# 3. Horner method for polynomial evaluation
def horner_vector(poly, x):
    result = poly[0] * np.ones_like(x)
    for coeff in poly[1:]:
        result = result * x + coeff
    return result
# 4. sigma_t calculation
def sigma(ksi_0, X, a_k, eps, H, T, n_steps, N_sims):
    poly_vals = horner_vector(a_k, X)
    f_X_squared = poly_vals ** 2
    expected_fx2 = np.mean(f_X_squared, axis=0) + 1e-10
    t_grid = np.linspace(0, T, n_steps)
    ksi_vals = ksi_0(t_grid)
    sigma_squared = f_X_squared * ksi_vals[None, :] / expected_fx2[None, :]

    return np.sqrt(sigma_squared)
"""
# 5. Simulate log(S_t)
def simulate_logS_batched(S0, sigma_t, rho, dt, Y_trajectory):
    N_sims, n_steps = sigma_t.shape
    log_S = np.zeros((N_sims, n_steps))
    log_S[:, 0] = np.log(S0)
    for i in range(1, n_steps):
        sigma_prev = sigma_t[:, i - 1]
        drift = -0.5 * (rho * sigma_prev)**2 * dt
        diffusion = rho * sigma_prev * np.sqrt(dt) * Y_trajectory[:, i - 1]
        log_S[:, i] = log_S[:, i-1] + drift + diffusion
    return log_S

# 6. Vectorized implied volatility wrapper
def vec_find_vol_rat(prices, S0, strikes, T, flag, r=0.0):
    def safe_iv(price, K):
        try:
            return implied_volatility(price, S0, K, T, r, flag)
        except Exception as e:
            print(f"IV error for price={price:.4f}, K={K:.2f}: {e}")
            return np.nan
    return np.vectorize(safe_iv)(prices, strikes)

# ==== PARAMETERS ====
eps = 1/52
H = 0.1
a_k = [0.001, 0, 0.01, 0, 0.001, 0]
T = 1
n_steps = 5000
N_sims = 1000
S0 = 100
rho = 0.85
dt = T / n_steps

# Forward variance function
T_array_nodes = np.array([0, 0.5, 1.0, 1000])
fv_nodes = np.array([0.04, 0.04, 0.04, 0.04])  # 20% vol^2
fv_func = make_fv_curve_interpolator(T_array_nodes, fv_nodes)

# ==== SIMULATIONS ====
Y_trajectory, X = simulate_OU_X(T, n_steps, eps, H, N_sims)
sigma_t = sigma(fv_func, X, a_k, eps, H, T, n_steps, N_sims)
log_S = simulate_logS_batched(S0, sigma_t, rho, dt, Y_trajectory)
S_T = np.exp(log_S[:, -1])

# ==== OPTION PRICING ====
flag = 'c'
r = 0.0
lm = np.linspace(-0.2, 0.3, 50)
strike_array = S0 * np.exp(lm)

call_payoffs = np.maximum(S_T[:, None] - strike_array, 0)
call_prices = np.mean(call_payoffs, axis=0)

# ==== IMPLIED VOL ====
imp_vols = vec_find_vol_rat(call_prices, S0, strike_array, T, flag)

# Keep only valid entries
intrinsic_values = np.maximum(S0 - strike_array, 0)
valid = call_prices > intrinsic_values + 1e-4
strike_array_valid = strike_array[valid]
imp_vols_valid = imp_vols[valid]

# ==== PLOT ====
plt.figure(figsize=(8, 5))
plt.plot(strike_array_valid, imp_vols_valid, marker='o')
plt.title("Implied Volatility Smile (via Let's Be Rational)")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.grid(True)
plt.show()
