import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd       


sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

h = 0.001  
hsamp = 0.1  
transient_time = 100.0 

def lorenz_system(state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def runge_kutta_step(state, h, sigma, rho, beta):
    k1 = lorenz_system(state, sigma, rho, beta)
    k2 = lorenz_system(state + 0.5 * h * k1, sigma, rho, beta)
    k3 = lorenz_system(state + 0.5 * h * k2, sigma, rho, beta)
    k4 = lorenz_system(state + h * k3, sigma, rho, beta)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

np.random.seed(42)
initial_condition = np.random.rand(3) * 20 - 10  
state = initial_condition.copy()

num_transient_steps = int(transient_time / h)
for _ in range(num_transient_steps):
    state = runge_kutta_step(state, h, sigma, rho, beta)

sampling_steps = int(hsamp / h)
trajectory = []
num_sampling_steps = 15000

for _ in range(num_sampling_steps):
    for _ in range(sampling_steps):
        state = runge_kutta_step(state, h, sigma, rho, beta)
    trajectory.append(state)

trajectory = np.array(trajectory)
x_component = trajectory[:, 0]


min_val = np.min(x_component)
max_val = np.max(x_component)
scaled_x_component = 2 * (x_component - min_val) / (max_val - min_val) - 1


class PolarComplexDE:
    def __init__(self, pSL, omega, gamma, kappa, phi_fb, tau, eta, dt=0.01):
        self.pSL_base = pSL
        self.omega = omega
        self.gamma = gamma
        self.kappa = kappa
        self.phi_fb = phi_fb
        self.tau = tau
        self.eta = eta
        self.dt = dt

    def hermite_interpolation(self, p0, m0, p1, m1, t=0.5):
        return (2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 * self.dt + (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * m1 * self.dt

    def derivatives(self, y, delay_data, pSL, interpol=False, t=0.5):
        A, phi = y
        A_delay, phi_delay = delay_data

        if interpol:
            A_delayed = self.hermite_interpolation(A_delay[0], A_delay[1], A_delay[2], A_delay[3], t)
            phi_delayed = self.hermite_interpolation(phi_delay[0], phi_delay[1], phi_delay[2], phi_delay[3], t)
        else:
            A_delayed = A_delay[0]
            phi_delayed = phi_delay[0]

        dA_dt = pSL * A + self.gamma.real * A**3 + self.kappa * A_delayed * np.cos(self.phi_fb + phi_delayed - phi)
        dphi_dt = self.omega + self.gamma.imag * A**2 + (self.kappa * A_delayed / A) * np.sin(self.phi_fb + phi_delayed - phi)

        return np.array([dA_dt, dphi_dt])

    def rk4_step(self, y, delay_data, pSL, dt):
        k1 = dt * self.derivatives(y, delay_data, pSL)
        y_mid = y + 0.5 * k1
        k2 = dt * self.derivatives(y_mid, delay_data, pSL, interpol=True, t=0.5)
        y_mid = y + 0.5 * k2
        k3 = dt * self.derivatives(y_mid, delay_data, pSL, interpol=True, t=0.5)
        k4 = dt * self.derivatives(y + k3, delay_data, pSL)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self, input_data):
        time = np.arange(0, len(input_data) * self.dt, self.dt)
        delay_steps = int(self.tau / self.dt)

        states = np.zeros((len(time), 2), dtype=float)
        states[0] = [np.abs(np.random.rand() + 1j * np.random.rand()), np.angle(np.random.rand() + 1j * np.random.rand())]

        for i in range(1, len(time)):
            delay_index = max(0, i - delay_steps)
            delay_data = (
                [states[delay_index, 0], (states[delay_index, 0] - states[max(0, delay_index - 1), 0]) / self.dt,
                 states[max(0, delay_index - 1), 0], (states[max(0, delay_index - 1), 0] - states[max(0, delay_index - 2), 0]) / self.dt],
                [states[delay_index, 1], (states[delay_index, 1] - states[max(0, delay_index - 1), 1]) / self.dt,
                 states[max(0, delay_index - 1), 1], (states[max(0, delay_index - 1), 1] - states[max(0, delay_index - 2), 1]) / self.dt]
            )

            pSL = self.pSL_base + self.eta * input_data[i]
            states[i] = self.rk4_step(states[i - 1], delay_data, pSL, self.dt)

        A = states[:, 0]
        phi = states[:, 1]
        X = A * np.exp(1j * phi)
        X_abs = A**2
        return X, X_abs, time


N_v = 25  
omega = 1.0
gamma = -0.1
kappa = 0.1
phi_fb = 0
dt = 0.01
eta = 1
pSL_base = 0.2


mask = np.random.uniform(-1, 1, N_v)
T_values = np.linspace(2500, 50000, num=10, dtype=int)  
nmse_train_values = []
nmse_test_values = []
theta_values = []

for T in T_values:
    theta = (T / N_v) * 0.01
    tau = 0.01 * 1.41 * T  
    theta_values.append(theta)
    
    lengthened_scaled_x_component = np.repeat(scaled_x_component, T)
    masked_lengthened_x_component = np.zeros_like(lengthened_scaled_x_component)
    for i in range(0, len(lengthened_scaled_x_component), T):
        for n in range(N_v):
            start_idx = i + n * (T // N_v)
            end_idx = i + (n + 1) * (T // N_v)
            masked_lengthened_x_component[start_idx:end_idx] = lengthened_scaled_x_component[start_idx:end_idx] * mask[n]
    
    reservoir = PolarComplexDE(pSL=pSL_base, omega=omega, gamma=gamma, kappa=kappa, phi_fb=phi_fb, tau=tau, eta=eta, dt=dt)
    X, X_abs, time = reservoir.solve(masked_lengthened_x_component)
    
    L = len(X_abs) // T
    S = np.zeros((L, N_v))
    for l in range(L):
        for n in range(N_v):
            end_idx = l * T + (n + 1) * (T // N_v) - 1 
            S[l, n] = X_abs[end_idx]
    
    bias_column = np.ones((S.shape[0], 1))
    S_with_bias = np.hstack((S, bias_column))
    
    target = x_component[1:]
    S = S_with_bias[:-1, :]
    S_train, S_test, target_train, target_test = train_test_split(S, target, test_size=0.3, random_state=42, shuffle=False)

    alpha = 1e-6  
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(S_train, target_train)
    
   
    predictions_train = ridge_model.predict(S_train)
    predictions_test = ridge_model.predict(S_test)
    
    mse_train = mean_squared_error(target_train, predictions_train)
    mse_test = mean_squared_error(target_test, predictions_test)
    
    variance_train = np.var(target_train)
    variance_test = np.var(target_test)
    
    nmse_train = np.sqrt(mse_train / variance_train)
    nmse_test = np.sqrt(mse_test / variance_test)
    
    nmse_train_values.append(nmse_train)
    nmse_test_values.append(nmse_test)


plt.figure(figsize=(10, 6))
plt.plot(theta_values, nmse_train_values, marker='o', linestyle='-', color='b', lw=1.5, label='Training NRMSE')
plt.plot(theta_values, nmse_test_values, marker='o', linestyle='-', color='r', lw=1.5, label='Testing NRMSE')
plt.xlabel('Theta')
plt.ylabel('Normalized Root Mean Squared Error')
plt.title('NRMSE vs Theta for Training and Testing Phases')
plt.legend()
plt.grid(True)
plt.show()
