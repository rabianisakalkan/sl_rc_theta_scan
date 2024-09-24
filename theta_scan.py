import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
import csv


sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
h = 0.001
hsamp = 0.1
transient_time = 500.0


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


np.random.seed(10)
initial_condition = np.random.rand(3) * 20 - 10
state = initial_condition.copy()

num_transient_steps = int(transient_time / h)
for _ in range(num_transient_steps):
    state = runge_kutta_step(state, h, sigma, rho, beta)

sampling_steps = int(hsamp / h)
trajectory = []
num_sampling_steps = 20000

for _ in range(num_sampling_steps):
    for _ in range(sampling_steps):
        state = runge_kutta_step(state, h, sigma, rho, beta)
    trajectory.append(state)

trajectory = np.array(trajectory)
x_component = trajectory[:, 0]
y_component = trajectory[:, 1]
z_component = trajectory[:, 2]

min_val = np.min(x_component)
max_val = np.max(x_component)
scaled_x_component = 2 * (x_component - min_val) / (max_val - min_val) - 1


N_v = 30
dt = 0.01

m = np.random.uniform(-1, 1, N_v)
input_data = scaled_x_component


@njit
def hermite_interpolation(p0, m0, p1, m1, dt, t=0.5):
    return (2 * t**3 - 3 * t**2 + 1) * p0 + \
           (t**3 - 2 * t**2 + t) * m0 * dt + \
           (-2 * t**3 + 3 * t**2) * p1 + \
           (t**3 - t**2) * m1 * dt

@njit
def derivatives(y, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, interpol=False, dt=0.01, t=0.5):
    A, phi = y
    A_delay, phi_delay = delay_data

    if interpol:
        A_delayed = hermite_interpolation(A_delay[0], A_delay[1], A_delay[2], A_delay[3], dt, t)
        phi_delayed = hermite_interpolation(phi_delay[0], phi_delay[1], phi_delay[2], phi_delay[3], dt, t)
    else:
        A_delayed = A_delay[0]
        phi_delayed = phi_delay[0]

    dA_dt = pSL * A + gamma_real * A**3 + kappa * A_delayed * np.cos(phi_fb + phi_delayed - phi)
    dphi_dt = 1.0 + gamma_imag * A**2 + (kappa * A_delayed / A) * np.sin(phi_fb + phi_delayed - phi)

    return np.array([dA_dt, dphi_dt])

@njit
def rk4_step(y, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, dt):
    k1 = dt * derivatives(y, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, False, dt)
    y_mid = y + 0.5 * k1
    k2 = dt * derivatives(y_mid, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, True, dt, t=0.5)
    y_mid = y + 0.5 * k2
    k3 = dt * derivatives(y_mid, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, True, dt, t=0.5)
    k4 = dt * derivatives(y + k3, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, False, dt)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@njit
def solve(input_data, m, theta, N_v, pSL_base, eta, dt, tau, gamma_real, gamma_imag, kappa, phi_fb):
    steps_per_virtual_node = int(theta / dt)
    steps_per_input_sample = N_v * steps_per_virtual_node
    total_time_steps = len(input_data) * steps_per_input_sample

    time = np.arange(0, total_time_steps) * dt
    delay_steps = int(tau / dt)

    states = np.zeros((total_time_steps, 2), dtype=np.float64)
    states[0] = [np.abs(np.random.rand() + 1j * np.random.rand()), np.angle(np.random.rand() + 1j * np.random.rand())]

    pSL_values = np.zeros(total_time_steps)
    J_t_values = np.zeros(total_time_steps)
    L = len(input_data)
    S = np.zeros((L, N_v))

    for t in range(1, total_time_steps):
        t_continuous = t * dt
        k = int(t_continuous // (N_v * theta))
        if k >= len(input_data):
            break
        j = int(t_continuous // theta) % N_v

        J_t = input_data[k] * m[j]
        pSL = pSL_base + eta * J_t
        pSL_values[t] = pSL
        J_t_values[t] = J_t

        delay_index = max(0, t - delay_steps)
        delay_data = (
            np.array([states[delay_index, 0], (states[delay_index, 0] - states[max(0, delay_index - 1), 0]) / dt, states[max(0, delay_index - 1), 0], (states[max(0, delay_index - 1), 0] - states[max(0, delay_index - 2), 0]) / dt]),
            np.array([states[delay_index, 1], (states[delay_index, 1] - states[max(0, delay_index - 1), 1]) / dt, states[max(0, delay_index - 1), 1], (states[max(0, delay_index - 1), 1] - states[max(0, delay_index - 2), 1]) / dt])
        )

        states[t] = rk4_step(states[t - 1], delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, dt)

        if t % steps_per_virtual_node == steps_per_virtual_node - 1:
            l = k
            n = j
            S[l, n] = np.abs(states[t, 0]) ** 2

    A = states[:, 0]
    phi = states[:, 1]
    X_abs = A ** 2

    final_states = X_abs[::steps_per_input_sample]
    return S, final_states, time, pSL_values, J_t_values, X_abs


def ridge_regression(state_matrix, target, alpha):
    XtX = np.matmul(state_matrix.T, state_matrix)
    identity_matrix = np.eye(XtX.shape[0])
    regularized_XtX = XtX + alpha * identity_matrix
    XtX_inv = np.linalg.pinv(regularized_XtX)
    weights = np.matmul(np.matmul(XtX_inv, state_matrix.T), target)
    return weights

def calculate_nrmse(predictions, target):
    mse = mean_squared_error(target, predictions)
    variance = np.var(target)
    return np.sqrt(mse / variance)

def nrmse_to_file(file_name, theta, tau, nrmse_train, nrmse_test):
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Theta', 'Tau', 'NRMSE Train', 'NRMSE Test'])
        writer.writerow([theta, tau, nrmse_train, nrmse_test])


omega = 1.0
gamma_real = -0.1
gamma_imag = 0.0
kappa = 0.1
phi_fb = 0.0
eta = 0.1
pSL_base = 0.05

theta_values = np.linspace(0.5, 5, 100)  
file_name = 'nrmse_theta_scan.csv'

for theta in theta_values:
    tau = 1.41 * N_v * theta

    S, final_states, time, pSL_values, J_t_values, X_abs = solve(
        input_data, m, theta, N_v, pSL_base, eta, dt, tau,
        gamma_real, gamma_imag, kappa, phi_fb)

    bias_column = np.ones((S.shape[0], 1))
    S_with_bias = np.hstack((S, bias_column))

    buffer_size = 800
    train_start_index = buffer_size
    train_end_index = int(len(S_with_bias) * 0.8)
    test_start_index = train_end_index + buffer_size
    test_end_index = len(S_with_bias) - buffer_size

    S_train = S_with_bias[train_start_index:train_end_index, :]
    S_test = S_with_bias[test_start_index:test_end_index, :]
    target_train = x_component[train_start_index + 1:train_end_index + 1]
    target_test = x_component[test_start_index + 1:test_end_index + 1]

    alpha = 1e-6
    weights = ridge_regression(S_train, target_train, alpha)
    predictions_train = np.matmul(S_train, weights)
    predictions_test = np.matmul(S_test, weights)

    nrmse_train = calculate_nrmse(predictions_train, target_train)
    nrmse_test = calculate_nrmse(predictions_test, target_test)

    print(f"Theta={theta}, Tau={tau}, NRMSE Train={nrmse_train}, NRMSE Test={nrmse_test}")
    nrmse_to_file(file_name, theta, tau, nrmse_train, nrmse_test)
