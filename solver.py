import numpy as np
from scipy.optimize import brentq
import pandas as pd

# Constants
h_channel = 135e-6  # m
g = 9.81  # m/s2
sigma = 0.059  # N/m
T_w = 473  # K
T_in = 293  # K
R_v = 461.5  # J/kg K for water vapor
rho_l = 958  # kg/m3
mu_l = 2.82e-4  # Pa s
k_l = 0.68  # W/m K
cp_l = 4216  # J/kg K
mu_v = 1.26e-5  # Pa s
k_v = 0.024  # W/m K
cp_v = 2080  # J/kg K
h_fg = 2.257e6  # J/kg

def T_sat_func(P):
    if P <= 0:
        return 273.15
    logP_torr = np.log10(P / 133.322)
    A = 5.1962
    B = 1730.63
    C = 233.426
    T_c = B / (A - logP_torr) - C
    return T_c + 273.15

def march(mdot, P_in, w, h_channel, L, N=200):
    A_cs = w * h_channel
    P_wet = 2 * (w + h_channel)
    D_h = 4 * A_cs / P_wet
    dx = L / N
    x = np.linspace(0, L, N+1)
    T_b = np.zeros(N+1)
    P = np.zeros(N+1)
    alpha = np.zeros(N+1)
    qpp_arr = np.zeros(N)
    T_b[0] = T_in
    P[0] = P_in
    alpha[0] = 0.0
    for i in range(N):
        curr_P = P[i]
        curr_T = T_b[i]
        curr_alpha = alpha[i]
        T_sat_curr = T_sat_func(curr_P)
        rho_v_curr = curr_P / (R_v * T_sat_curr)
        rho_m = (1 - curr_alpha) * rho_l + curr_alpha * rho_v_curr
        mu_m = (1 - curr_alpha) * mu_l + curr_alpha * mu_v
        if rho_m == 0 or mu_m == 0:
            rho_m = rho_l
            mu_m = mu_l
        u_m = mdot / (rho_m * A_cs)
        Re_m = rho_m * u_m * D_h / mu_m
        f = 64 / Re_m if Re_m > 1e-6 else 0
        f *= 1.2  # serpentine enhancement
        dP_dx = -f * rho_m * u_m**2 / (2 * D_h)
        # Heat transfer fixed
        h_l_curr = 30000  # fixed
        h_v_curr = 2000
        h_boiling_curr = 1e5
        if curr_T < T_sat_curr:
            h = h_l_curr
            DeltaT = T_w - curr_T
            qpp = max(0, h * DeltaT)
            dT_dx = qpp * P_wet / (mdot * cp_l)
            dalpha_dx = 0
            T_next = curr_T + dT_dx * dx
            phase = 'single'
        elif curr_alpha < 1:
            h = h_boiling_curr
            DeltaT = T_w - T_sat_curr
            qpp = max(0, h * DeltaT)
            dalpha_dx = qpp * P_wet / (mdot * h_fg)
            dT_dx = 0
            T_next = T_sat_curr
            phase = 'boiling'
        else:
            h = h_v_curr
            DeltaT = T_w - curr_T
            qpp = max(0, h * DeltaT)
            dT_dx = qpp * P_wet / (mdot * cp_v)
            dalpha_dx = 0
            T_next = curr_T + dT_dx * dx
            phase = 'super'
        # CHF check
        G = mdot / A_cs
        Eo = g * (rho_l - rho_v_curr) * D_h**2 / sigma if sigma > 0 else 1
        Bo_crit = 0.12 * np.sqrt(rho_v_curr / rho_l) * (1 + Eo**(-0.5)) if Eo > 0 else 0.12
        Bo = qpp / (G * h_fg) if G > 0 and h_fg > 0 else 0
        if Bo > Bo_crit:
            qpp = Bo_crit * G * h_fg
            if phase == 'boiling':
                dalpha_dx = qpp * P_wet / (mdot * h_fg)
            elif phase == 'single':
                dT_dx = qpp * P_wet / (mdot * cp_l)
                T_next = curr_T + dT_dx * dx
            elif phase == 'super':
                dT_dx = qpp * P_wet / (mdot * cp_v)
                T_next = curr_T + dT_dx * dx
        # Update
        alpha[i+1] = min(1.0, curr_alpha + dalpha_dx * dx)
        T_b[i+1] = min(T_w, T_next)
        P[i+1] = max(1e3, curr_P + dP_dx * dx)
        qpp_arr[i] = qpp
    Q_total = np.trapz(qpp_arr, x[:-1]) * P_wet
    return x, T_b, P, alpha, Q_total

def find_mdot(P_in, W, w, h_channel, L):
    def objective(mdot):
        _, _, _, _, Q = march(mdot, P_in, w, h_channel, L)
        return Q - W
    sol = brentq(objective, 1e-9, 1e-3)
    return sol

# Cases with longer L
cases = []
ws = [5e-6, 10e-6]
Ls = [0.01, 0.02]
P_ins = [1e5, 5e5, 1e6]
Ws = [1, 5, 10]
example_case = None
for w in ws:
    for L in Ls:
        for P_in in P_ins:
            for W in Ws:
                mdot = find_mdot(P_in, W, w, h_channel, L)
                x, T, P_arr, alpha, _ = march(mdot, P_in, w, h_channel, L)
                idx_vap = np.where(alpha >= 0.9)[0]
                x_vap = x[idx_vap[0]] if len(idx_vap) > 0 else L
                P_exit = P_arr[-1] / 1e5
                T_exit = T[-1]
                cases.append({
                    'w (um)': round(w * 1e6),
                    'L (mm)': round(L * 1000),
                    'P_in (bar)': round(P_in / 1e5, 1),
                    'W (W)': W,
                    'mdot (ug/s)': round(mdot * 1e6, 2),
                    'x_vap (mm)': round(x_vap * 1000, 1),
                    'T_exit (K)': round(T_exit, 1),
                    'P_exit (bar)': round(P_exit, 3),
                    'alpha_exit': round(alpha[-1], 2)
                })
                if example_case is None:
                    example_case = {'x': x[::20], 'T': T[::20], 'P': P_arr[::20] / 1e5, 'alpha': alpha[::20]}

df = pd.DataFrame(cases)
print("Summary Table:")
print(df.to_string(index=False))



print("\nExample Distributions for first case:")
for i in range(len(example_case['x'])):
    print(f"x={example_case['x'][i]:.5f}, T={example_case['T'][i]:.1f}, P={example_case['P'][i]:.3f}, alpha={example_case['alpha'][i]:.3f}")