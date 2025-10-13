import numpy as np
from scipy.optimize import brentq
import pandas as pd
from CoolProp.CoolProp import PropsSI

# ---------------------------
# Constants (fluid & physics)
# ---------------------------
g = 9.81  # m/s2
sigma = 0.059  # N/m (surface tension for water - keep your value)
T_in = 293.0  # K (inlet)
R_v = 461.5  # J/kg K for water vapor (as you used)
rho_l = 958.0  # kg/m3 (liquid water approx at ~100C)
mu_l = 2.82e-4  # Pa s
k_l = 0.68  # W/m K
cp_l = 4216  # J/kg K
mu_v = 1.26e-5  # Pa s
k_v = 0.024  # W/m K
cp_v = 2080  # J/kg K
h_fg = 2.257e6  # J/kg

# Heater resistances (user-specified)
R_heater = {1: 3.40, 2: 2.38}  # ohm

# Material conductivities (used for 1D conduction stack)
# I use representative values; see references below for sources and ranges.
k_silicon = 148.0   # W/m K (crystalline silicon at room/near-room)
k_sin = 2.5         # W/m K (thin LPCVD silicon nitride films vary widely; 1-5 W/mK is typical for thin films)
k_molybdenum = 138.0  # W/m K (metal heater, not explicitly used here except if you want heater conduction)

# Silicon nitride thickness (you confirmed 500 nm)
t_sin = 500e-9  # m

# Serpentine curvature radius (kept from your code)
R_c = 1e-3  # m, adjust if needed

DEBUG = True

# ---------------------------
# Thermo helper functions
# ---------------------------
def T_sat_func(P):
    """Return saturation temperature (K) for water at pressure P (Pa)."""
    return PropsSI('T', 'P', P, 'Q', 0, 'Water')

# ---------------------------
# Annular geometry helpers
# ---------------------------
def annulus_geom(r_in, r_out):
    """Return cross-sectional area, wetted perimeter, hydraulic diameter for concentric annulus."""
    if not (0 <= r_in < r_out):
        raise ValueError("Require 0 <= r_in < r_out")
    A_cs = np.pi * (r_out**2 - r_in**2)
    P_wet = 2.0 * np.pi * (r_out + r_in)
    D_h = 4.0 * A_cs / P_wet
    return A_cs, P_wet, D_h

# ---------------------------
# Nusselt for annulus (fully-developed, laminar, constant heat flux)
# ---------------------------
def Nu_annulus_constq(r_in, r_out):
    """
    Approximate fully-developed laminar Nusselt number for a concentric annulus, constant heat flux.
    Implementation: anchor at two canonical limits and interpolate smoothly:
      - tube (eta -> 0): Nu_constq_tube = 4.36
      - narrow-gap / parallel plate (eta -> 1): Nu_constq_plate = 8.235
    For intermediate radius ratios eta = r_in / r_out we interpolate on a small canonical table.
    Sources: classical duct convection data (see references: Shah & London; Incropera & DeWitt).
    """
    eta = r_in / r_out
    # canonical points (eta, Nu) chosen to follow classical tabulated behaviour; coarse but robust.
    # If you need higher-fidelity, we can replace with a full Shah & London table.
    eta_points = np.array([0.0, 0.1, 0.2, 0.33, 0.5, 0.75, 0.9, 0.99, 1.0])
    Nu_points = np.array([4.36, 4.5, 4.8, 5.5, 6.6, 7.2, 7.8, 8.1, 8.235])
    # clamp and interpolate
    eta_clamped = max(0.0, min(1.0, eta))
    Nu = float(np.interp(eta_clamped, eta_points, Nu_points))
    return Nu

# ---------------------------
# Thermal stack conduction (per unit area)
# ---------------------------
def conduction_resistance_per_area(t_si, k_si, t_sin, k_sin):
    """
    Series thermal resistance per unit area (m^2 K / W) for heater -> SiN -> Si -> fluid interface.
    We model heater -> SiN (t_sin) -> Si (t_si) conduction (1D series).
    If you want to add moly heater thickness or other layers, add them here.
    """
    Rpp = 0.0
    if k_si <= 0 or k_sin <= 0:
        return np.inf
    Rpp = t_si / k_si + t_sin / k_sin
    return Rpp

# ---------------------------
# Revised march() with annulus + conduction stack + heater-as-voltage
# ---------------------------
def march(mdot, P_in, r_in, r_out, A_module, L, heater_voltage, heater_id=1, N=200):
    """
    Marching solver for annular channel.

    Inputs:
      - mdot: mass flow rate (kg/s)
      - P_in: inlet pressure (Pa)
      - r_in, r_out: inner and outer radius of annulus (m)
      - A_module: heated module area (m^2)
      - L: axial length (m) computed as A_module/P_wet
      - heater_voltage: applied voltage (V)
      - heater_id: 1 or 2 (selects R_heater)
    Returns:
      x, T_b, P_arr, alpha, Q_total_absorbed
    """
    # geometry
    A_cs, P_wet, D_h = annulus_geom(r_in, r_out)
    dx = L / N
    x = np.linspace(0.0, L, N+1)
    T_b = np.zeros(N+1)
    P = np.zeros(N+1)
    alpha = np.zeros(N+1)
    qpp_arr = np.zeros(N)

    # initial conditions
    T_b[0] = T_in
    P[0] = P_in
    alpha[0] = 0.0

    # compute heater power W from voltage & heater id
    if heater_id not in R_heater:
        raise ValueError("heater_id must be 1 or 2")
    Rh = R_heater[heater_id]
    W_total = (heater_voltage**2) / Rh   # W total delivered to module
    if A_module <= 0:
        raise ValueError("A_module must be > 0")
    qpp_heater_global = W_total / A_module   # W/m^2 (electrical power per heated area)

    # Conduction stack: silicon thickness between heater and fluid assumed r_out - r_in
    t_si = max(1e-12, (r_out - r_in))
    Rpp_stack = conduction_resistance_per_area(t_si, k_silicon, t_sin, k_sin)  # m^2 K / W

    for i in range(N):
        curr_P = P[i]
        curr_T = T_b[i]
        curr_alpha = alpha[i]

        # local saturation and vapor properties
        T_sat_curr = T_sat_func(curr_P)
        rho_v_curr = curr_P / (R_v * T_sat_curr) if (R_v>0 and T_sat_curr>0) else 0.0
        rho_m = (1.0 - curr_alpha) * rho_l + curr_alpha * rho_v_curr
        mu_m = (1.0 - curr_alpha) * mu_l + curr_alpha * mu_v

        if rho_m <= 0 or mu_m <= 0:
            rho_m = rho_l
            mu_m = mu_l

        # mean velocity and Re
        u_m = mdot / (rho_m * A_cs) if (rho_m * A_cs) > 0 else 0.0
        Re_m = rho_m * u_m * D_h / mu_m if mu_m > 0 else 0.0

        # friction (use your rectangular friction replacement? here we use smooth laminar/turbulent fallback)
        # For simplicity in annulus: use Blasius-like turbulent or laminar 64/Re if laminar
        if Re_m > 2300:
            f = 0.3164 * Re_m**(-0.25)
        else:
            f = 64.0 / Re_m if Re_m > 1e-12 else 0.0

        delta = D_h / (2.0 * R_c) if R_c > 0 else 0.0
        De = Re_m * np.sqrt(delta) if delta > 0 else 0.0
        dP_dx = -f * rho_m * u_m**2 / (2.0 * D_h)

        # CHF baseline variable (for debugging)
        G = mdot / A_cs if A_cs>0 else 1.0
        Eo = g * (rho_l - rho_v_curr) * D_h**2 / sigma if sigma>0 else 1.0
        Bo_crit = 0.12 * np.sqrt(rho_v_curr / rho_l) * (1.0 + Eo**(-0.5)) if Eo>0 else 0.12
        qpp_chf = Bo_crit * G * h_fg if (G>0 and h_fg>0) else 0.0

        # sensible convective coefficient (annulus Nu)
        Nu_l = Nu_annulus_constq(r_in, r_out)
        h_l_curr = max(1e-12, Nu_l * k_l / D_h)

        # Now compute the effective heater-side temperature for the given electrical flux:
        # We treat the heater flux q'' = qpp_heater_global as flowing through Rpp_stack and then across convective h.
        # This gives a heater temperature (not used directly but used to compute the actual convective driving).
        qpp = qpp_heater_global
        # local bulk temp that convection sees - for boiling we use saturation temperature; else use local T_b
        T_bulk_local = T_sat_curr if curr_T >= T_sat_curr - 1e-6 else curr_T

        # total thermal series resistance from heater to bulk fluid (per unit area)
        Rpp_total = Rpp_stack + 1.0 / h_l_curr  # (m^2 K / W)

        # heater-side temperature (if you need it) and the effective convective driving
        T_heater_side = T_bulk_local + qpp * Rpp_total

        # convective flux that can actually transfer to fluid (cap by qpp and by h*(T_heater_side - curr_T))
        qpp_convective = h_l_curr * max(0.0, T_heater_side - curr_T)
        qpp_used = 0.0
        E_in = 0.0
        dalpha_dx = 0.0
        T_next = curr_T
        phase = 'unknown'

        # sensible heating capacity (per slice) limited by available convective flux
        qpp_sensible = min(qpp, qpp_convective)
        E_sensible = qpp_sensible * P_wet * dx   # sensible energy used in this slice (W)

        # energy required to reach saturation for entire flow in slice
        E_needed_to_sat = mdot * cp_l * max(0.0, (T_sat_curr - curr_T))

        if curr_T < T_sat_curr - 1e-12:
            # single-phase heating: may reach saturation
            if E_sensible < E_needed_to_sat - 1e-15:
                # only sensible heating
                DeltaT = E_sensible / (mdot * cp_l) if (mdot>0 and cp_l>0) else 0.0
                T_next = curr_T + DeltaT
                qpp_used = qpp_sensible
                phase = 'single'
                E_in = E_sensible
            else:
                # hits saturation: use remaining energy for latent (CHF-limited)
                E_remain_total = E_sensible - E_needed_to_sat
                # CHF-limited latent flux (local)
                Eo_loc = g * (rho_l - rho_v_curr) * D_h**2 / sigma if sigma>0 else 1.0
                Bo_crit_loc = 0.12 * np.sqrt(rho_v_curr / rho_l) * (1.0 + Eo_loc**(-0.5)) if Eo_loc>0 else 0.12
                qpp_chf_loc = Bo_crit_loc * (mdot / A_cs) * h_fg if (A_cs>0 and h_fg>0) else 0.0
                qpp_latent = min(qpp, qpp_chf_loc)
                E_latent_available = qpp_latent * P_wet * dx
                E_latent_used = min(E_remain_total, E_latent_available)
                dalpha = E_latent_used / (mdot * h_fg) if (mdot>0 and h_fg>0) else 0.0
                dalpha_dx = dalpha / dx if dx>0 else 0.0
                T_next = T_sat_curr
                phase = 'boiling'
                qpp_used = (E_sensible + E_latent_used) / (P_wet * dx)
                E_in = E_sensible + E_latent_used

        elif curr_alpha < 1.0:
            # boiling region: latent limited by CHF
            Eo_loc = g * (rho_l - rho_v_curr) * D_h**2 / sigma if sigma>0 else 1.0
            Bo_crit_loc = 0.12 * np.sqrt(rho_v_curr / rho_l) * (1.0 + Eo_loc**(-0.5)) if Eo_loc>0 else 0.12
            qpp_chf_loc = Bo_crit_loc * (mdot / A_cs) * h_fg if (A_cs>0 and h_fg>0) else 0.0
            qpp_latent = min(qpp, qpp_chf_loc)
            E_latent = qpp_latent * P_wet * dx
            dalpha = E_latent / (mdot * h_fg) if (mdot>0 and h_fg>0) else 0.0
            dalpha_dx = dalpha / dx if dx>0 else 0.0
            T_next = T_sat_curr
            phase = 'boiling'
            qpp_used = qpp_latent
            E_in = E_latent

        else:
            # all vapor: superheat by convection from heated wall
            Nu_v = Nu_annulus_constq(r_in, r_out)  # reuse same function for vapor side
            h_v_curr = max(1e-12, Nu_v * k_v / D_h)
            # heater-side temp is T_heater_side computed above; convective driving = h_v*(T_heater_side - curr_T)
            qpp_v = min(qpp, h_v_curr * max(0.0, (T_heater_side - curr_T)))
            E_v = qpp_v * P_wet * dx
            DeltaT = E_v / (mdot * cp_v) if (mdot>0 and cp_v>0) else 0.0
            T_next = curr_T + DeltaT
            phase = 'super'
            qpp_used = qpp_v
            E_in = E_v

        # guard alpha
        delta_alpha = (dalpha_dx * dx) if (dalpha_dx is not None) else 0.0
        if curr_alpha + delta_alpha > 1.0:
            allowable = max(0.0, 1.0 - curr_alpha)
            if allowable > 0 and (mdot>0 and h_fg>0):
                E_latent_allowed = allowable * mdot * h_fg
                E_sensible_used = min(E_sensible, E_needed_to_sat) if 'E_sensible' in locals() else 0.0
                total_E_used = E_sensible_used + E_latent_allowed
                qpp_used = total_E_used / (P_wet * dx)
                dalpha_dx = allowable / dx if dx>0 else 0.0
                E_in = total_E_used
            else:
                dalpha_dx = 0.0

        qpp_arr[i] = float(qpp_used)
        alpha[i+1] = min(1.0, max(0.0, curr_alpha + (dalpha_dx * dx if dalpha_dx is not None else 0.0)))
        T_b[i+1] = min(T_heater_side, T_next)  # limiting: can't exceed heater-side computed
        P[i+1] = max(1e3, curr_P + dP_dx * dx)

        if DEBUG and (i % max(1, N//20) == 0):
            print(f"[i={i}] x={x[i]:.6f} m, Re={Re_m:.3g}, De={De:.3g}, qpp_used={qpp_used:.3g} W/m2, qpp_chf={qpp_chf:.3g},")
            print(f"         E_in={E_in:.3e} W, E_need={E_needed_to_sat:.3e} W, phase={phase}, T={curr_T:.2f}->{T_b[i+1]:.2f}, alpha={alpha[i+1]:.3f}")

    # total absorbed heat (use actual qpp_arr)
    a_per_length = A_module / L   # m^2 of heated surface per m axial
    Q_total = a_per_length * np.trapz(qpp_arr, x[:-1])  # W
    return x, T_b, P, alpha, Q_total

# ---------------------------
# find_mdot_with_check (wrapper) - minor adaptions to new signature
# ---------------------------
def estimate_mdot_needed_for_full_absorption(P_in, W, A_cs, A_module, r_in, r_out):
    """
    Quick estimate: compute G needed so qpp_chf >= qpp_heater (approx).
    Returns mdot_needed and info dict for debugging.
    """
    T_sat = T_sat_func(P_in)
    rho_v = P_in / (R_v * T_sat) if (R_v>0 and T_sat>0) else 0.0
    A_cs_local = A_cs
    D_h_local = 4.0 * A_cs_local / (2.0 * np.pi * (r_out + r_in))
    Eo = g * (rho_l - rho_v) * D_h_local**2 / sigma if sigma>0 else 1.0
    Bo_crit = 0.12 * np.sqrt(rho_v / rho_l) * (1.0 + Eo**(-0.5)) if Eo>0 else 0.12
    qpp_heater = W / A_module
    if Bo_crit * h_fg <= 0:
        return np.nan, {'Bo_crit':Bo_crit, 'qpp_heater':qpp_heater}
    G_needed = qpp_heater / (Bo_crit * h_fg)
    mdot_needed = G_needed * A_cs_local
    return mdot_needed, {'T_sat':T_sat, 'rho_v':rho_v, 'D_h':D_h_local, 'Eo':Eo, 'Bo_crit':Bo_crit, 'qpp_heater':qpp_heater, 'G_needed':G_needed}

def find_mdot_with_check(P_in, heater_voltage, heater_id, r_in, r_out, A_module,
                         mdot_min=1e-12, mdot_max=1e-3, n_samples=80):
    """
    Wrapper to sample Q(mdot) and find mdot that absorbs the heater power.
    """
    # compute length from wetted perimeter
    A_cs, P_wet, D_h = annulus_geom(r_in, r_out)
    L = A_module / P_wet
    # compute electrical W
    if heater_id not in R_heater:
        raise ValueError("heater_id must be 1 or 2")
    W = (heater_voltage**2) / R_heater[heater_id]

    mdot_est, info = estimate_mdot_needed_for_full_absorption(P_in, W, A_cs, A_module, r_in, r_out)
    print(f"Estimate mdot required to avoid CHF-limiting <--> mdot_needed = {mdot_est:.3e} kg/s")
    print("Intermediate info:", info)

    mdot_candidates = np.logspace(np.log10(mdot_min), np.log10(mdot_max), n_samples)
    Qs = np.full_like(mdot_candidates, np.nan, dtype=float)

    for i, m in enumerate(mdot_candidates):
        try:
            _,_,_,_,Q = march(m, P_in, r_in, r_out, A_module, L, heater_voltage, heater_id, N=200)
            Qs[i] = Q
        except Exception as e:
            Qs[i] = np.nan
            if DEBUG:
                print(f" march failed at mdot={m:.3e}: {e}")

    if np.all(np.isnan(Qs)):
        raise RuntimeError("All march() attempts failed in find_mdot_with_check sampling. Check march() implementation.")

    Qmax = np.nanmax(Qs)
    idx_max = int(np.nanargmax(Qs))
    m_at_Qmax = mdot_candidates[idx_max]
    print(f"Maximum Q achievable in sampled range = {Qmax:.4g} W at mdot = {m_at_Qmax:.3e} kg/s")

    if Qmax < W * 0.999:
        print("WARNING: heater power cannot be fully absorbed by fluid in sampled mdot range.")
        return m_at_Qmax, {'Qmax':Qmax, 'mdot_best':m_at_Qmax, 'L':L, 'A_cs':A_cs}

    # find bracket where Q-W crosses zero and rootfind
    for k in range(len(mdot_candidates)-1):
        f1 = Qs[k] - W
        f2 = Qs[k+1] - W
        if np.isnan(f1) or np.isnan(f2):
            continue
        if f1 == 0.0:
            return mdot_candidates[k], {'Q':Qs[k], 'L':L}
        if f1 * f2 < 0:
            a = mdot_candidates[k]; b = mdot_candidates[k+1]
            sol = brentq(lambda m: march(m, P_in, r_in, r_out, A_module, L, heater_voltage, heater_id)[4] - W, a, b)
            return sol, {'Q':W, 'L':L}

    return m_at_Qmax, {'Qmax':Qmax, 'mdot_best':m_at_Qmax, 'L':L}

# ---------------------------
# CASES (large / small)
# ---------------------------
if __name__ == "__main__":
    # your two As and radii (large first, small second)
    As = [5.4e-6, 5.16e-6]  # m^2
    r_ins = [54e-6, 20e-6]  # m
    r_outs = [266e-6, 60e-6]  # m

    cases = []
    for i in range(2):
        A_module = As[i]
        r_in = r_ins[i]
        r_out = r_outs[i]
        A_cs, P_wet, D_h = annulus_geom(r_in, r_out)
        L = A_module / P_wet
        cases.append({'label': 'large' if i==0 else 'small',
                      'A_module':A_module,
                      'r_in':r_in,
                      'r_out':r_out,
                      'A_cs':A_cs,
                      'P_wet':P_wet,
                      'D_h':D_h,
                      'L':L})

    print("Defined cases (computed lengths from A_module / P_wet):")
    for c in cases:
        print(f"{c['label']:>6} : A={c['A_module']:.3e} m2, r_in={c['r_in']*1e6:.1f} um, r_out={c['r_out']*1e6:.1f} um, L={c['L']*1e3:.3f} mm")

    # Example: pick case 0 (large), ambient P_in and some heater voltage
    P_in = 5e5
    heater_voltage = 5.0  # V (example)
    heater_id = 1
    case = cases[0]
    mdot, info = find_mdot_with_check(P_in, heater_voltage, heater_id, case['r_in'], case['r_out'], case['A_module'],
                                      mdot_min=1e-12, mdot_max=1e-4, n_samples=120)
    x, T, P_arr, alpha, Q = march(mdot, P_in, case['r_in'], case['r_out'], case['A_module'], case['L'], heater_voltage, heater_id, N=200)
    print("mdot found (kg/s) = ", mdot, " Q_absorbed = ", Q)
