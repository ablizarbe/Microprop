import numpy as np
from scipy.optimize import brentq
import pandas as pd
from scipy.interpolate import interp1d
from CoolProp.CoolProp import PropsSI

class SerpentineCorrelation:
    def __init__(self, eNu_file='eNu_data.csv', ef_file='ef_data.csv'):
        try:
            # Load data using pandas
            eNu_data = pd.read_csv(eNu_file)
            ef_data = pd.read_csv(ef_file)

            # Create interpolation functions
            self.eNu_interp = interp1d(eNu_data['Reynolds'], eNu_data['eNu'], bounds_error=False, fill_value="extrapolate")
            self.ef_interp = interp1d(ef_data['Reynolds'], ef_data['ef'], bounds_error=False, fill_value="extrapolate")

            if DEBUG:
                print("Successfully loaded and initialized serpentine correlations.")

        except FileNotFoundError as e:
            print(f"Error: Correlation file not found: {e}. Please ensure '{eNu_file}' and '{ef_file}' are in the same directory.")
            # Fallback to default values if files are not found
            self.eNu_interp = lambda x: 1.0
            self.ef_interp = lambda x: 1.0
        except Exception as e:
            print(f"An error occurred during correlation initialization: {e}")
            self.eNu_interp = lambda x: 1.0
            self.ef_interp = lambda x: 1.0

    def get_enhancement_factors(self, Re):
        """
        Returns the enhancement factors eNu and ef for a given Reynolds number.
        """
        return self.eNu_interp(Re), self.ef_interp(Re)


# -------------------------
# USER / MATERIAL SETTINGS
# -------------------------
# Two cases (large, small) as requested:
As_list = [5.4e-6, 5.16e-6]      # total heater area (m^2) - large, small (heater footprint)
r_in_list = [54e-6, 20e-6]       # inner radius (m)
r_out_list = [266e-6, 60e-6]     # outer radius (m)

# Fixed length
L_fixed = 8.96e-3  # m

# Number of parallel channels
n_channels = 5

# Heater resistances (ohms)
R_heater_1 = 3.40
R_heater_2 = 2.38

# Heater wall temperature
T_w = 473.0  # K

# Si / SiN conductivities (defaults; you can override)
k_si_default = 148.0    # W/mK (bulk Si at room temperature ~148)
k_sin_default = 3.0     # W/mK (LPCVD SiN typical thin-film estimate; process-dependent)
t_sin_default = 500e-9  # m (500 nm as corrected)

# Add constants for Thermal Boundary Resistance (TBR) per unit area.
Rpp_tbr_mosin = 2.0e-8 # m^2*K/W (for the Molybdenum/SiN interface)
Rpp_tbr_sinsi = 2.0e-8 # m^2*K/W (for the SiN/Silicon interface)

# Physical properties 
g = 9.81
sigma = 0.059
R_v = 461.5
rho_l = 958.0
mu_l = 2.82e-4
k_l = 0.68
cp_l = 4216.0
mu_v = 1.26e-5
k_v = 0.024
cp_v = 2080.0
h_fg = 2.257e6
R_c = 1e-3  # curvature radius

DEBUG = False

# -------------------------
# Temperature-Dependent Conductivity
# -------------------------
def get_temp_dependent_k(T_k):
    """
    Calculates temperature-dependent thermal conductivities for materials.
    T_k: Temperature in Kelvin
    """
    # Formula for Silicon (Si)
    k_si = 134122.4940 * (T_k ** -1.2073)
    
    # Formula for Molybdenum (Mo)
    k_mo = 152.78 - 5.0884e-2 * T_k + 9.675e-6 * (T_k ** 2)
    
    return k_si

# -------------------------
# Geometry helpers (annulus)
# -------------------------
def channel_geom_from_radii(r_in, r_out):
    """Annular cross-section geometry (per single channel)."""
    if r_out <= r_in:
        raise ValueError("r_out must be > r_in")
    A_cs = np.pi * (r_out**2 - r_in**2)           # cross-sectional area (m^2) per channel
    P_wet = 2.0 * np.pi * (r_out + r_in)          # wetted perimeter (m) per channel (inner+outer)
    D_h = 4.0 * A_cs / P_wet                      # hydraulic diameter per channel
    return A_cs, P_wet, D_h

# -------------------------
# Heaters by voltage
# -------------------------
def heater_power_from_voltage(V, heater_id=1, both=False):
    if both:
        return V**2 / R_heater_1 + V**2 / R_heater_2
    if heater_id == 1:
        return V**2 / R_heater_1
    elif heater_id == 2:
        return V**2 / R_heater_2
    else:
        raise ValueError("heater_id must be 1 or 2 (or set both=True)")

# -------------------------
# CHF Correlation
# -------------------------
def calculate_chf_qu_mudawar(G, L, D_h, h_fg, rho_f, rho_g, sigma):
    """
    Calculates the Critical Heat Flux (CHF) for saturated flow boiling in parallel
    microchannels using the Qu & Mudawar (2004) correlation.

    This correlation is based on heat flux per unit wetted area.

    Args:
        G (float): Mass flux (kg/m^2s)
        L (float): Total heated length of the microchannel (m)
        D_h (float): Hydraulic diameter of the microchannel (m)
        h_fg (float): Latent heat of vaporization (J/kg)
        rho_f (float): Saturated liquid density (kg/m^3)
        rho_g (float): Saturated vapor density (kg/m^3)
        sigma (float): Surface tension (N/m)

    Returns:
        float: CHF value per unit wetted area (W/m^2_wetted)
    """
    # Guard against division by zero or invalid inputs
    if not all([G > 0, L > 0, D_h > 0, h_fg > 0, rho_f > 0, rho_g > 0, sigma > 0]):
        return 0.0

    # Calculate dimensionless numbers
    weber_number = (G**2 * L) / (sigma * rho_f)
    density_ratio = rho_g / rho_f
    length_to_diameter_ratio = L / D_h

    if weber_number <= 0 or density_ratio <= 0 or length_to_diameter_ratio <= 0:
        return 0.0

    # Qu & Mudawar (2004) correlation for boiling number
    boiling_number = 33.43 * (density_ratio**-1.11) * \
                     (weber_number**0.21) * \
                     (length_to_diameter_ratio**-0.36)

    # Calculate CHF per unit wetted area
    q_chf_per_wetted = G * h_fg * boiling_number
    
    return q_chf_per_wetted

# -------------------------
# Core march() for multiple channels
# -------------------------
def march_annulus_multichannel(mdot_total, P_in, r_in, r_out, A_module, W_total,
                               L=L_fixed, n_channels=5,
                               t_sin=t_sin_default, k_sin=k_sin_default, N=200):
    """
    mdot_total: total mass flow across all channels (kg/s)
    r_in, r_out: radii of each annular channel (m)
    A_module: total heater footprint area (m^2) - for all channels combined
    W_total: total heater electrical power (W) - applied to entire heater footprint
    L: channel length (m) fixed to 8.96 mm (user-specified)
    n_channels: number of channels in parallel (e.g. 5)
    Returns x, T_b, P, alpha, Q_total_absorbed
    """

    # Ensure L is as requested (do not compute from A_module)
    if L is None:
        L = L_fixed

    # Per-channel geometry
    A_cs, P_wet_channel, D_h = channel_geom_from_radii(r_in, r_out)

    # Split total mdot among channels
    mdot_channel = mdot_total / float(n_channels)

    dx = L / N
    x = np.linspace(0.0, L, N+1)
    T_b = np.zeros(N+1)
    P = np.zeros(N+1)
    alpha = np.zeros(N+1)
    qpp_arr = np.zeros(N)  # heat flux per unit heater area (W/m^2_heater) for each axial slice

    # initial conditions
    T_b[0] = 293.0
    P[0] = P_in
    alpha[0] = 0.0

    # Heater flux per heater area (heater area = A_module total)
    if A_module <= 0:
        raise ValueError("A_module must be > 0")
    qpp_heater_global = W_total / A_module   # W per m^2_heater

    # Heater area per axial length (m^2_heater per m_axial)
    a_per_length = A_module / L

    # Conduction thickness for Si (you explicitly wanted t_si = r_out - r_in)
    t_si = max(1e-12, r_out - r_in)

    # Resistances per unit heater area (m^2 K / W)
    Rpp_sin = t_sin / k_sin

    # Total wetted perimeter for all channels (m per channel * n_channels)
    P_wet_total = P_wet_channel * n_channels

    # Instantiate the correlation helper
    correlation_helper = SerpentineCorrelation()

    # iterate slices
    for i in range(N):
        curr_P = P[i]
        curr_T = T_b[i]
        curr_alpha = alpha[i]

        # Update to use temperature-dependent k_si
        k_si_local = get_temp_dependent_k(curr_T)
        Rpp_si = t_si / k_si_local

        # local saturation and mixture props
        T_sat_curr = PropsSI('T', 'P', curr_P, 'Q', 0, 'Water')
        # Use CoolProp for all local saturated properties for consistency
        rho_f_curr = PropsSI('D', 'P', curr_P, 'Q', 0, 'Water')
        rho_g_curr = PropsSI('D', 'P', curr_P, 'Q', 1, 'Water')
        h_fg_curr = PropsSI('H', 'P', curr_P, 'Q', 1, 'Water') - PropsSI('H', 'P', curr_P, 'Q', 0, 'Water')
        sigma_curr = PropsSI('I', 'P', curr_P, 'Q', 0, 'Water') # Surface tension

        rho_m = (1.0 - curr_alpha) * rho_f_curr + curr_alpha * rho_g_curr
        mu_m = (1.0 - curr_alpha) * mu_l + curr_alpha * mu_v # Note: mu_l and mu_v are still constants, can be updated if desired

        if rho_m <= 0 or mu_m <= 0:
            rho_m = rho_l
            mu_m = mu_l

        # velocity and Re per channel (use per-channel area)
        u_m = mdot_channel / (rho_m * A_cs) if (rho_m * A_cs) > 0 else 0.0
        Re_m = rho_m * u_m * D_h / mu_m if mu_m > 0 else 0.0

        # Get enhancement factors from the correlation helper
        eNu, ef = correlation_helper.get_enhancement_factors(Re_m)

        # friction factor calculation for serpentine channels
        f_s = 64.0 / Re_m if Re_m > 1e-12 else 1e12
        f = ef * f_s
        dP_dx = -f * rho_m * u_m**2 / (2.0 * D_h)

        # CHF Calculation using Qu & Mudawar (2004) correlation
        G_channel = mdot_channel / A_cs if A_cs > 0 else 0.0
        qpp_chf_per_wetted = calculate_chf_qu_mudawar(
            G=G_channel,
            L=L,
            D_h=D_h,
            h_fg=h_fg_curr,
            rho_f=rho_f_curr,
            rho_g=rho_g_curr,
            sigma=sigma_curr
        )
        # Convert CHF limit to per-heater-area basis (W per m^2_heater)
        # total latent power available per axial length (W/m_axial) limited by qpp_chf_per_wetted * P_wet_total
        # convert to flux per heater area: (W/m_axial) / a_per_length = qpp_chf_per_wetted * (P_wet_total / a_per_length)
        qpp_chf = qpp_chf_per_wetted * (P_wet_total / a_per_length) if a_per_length > 0 else 0.0

        # Nusselt number calculation for serpentine channels
        Nu_straight = 4.36  # Baseline for straight circular pipe with uniform heat flux
        Nu = eNu * Nu_straight
        h_l_curr = max(1e-12, Nu * k_l / D_h)

        # convective conductance per heater area uses total wetted perimeter and heater area per length:
        conv_cond_per_heater_area = h_l_curr * (P_wet_total / a_per_length)   # [W/(m2_heater K)]

        # series conduction resistance per heater area (SiN + Si + conv)
        Rpp_total = Rpp_tbr_mosin + Rpp_sin + Rpp_tbr_sinsi + Rpp_si + 1.0 / conv_cond_per_heater_area

        # possible conductive-limited heat flux per heater area
        qpp_possible_cond = (T_w - curr_T) / Rpp_total if Rpp_total > 0 else 0.0

        # convective-only limit per heater area
        qpp_convective_heater = conv_cond_per_heater_area * max(0.0, (T_w - curr_T))

        # sensible heat available (per heater area) limited by heater, convection, conduction
        qpp_sensible_heaterbasis = min(qpp_heater_global, qpp_convective_heater, qpp_possible_cond)

        # energy (power) available in this slice (W) from heater sensible portion:
        E_sensible = qpp_sensible_heaterbasis * a_per_length * dx   # W * (m) => W (since qpp [W/m2] * area [m2] -> W); same as before

        # energy required (power) to raise all liquid in channels in slice to saturation:
        # using mdot_total (mass flow of all channels)
        E_needed_to_sat = mdot_total * cp_l * max(0.0, (T_sat_curr - curr_T))

        # initialize branch variables
        T_next = curr_T
        dalpha_dx = 0.0
        phase = 'unknown'
        qpp_used = 0.0
        E_in = 0.0

        # Branch logic (keeps your previous structure but adapted areas & mdot_total)
        if curr_T < T_sat_curr:
            # single-phase sensible heating (may reach saturation)
            if E_sensible < E_needed_to_sat:
                DeltaT = E_sensible / (mdot_total * cp_l) if (mdot_total > 0 and cp_l > 0) else 0.0
                T_next = curr_T + DeltaT
                qpp_used = qpp_sensible_heaterbasis
                phase = 'single'
                E_in = E_sensible
            else:
                # reach saturation inside slice: evaporate as allowed by CHF & conduction
                E_remain_total = E_sensible - E_needed_to_sat

                # CHF-limited latent per heater-area (use qpp_chf computed earlier)
                qpp_latent = min(qpp_heater_global, qpp_chf, qpp_possible_cond)

                # latent energy available in slice (W)
                E_latent_available = qpp_latent * a_per_length * dx
                E_latent_used = min(E_remain_total, E_latent_available)

                # update alpha using total mdot (global)
                dalpha = E_latent_used / (mdot_total * h_fg) if (mdot_total > 0 and h_fg > 0) else 0.0
                dalpha_dx = dalpha / dx if dx > 0 else 0.0
                T_next = T_sat_curr
                phase = 'boiling'
                # qpp_used expressed per-heater-area
                qpp_used = (E_sensible + E_latent_used) / (a_per_length * dx)
                E_in = E_sensible + E_latent_used

        elif curr_alpha < 1.0:
            # boiling region: latent limited by CHF
            qpp_latent = min(qpp_heater_global, qpp_chf, qpp_possible_cond)

            if i % (N // 2) == 0: # Print this for a few slices to avoid clutter
                print(f" Limiting Fluxes (W/m^2):")
                print(f"  Heater Power : {qpp_heater_global:.3e}")
                print(f"  Conduction   : {qpp_possible_cond:.3e}")
                print(f"  CHF (New)    : {qpp_chf:.3e}")
                print(f"  --> Active Limit: {qpp_latent:.3e}\n")
            
            E_latent = qpp_latent * a_per_length * dx
            dalpha = E_latent / (mdot_total * h_fg) if (mdot_total > 0 and h_fg > 0) else 0.0
            dalpha_dx = dalpha / dx if dx > 0 else 0.0
            T_next = T_sat_curr
            phase = 'boiling'
            qpp_used = qpp_latent
            E_in = E_latent
        else:
            # all vapor (superheated)
            Nu_v = eNu * 4.36
            h_v_curr = max(1e-12, Nu_v * k_v / D_h)
            conv_cond_v_per_heater_area = h_v_curr * (P_wet_total / a_per_length)
            qpp_v_heaterbasis = min(qpp_heater_global,
                                    conv_cond_v_per_heater_area * max(0.0, (T_w - curr_T)),
                                    (T_w - curr_T) / (Rpp_tbr_mosin + Rpp_sin + Rpp_tbr_sinsi + Rpp_si + 1.0 / conv_cond_v_per_heater_area))
            E_v = qpp_v_heaterbasis * a_per_length * dx
            DeltaT = E_v / (mdot_total * cp_v) if (mdot_total > 0 and cp_v > 0) else 0.0
            T_next = curr_T + DeltaT
            phase = 'super'
            qpp_used = qpp_v_heaterbasis
            E_in = E_v

        # Guard against alpha > 1
        delta_alpha = dalpha_dx * dx if dalpha_dx is not None else 0.0
        if curr_alpha + delta_alpha > 1.0:
            allowable = max(0.0, 1.0 - curr_alpha)
            if allowable > 0 and (mdot_total > 0 and h_fg > 0):
                E_latent_allowed = allowable * mdot_total * h_fg
                E_sensible_used = min(E_sensible, E_needed_to_sat) if 'E_sensible' in locals() else 0.0
                total_E_used = E_sensible_used + E_latent_allowed
                qpp_used = total_E_used / (a_per_length * dx)
                dalpha_dx = allowable / dx if dx > 0 else 0.0
                E_in = total_E_used
            else:
                dalpha_dx = 0.0

        qpp_arr[i] = float(qpp_used)

        # update
        alpha[i+1] = min(1.0, max(0.0, curr_alpha + (dalpha_dx * dx if dalpha_dx is not None else 0.0)))
        T_b[i+1] = min(T_w, T_next)
        P[i+1] = max(1e3, curr_P + dP_dx * dx)

        if DEBUG and (i % max(1, N//20) == 0):
            print(f"[i={i}] x={x[i]:.6f} m, Re={Re_m:.3g}, D_h={D_h:.3g}, qpp_used={qpp_used:.3g} W/m2_heater, qpp_chf={qpp_chf:.3g}")
            print(f"         E_in={E_in:.3e} W, E_need={E_needed_to_sat:.3e} W, phase={phase}, T={curr_T:.2f}->{T_b[i+1]:.2f}, alpha={alpha[i+1]:.3f}")

    # total absorbed heat (W) across all heater area and axial integration
    Q_total = a_per_length * np.trapezoid(qpp_arr, x[:-1])
    return x, T_b, P, alpha, Q_total

# -------------------------
# Estimate / search wrappers
# -------------------------
def estimate_mdot_needed_annulus(P_in, W, r_in, r_out, A_module, n_channels=5):
    T_sat = PropsSI('T','P',P_in,'Q',0,'Water')
    rho_v = P_in / (R_v * T_sat) if (R_v > 0 and T_sat > 0) else 0.0
    A_cs, P_wet_channel, D_h = channel_geom_from_radii(r_in, r_out)
    A_cs_total = A_cs * n_channels
    Eo = g * (rho_l - rho_v) * D_h**2 / sigma if sigma > 0 else 1.0
    Bo_crit = 0.12 * np.sqrt(rho_v / rho_l) * (1.0 + Eo**(-0.5)) if Eo > 0 else 0.12
    qpp_heater = W / A_module
    if Bo_crit * h_fg <= 0:
        return np.nan, {'Bo_crit':Bo_crit, 'qpp_heater':qpp_heater}
    G_needed = qpp_heater / (Bo_crit * h_fg)
    mdot_needed = G_needed * A_cs_total
    info = {'T_sat':T_sat, 'rho_v':rho_v, 'D_h':D_h, 'Eo':Eo, 'Bo_crit':Bo_crit, 'qpp_heater':qpp_heater, 'G_needed':G_needed}
    return mdot_needed, info

def find_mdot_with_check_annulus(P_in, W, r_in, r_out, A_module,
                                 n_channels=5, mdot_min=1e-12, mdot_max=1e-3, n_samples=80):
    mdot_est, info = estimate_mdot_needed_annulus(P_in, W, r_in, r_out, A_module, n_channels=n_channels)
    print(f"Estimate mdot required to avoid CHF-limiting <--> mdot_needed = {mdot_est:.3e} kg/s")
    print("Intermediate info:", info)

    mdot_candidates = np.logspace(np.log10(mdot_min), np.log10(mdot_max), n_samples)
    Qs = np.full_like(mdot_candidates, np.nan, dtype=float)

    for i, m in enumerate(mdot_candidates):
        try:
            _,_,_,_,Q = march_annulus_multichannel(m, P_in, r_in, r_out, A_module, W, L=L_fixed, n_channels=n_channels, N=200)
            Qs[i] = Q
        except Exception as e:
            Qs[i] = np.nan
            if DEBUG:
                print(f" march failed at mdot={m:.3e}: {e}")

    if np.all(np.isnan(Qs)):
        raise RuntimeError("All march attempts failed in sampling. Check implementation.")

    Qmax = np.nanmax(Qs)
    idx_max = int(np.nanargmax(Qs))
    m_at_Qmax = mdot_candidates[idx_max]
    print(f"Maximum Q achievable in sampled range = {Qmax:.4g} W at mdot = {m_at_Qmax:.3e} kg/s")

    if Qmax < W * 0.999:
        print("WARNING: heater power cannot be fully absorbed by fluid in sampled mdot range.")
        return m_at_Qmax, {'Qmax':Qmax, 'mdot_best':m_at_Qmax}

    # find root bracket
    for k in range(len(mdot_candidates)-1):
        f1 = Qs[k] - W
        f2 = Qs[k+1] - W
        if np.isnan(f1) or np.isnan(f2):
            continue
        if f1 == 0.0:
            return mdot_candidates[k], {'Q':Qs[k]}
        if f1 * f2 < 0:
            a = mdot_candidates[k]; b = mdot_candidates[k+1]
            sol = brentq(lambda m: march_annulus_multichannel(m, P_in, r_in, r_out, A_module, W, L=L_fixed, n_channels=n_channels)[4] - W, a, b)
            return sol, {'Q':W}

    return m_at_Qmax, {'Qmax':Qmax, 'mdot_best':m_at_Qmax}

# -------------------------
# Example run for both cases
# -------------------------
if __name__ == "__main__":
    # Example: apply voltage and heater 1
    V_applied = 7.0
    W_heater1 = heater_power_from_voltage(V_applied, heater_id=1)
    print("Heater powers (W): heater1=", W_heater1, " heater2=", heater_power_from_voltage(V_applied, heater_id=2),
          " both=", heater_power_from_voltage(V_applied, both=True))

    results = []

    for A_module, r_in, r_out in zip(As_list, r_in_list, r_out_list):
        # Use L_fixed (explicit)
        L_calc = L_fixed
        print(f"\nCase (A_module={A_module:.3e} m2, r_in={r_in:.3e} m, r_out={r_out:.3e} m) -> L = {L_calc*1e3:.3f} mm")

        # find mdot that absorbs heater1 power
        mdot, info = find_mdot_with_check_annulus(1e5, W_heater1, r_in, r_out, A_module,
                                                  n_channels=n_channels, mdot_min=1e-12, mdot_max=1e-3, n_samples=120)
        x, T, P_arr, alpha, Q = march_annulus_multichannel(mdot, 1e5, r_in, r_out, A_module, W_heater1,
                                                           L=L_calc, n_channels=n_channels, N=200)
        results.append({'A':A_module, 'r_in':r_in, 'r_out':r_out, 'L':L_calc, 'mdot':mdot, 'Q':Q, 'alpha_exit':alpha[-1]})
        print(" mdot found (kg/s) = ", mdot, " Q_absorbed = ", Q)

    df = pd.DataFrame(results)
    print("\nSummary:")
    print(df.to_string(index=False))