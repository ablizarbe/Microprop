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
As_list = [5.4e-6, 5.16e-6]      # total heater area (m^2) - large, small (heater footprint)
r_in_list = [54e-6, 20e-6]       # t_wall (m) per case
r_out_list = [266e-6, 60e-6]     # D_o (m) per case

# Fixed length (you specified): 8.96 mm
L_fixed = 8.96e-3  # m



# Number of parallel channels
n_channels = 5

# Heater resistances (ohms)
R_heater_1 = 3.40
R_heater_2 = 2.38

# Heater wall temperature (kept for conduction model)
T_w = 473.0  # K

# Si / SiN conductivities (defaults; you can override)
k_si_default = 148.0    # W/mK (bulk Si at room temperature ~148)
k_sin_default = 3.0     # W/mK (LPCVD SiN typical thin-film estimate; process-dependent)
t_sin_default = 500e-9  # m (500 nm as corrected)

t_si_default = 100e-6   # m (default silicon conduction thickness; update to your actual value)

# Add new constants for Thermal Boundary Resistance (TBR) per unit area.
Rpp_tbr_mosin = 2.0e-8 # m^2*K/W (for the Molybdenum/SiN interface)
Rpp_tbr_sinsi = 2.0e-8 # m^2*K/W (for the SiN/Silicon interface)

# Physical properties (kept from your original file)
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

# Bypass for CHF limiter (when True, do not limit by CHF; use heater/conduction limits only)
CHF_BYPASS = True

# Default nozzle depth (out-of-plane) in meters (assumed; update with actual wafer/channel depth)
nozzle_depth_default = 100e-6

# Ambient/back pressure for thrust calculation (Pa)
P_AMBIENT = 0

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
# Geometry helper (circular pipe)
# -------------------------
def channel_geom_from_radii(r_in, r_out):
    """
    Circular pipe geometry (per single channel), corrected:
    Interprets inputs as:
      - r_in  -> t_wall (pipe wall thickness) [m]
      - r_out -> D_o    (pipe outer diameter) [m]

    Computes inner diameter D_i = D_o - 2*t_wall and returns:
      - A_cs  = π D_i^2 / 4
      - P_wet = π D_i
      - D_h   = D_i

    Note: The previous implementation incorrectly treated r_in and r_out as
    inner/outer radii of an annulus. This has been corrected per project specs.
    """
    t_wall = float(r_in)
    D_o = float(r_out)
    D_i = D_o - 2.0 * t_wall
    if D_i <= 0:
        raise ValueError(
            f"Invalid pipe geometry: inner diameter <= 0 (D_i={D_i:.3e} m) given t_wall={t_wall:.3e} m and D_o={D_o:.3e} m"
        )
    A_cs = 0.25 * np.pi * D_i**2                 # cross-sectional area (m^2) per channel
    P_wet = np.pi * D_i                          # wetted perimeter (m) per channel
    D_h = D_i                                    # hydraulic diameter for circular pipe
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
# CHF Diagnostic Helper
# -------------------------
def debug_chf_diagnostics(
    mdot_channel, A_cs, G_channel, D_h, T_sat_curr, curr_P,
    rho_v_curr, Eo, Bo_crit, qpp_chf_per_wetted,
    P_wet_total, a_per_length, qpp_chf,
    qpp_heater_global, qpp_possible_cond=None,
    qpp_convective_heater=None, qpp_sensible_heaterbasis=None,
    R_v=461.5, DEBUG=False
):
    """
    Prints a diagnostic table for CHF calculation variables if DEBUG is True.
    This is a non-invasive helper function.
    """
    if DEBUG:
        print("\n--- CHF DIAGNOSTICS ---")
        
        # CoolProp-based densities for comparison
        try:
            rho_l_coolprop = PropsSI('D', 'P', curr_P, 'Q', 0, 'Water')
            rho_v_coolprop = PropsSI('D', 'P', curr_P, 'Q', 1, 'Water')
        except Exception:
            rho_l_coolprop = float('nan')
            rho_v_coolprop = float('nan')
            
        # Ideal gas density for comparison
        rho_v_ideal = curr_P / (R_v * T_sat_curr) if T_sat_curr > 0 else float('nan')

        # Ratio for conversion
        ratio = P_wet_total / a_per_length if a_per_length > 0 else float('nan')

        print(f"  mdot_channel (kg/s)      : {mdot_channel:.3e}")
        print(f"  A_cs (m²)                : {A_cs:.3e}")
        print(f"  G_channel (kg/m²·s)      : {G_channel:.3e}")
        print(f"  D_h (m)                  : {D_h:.3e}")
        print(f"  T_sat_curr (K)           : {T_sat_curr:.3f}")
        print(f"  curr_P (Pa)              : {curr_P:.3e}")
        print(f"  rho_l_coolprop (kg/m³)   : {rho_l_coolprop:.3e}")
        print(f"  rho_v_coolprop (kg/m³)   : {rho_v_coolprop:.3e}")
        print(f"  rho_v_ideal (kg/m³)      : {rho_v_ideal:.3e} (model uses rho_v_curr={rho_v_curr:.3e})")
        print(f"  Eo (dim-less)            : {Eo:.3e}")
        print(f"  Bo_crit (dim-less)       : {Bo_crit:.3e}")
        print(f"  qpp_chf_per_wetted (W/m²): {qpp_chf_per_wetted:.3e}")
        print(f"  P_wet_total (m)          : {P_wet_total:.3e}")
        print(f"  a_per_length (m²/m)      : {a_per_length:.3e}")
        print(f"  Ratio (P_wet/a_len) (1/m): {ratio:.3e}")
        print(f"  qpp_chf final (W/m²_h)   : {qpp_chf:.3e}")
        print(f"  qpp_heater_global (W/m²_h): {qpp_heater_global:.3e}")
        if qpp_possible_cond is not None:
            print(f"  qpp_possible_cond (W/m²_h): {qpp_possible_cond:.3e}")
        if qpp_convective_heater is not None:
             print(f"  qpp_convective (W/m²_h)  : {qpp_convective_heater:.3e}")
        if qpp_sensible_heaterbasis is not None:
             print(f"  qpp_sensible (W/m²_h)    : {qpp_sensible_heaterbasis:.3e}")
        print("-------------------------\n")

# -------------------------
# Core march() for multiple channels
# -------------------------
def march_annulus_multichannel(mdot_total, P_in, r_in, r_out, A_module, W_total,
                               L=L_fixed, n_channels=5,
                               t_sin=t_sin_default, k_sin=k_sin_default,
                               t_si=t_si_default,
                               N=200):
    """
    mdot_total: total mass flow across all channels (kg/s)
    r_in, r_out: interpreted as t_wall (m) and D_o (m) for standard circular pipes
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

    # Conduction thickness for Si is now explicit and decoupled from flow geometry
    # (do NOT derive from r_out - r_in which are pipe wall thicknesses, not silicon)
    t_si = max(1e-12, t_si)

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
        rho_v_curr = curr_P / (R_v * T_sat_curr) if (R_v > 0 and T_sat_curr > 0) else 0.0
        rho_m = (1.0 - curr_alpha) * rho_l + curr_alpha * rho_v_curr
        mu_m = (1.0 - curr_alpha) * mu_l + curr_alpha * mu_v

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

        # CHF baseline (use per-channel G when computing Bo_crit, but qpp_chf is per heater area)
        G_channel = mdot_channel / A_cs if A_cs > 0 else 0.0
        Eo = g * (rho_l - rho_v_curr) * D_h**2 / sigma if sigma > 0 else 1.0
        Bo_crit = 0.12 * np.sqrt(rho_v_curr / rho_l) * (1.0 + Eo**(-0.5)) if Eo > 0 else 0.12
        # qpp_chf derived from per-channel G but expressed as per-heater-area limit:
        # latent power per channel per unit axial length (W/m_axial) = Bo_crit * G_channel * h_fg * (A_heater_per_channel_per_length?)
        # Simpler consistent approach: compute qpp_chf_per_wetted_area ~ Bo_crit * G_channel * h_fg,
        # then convert to per-heater-area by multiplying by (P_wet_channel / a_per_length) ratio if needed.
        # For consistency with previous code and your Bo_crit formulation, we compute qpp_chf as:
        qpp_chf_per_wetted = Bo_crit * G_channel * h_fg   # W per m^2_wetted (approx)
        # Convert CHF limit to per-heater-area basis (W per m^2_heater)
        # total latent power available per axial length (W/m_axial) limited by qpp_chf_per_wetted * P_wet_total
        # convert to flux per heater area: (W/m_axial) / a_per_length = qpp_chf_per_wetted * (P_wet_total / a_per_length)
        qpp_chf = qpp_chf_per_wetted * (P_wet_total / a_per_length) if a_per_length > 0 else 0.0
        
        # Apply CHF bypass if requested
        effective_qpp_chf = 1e30 if CHF_BYPASS else qpp_chf

        # --- DIAGNOSTIC HOOK ---
        # Call the helper function to print intermediate CHF variables if DEBUG is on.
        debug_chf_diagnostics(
            mdot_channel, A_cs, G_channel, D_h, T_sat_curr, curr_P,
            rho_v_curr, Eo, Bo_crit, qpp_chf_per_wetted,
            P_wet_total, a_per_length, qpp_chf, qpp_heater_global, DEBUG=DEBUG
        )
        # Non-invasive sanity check for extremely large CHF values.
        if DEBUG and qpp_chf > 1e9:
            print(f"WARNING: Extremely high qpp_chf calculated ({qpp_chf:.3e} W/m²_heater). Check inputs and formulas.")
        # --- END DIAGNOSTIC HOOK ---

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

                # Latent per heater-area (CHF may be bypassed)
                qpp_latent = min(qpp_heater_global, effective_qpp_chf, qpp_possible_cond)

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
            # boiling region: latent (CHF may be bypassed)
            qpp_latent = min(qpp_heater_global, effective_qpp_chf, qpp_possible_cond)
            if DEBUG and i % (N // 2) == 0: # Print this for a few slices to avoid clutter
                print(f" Limiting Fluxes (W/m^2):")
                print(f"  Heater Power : {qpp_heater_global:.3e}")
                print(f"  Conduction   : {qpp_possible_cond:.3e}")
                print(f"  CHF (New)    : {qpp_chf:.3e} (bypass={'ON' if CHF_BYPASS else 'OFF'})")
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
# Nozzle modeling utilities
# -------------------------
def nozzle_geometry_areas(params, depth=nozzle_depth_default):
        """
        params: dict with keys (units in micrometers):
            - w_nd: nozzle divergent exit width [µm]
            - l_nd: nozzle divergent length [µm]
            - w_nc: nozzle convergent inlet width at chamber [µm]
            - l_nc: nozzle convergent length [µm]
            - w_t : throat width [µm]
        depth: out-of-plane nozzle depth [m]. Defaults to nozzle_depth_default.

        Returns dict with cross-sectional areas in m^2:
            A_t (throat), A_e (exit). Approximates rectangular cross-sections: A = width * depth.
        """
        um = 1e-6
        w_t = params['w_t'] * um
        w_nd = params['w_nd'] * um
        A_t = w_t * depth
        A_e = w_nd * depth
        return {'A_t': A_t, 'A_e': A_e}

def _area_ratio_from_M(M, gamma):
    term = (1 + (gamma - 1) / 2 * M**2)
    exponent = (gamma + 1) / (2 * (gamma - 1))
    return (1.0 / M) * ( (2 / (gamma + 1)) * term )**exponent

def _solve_M_from_area_ratio(Ae_At, gamma, supersonic=True):
    # Solve A/A* = f(M) for M; choose supersonic branch if requested
    # Use a robust bracket: [1+eps, 50] for supersonic; [1e-6, 1-eps] for subsonic (not used here)
    if supersonic:
        a, b = 1.0001, 50.0
    else:
        a, b = 1e-6, 0.9999
    f = lambda M: _area_ratio_from_M(M, gamma) - Ae_At
    return brentq(f, a, b)

def isentropic_nozzle_performance(P1, T1, A_t, A_e, gamma=1.33, R=R_v, p_ambient=P_AMBIENT):
    """
    Compute nozzle exit performance using the user's equations (isentropic relations):
      - Thrust:        F = m_dot * Ve + (pe - pa) * Ae
      - Exhaust vel.:  Ve = Me * sqrt(k * R * T1)
      - Mass flow:     m_dot = (At * p1 * k * sqrt( (2/(k+1))^((k+1)/(k-1)) )) / sqrt(k * R * T1)
      - Area ratio:    Ae/At = ((k+1)/2)^(-(k+1)/(2*(k-1))) * Me^-1 * (1 + ((k-1)/2) * Me^2)^((k+1)/(2*(k-1)))
      - Exit temp.:    Te = T1 * (1 + ((k-1)/2) * Me^2)^-1
      - Exit pressure: pe = p1 * (1 + ((k-1)/2) * Me^2)^(-k / (k-1))
      - Isp:           Isp = F / (m_dot * g)
    Returns dict with F, Isp, pe, Te, Ve, mdot, Me.
    """
    if A_t <= 0 or A_e <= 0:
        return {'F':0.0,'Isp':0.0,'p_e':np.nan,'T_e':np.nan,'Ve':0.0,'mdot':0.0,'Me':np.nan}

    # Compute area ratio from geometry; depth cancels, so this equals width_exit/width_throat
    Ae_At = A_e / A_t

    # Solve for supersonic exit Mach number using area-M relation
    Me = _solve_M_from_area_ratio(Ae_At, gamma, supersonic=True)

    # User-provided mass flow relation (note: uses chamber total conditions)
    mdot = (A_t * P1 * gamma * np.sqrt( (2.0 / (gamma + 1.0))**((gamma + 1.0) / (gamma - 1.0)) )) / np.sqrt(gamma * R * T1)

    # Exit velocity using user's formula (depends on T1)
    Ve = Me * np.sqrt(gamma * R * T1)

    # Exit temperature and pressure using isentropic relations
    Te = T1 * (1.0 + (gamma - 1.0) / 2.0 * Me**2)**-1.0
    pe = P1 * (1.0 + (gamma - 1.0) / 2.0 * Me**2)**(-gamma / (gamma - 1.0))

    # Thrust including pressure term
    F = mdot * Ve + (pe - p_ambient) * A_e

    g0 = 9.80665
    Isp = F / (mdot * g0) if mdot > 0 else 0.0

    return {'F': F, 'Isp': Isp, 'p_e': pe, 'T_e': Te, 'Ve': Ve, 'mdot': mdot, 'Me': Me}

def nozzle_params_catalog():
        """Return the three nozzle types provided by user, with dimensions in micrometers."""
        return {
                'L': {'w_nd':500, 'l_nd':645, 'w_nc':3000, 'l_nc':2600, 'w_t':45},
                'W': {'w_nd':780, 'l_nd':660, 'w_nc':3000, 'l_nc':1500, 'w_t':45},
                'B': {'w_nd':500, 'l_nd':500, 'w_nc':3000, 'l_nc':1600, 'w_t':45},
        }

# -------------------------
# Example run for both cases
# -------------------------
if __name__ == "__main__":
    # Power selection
    V_applied = 7.0
    W_heater1 = heater_power_from_voltage(V_applied, heater_id=1)
    print(
        "Heater powers (W): heater1=", W_heater1,
        " heater2=", heater_power_from_voltage(V_applied, heater_id=2),
        " both=", heater_power_from_voltage(V_applied, both=True)
    )

    nozzle_types = nozzle_params_catalog()
    results = []

    # Run 6 cases: 2 serpentine designs x 3 nozzle types
    case_id = 1
    for A_module, r_in, r_out in zip(As_list, r_in_list, r_out_list):
        L_calc = L_fixed
        # Flow solution with CHF bypass active
        mdot, info = find_mdot_with_check_annulus(
            1e5, W_heater1, r_in, r_out, A_module,
            n_channels=n_channels, mdot_min=1e-12, mdot_max=1e-3, n_samples=120
        )
        x, T, P_arr, alpha, Q = march_annulus_multichannel(
            mdot, 1e5, r_in, r_out, A_module, W_heater1,
            L=L_calc, n_channels=n_channels, N=200
        )

        # Chamber conditions for nozzle model: use outlet conditions
        P0 = P_arr[-1]
        T0 = T[-1]

        for nozzle_name, params in nozzle_types.items():
            areas = nozzle_geometry_areas(params, depth=nozzle_depth_default)
            perf = isentropic_nozzle_performance(
                P1=P0, T1=T0, A_t=areas['A_t'], A_e=areas['A_e'], gamma=1.33, R=R_v, p_ambient=P_AMBIENT
            )

            # Metrics: P (W), mdot (mg/s), F (mN), Isp (s), p (bar), T (K), tau (mN/W)
            P_in_W = W_heater1
            mdot_mgs = perf['mdot'] * 1e6
            F_mN = perf['F'] * 1e3
            Isp_s = perf['Isp']
            p_bar = perf['p_e'] / 1e5
            T_K = perf['T_e']
            tau_mN_per_W = F_mN / P_in_W if P_in_W > 0 else 0.0

            print(f"Case {case_id}: A={A_module:.3e} m2, t_wall={r_in:.3e} m, D_o={r_out:.3e} m, Nozzle={nozzle_name}")
            print(f"  P={P_in_W:.3f} W, mdot={mdot_mgs:.3f} mg/s, F={F_mN:.3f} mN, Isp={Isp_s:.2f} s, p={p_bar:.3f} bar, T={T_K:.2f} K, tau={tau_mN_per_W:.3f} mN/W")

            results.append({
                'case': case_id,
                'A': A_module,
                't_wall': r_in,
                'D_o': r_out,
                'nozzle': nozzle_name,
                'P_W': P_in_W,
                'mdot_mgs': mdot_mgs,
                'F_mN': F_mN,
                'Isp_s': Isp_s,
                'p_bar': p_bar,
                'T_K': T_K,
                'tau_mN_per_W': tau_mN_per_W,
                'Me': perf['Me'],
                'alpha_exit': alpha[-1],
            })
            case_id += 1

    df = pd.DataFrame(results)
    print("\nSummary (6 cases):")
    print(df.to_string(index=False))