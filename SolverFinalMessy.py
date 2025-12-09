import numpy as np
from scipy.optimize import brentq
import pandas as pd
from scipy.interpolate import interp1d
import os

# Optional plotting (installed via requirements); script will continue if missing
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# =============================================================
# CONFIGURATION (Adjustable Parameters)
# Grouped for quick tuning
# =============================================================
# Operating Conditions
V_APPLIED = 5.0          # Applied voltage (V) driving both heaters
MDOT_TOTAL = 0.82e-6     # Total inlet mass flow (kg/s)
P_INLET = 5.0e5          # Inlet / chamber feed pressure (Pa)
T_INLET = 293.0          # Inlet fluid temperature (K)

# Simulation Controls
N_SLICES = 200           # Axial discretization slices
N_CHANNELS = 5           # Number of parallel microchannels

# Heater & Material / Thermal Controls
USE_BOTH_HEATERS = True  # Always use combined heater power (no selection logic)
HEATER_WALL_T = 473.0    # Assumed wall temperature (K) for thermal resistance cap
MAX_SUPERHEAT = 30.0     # Bulk superheat cap above local saturation (K)
WALL_CAP_ENABLED = False  # Disable artificial wall temperature cap; compute wall from energy balance
REMOVE_SUPERHEAT_CAP = True  # Superheat bound disabled; fluid heating limited by wall temp and energy only

# Wall-to-fluid inefficiency parameters (reduce effective convective coupling)
FOULING_RPP = 1.0e-4           # Added fouling thermal resistance per unit heater area (m^2*K/W)
MALDISTRIBUTION_FACTOR = 0.85  # Fraction of heat to fluid after maldistribution (<1 reduces efficiency)
CONTACT_MIN = 0.60             # Min effective wall-fluid contact fraction due to bubble coverage
ENTRANCE_EFFECTS_ENABLED = True  # Apply thermal entrance-length development factor on Nu
FOULING_ENABLED = True           # Toggle fouling resistance
MALDISTRIBUTION_ENABLED = True   # Toggle maldistribution scaling
BOILING_CONTACT_ENABLED = True   # Toggle bubble coverage reduction in two-phase

# External/environmental heat loss parameters
T_AMBIENT_K = 293.15     # Ambient temperature for radiation/convection (K)
T_BASE_K = 293.15        # Backside heat sink temperature (K)
EPSILON_RAD = 0.8        # Emissivity of top heater surface (0..1)
H_EXT = 10.0             # External convection coefficient on top surface (W/m^2-K)
T_BACK_THICK = 500e-6    # Effective thickness from heater to ambient sink at the backside (m)
RPP_BACK_CONTACT = 2.0e-4  # Contact/interface resistance to sink per unit area (m^2-K/W)
SIGMA_SB = 5.670374419e-8 # Stefan-Boltzmann constant (W/m^2-K^4)

# Geometry (Heater footprint and pipe dimensions)
# Large and small heater footprints and associated channel wall thickness & outer diameters
HEATER_A_LIST = [5.4e-6, 5.16e-6]      # total heater area (m^2) - large, small
WALL_THICKNESS_LIST = [54e-6, 20e-6]   # t_wall (m) per case
OUTER_DIAM_LIST = [266e-6, 60e-6]      # D_o (m) per case
L_fixed = 8.96e-3                     # Fixed channel length (m)

# Nozzle depth (out-of-plane) in meters
NOZZLE_DEPTH_DEFAULT = 100e-6

# Heater resistances (Ohms)
R_heater_1 = 3.40
R_heater_2 = 2.38

# Thermal conductivity reference (used if not temp-dependent override)
k_si_default = 148.0    # W/mK (bulk Si)
k_sin_default = 3.0     # W/mK (SiN thin film)
t_sin_default = 500e-9  # m (SiN thickness)
t_si_default = 100e-6   # m (Silicon conduction thickness)

# Thermal Boundary Resistances (interface per unit area)
Rpp_tbr_mosin = 2.0e-8  # m^2*K/W (Mo/SiN)
Rpp_tbr_sinsi = 2.0e-8  # m^2*K/W (SiN/Si)

# Fluid / Physical Properties (water / steam approximation)
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
R_c = 1e-3  # curvature radius (placeholder)

# Ambient / back pressure (Pa)
P_AMBIENT = 0

# Default nozzle discharge coefficient if none specified
NOZZLE_CD_DEFAULT = 0.9

# Create legacy variable aliases (minimal downstream edits)
As_list = HEATER_A_LIST
r_in_list = WALL_THICKNESS_LIST
r_out_list = OUTER_DIAM_LIST
n_channels = N_CHANNELS
T_w = HEATER_WALL_T
nozzle_depth_default = NOZZLE_DEPTH_DEFAULT

# -------------------------
# Saturation temperature approximation
# -------------------------
def get_T_sat(P_pa):
    """Return saturation temperature of water (K) for pressure P_pa (Pa) using Antoine equation.
    Valid roughly for 1e4 Pa to ~1e6 Pa (0.1–10 bar). Clamps outside range.
    Antoine form (log10 P_mmHg = A - B/(C+T_C)). We invert for T.
    Using coefficients for water: A=8.14019, B=1810.94, C=244.485 (range ~1–100 °C).
    For higher pressures (>1 bar up to ~5 bar) we extend with a simple correction.
    """
    P_bar = max(0.01, P_pa / 1e5)
    P_mmHg = P_bar * 750.062
    A = 8.14019
    B = 1810.94
    C = 244.485
    # Avoid domain issues
    # Solve T_C = B / (A - log10(P_mmHg)) - C
    logP = np.log10(P_mmHg)
    denom = A - logP
    if denom <= 0.01:
        return 647.1  # near critical
    T_C = B / denom - C
    T_K = T_C + 273.15
    # Simple adjustment for pressures above Antoine range (~>2 bar): use steam tables slope
    if P_bar > 2.0:
        # Linear correction nudging toward ~425 K at 5 bar
        # At 2 bar Antoine gives ~393 K; actual ~393. At 5 bar true ~425 K.
        # Blend: add 0.0 at 2 bar up to + (425- Antoine(5bar)) at 5 bar.
        P5_mmHg = 5.0 * 750.062
        denom5 = A - np.log10(P5_mmHg)
        T5_C = B / denom5 - C
        T5_K = T5_C + 273.15
        delta5 = 425.0 - T5_K
        frac = (P_bar - 2.0) / 3.0
        T_K += max(0.0, delta5) * min(1.0, frac)
    return float(np.clip(T_K, 273.15, 647.1))

## User/material settings consolidated at top in CONFIGURATION section.


class SerpentineCorrelation:
    def __init__(self, eNu_file='eNu_data.csv', ef_file='ef_data.csv'):
        try:
            # Load data using pandas
            eNu_data = pd.read_csv(eNu_file)
            ef_data = pd.read_csv(ef_file)

            # Create interpolation functions
            self.eNu_interp = interp1d(eNu_data['Reynolds'], eNu_data['eNu'], bounds_error=False, fill_value="extrapolate")
            self.ef_interp = interp1d(ef_data['Reynolds'], ef_data['ef'], bounds_error=False, fill_value="extrapolate")

            # Correlations loaded successfully

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
        eNu = self.eNu_interp(Re)
        ef = self.ef_interp(Re)
        # Tighten/clamp to avoid unrealistic extrapolations at very low Re typical of microflows
        try:
            eNu = float(np.clip(eNu, 1.0, 3.0))  # heat transfer enhancement
            ef = float(np.clip(ef, 1.0, 1.5))    # friction enhancement
        except Exception:
            pass
        return eNu, ef


# -------------------------
# Dean-number-based Nusselt utility
# -------------------------
def compute_nusselt_dean(Re, Pr, D_h, R_c_eff, regime_hint=None):
    """
    Returns an effective Nusselt number for a curved (serpentine) circular channel
    using a simple Dean-number enhancement applied to straight-pipe baselines.

    - Dean number: De = Re * sqrt(D_h / (2*R_c))
    - Laminar baseline (constant heat flux): Nu0 = 4.36
    - Turbulent baseline: Gnielinski correlation
    - Curvature enhancement factors (conservative clamps to avoid blow-up):
        laminar:   e_De = 1 + 0.10 * De^0.50   (clamped to [1, 3])
        turbulent: e_De = 1 + 0.05 * De^0.20   (clamped to [1, 2])

    This is a pragmatic approximation for serpentine microchannels where Re is
    typically low-to-moderate and curvature increases secondary-flow mixing.
    """
    Re = max(1e-9, float(Re))
    Pr = max(1e-6, float(Pr))
    D_h = max(1e-12, float(D_h))
    R_c_eff = max(1e-9, float(R_c_eff))

    # Dean number
    De = Re * np.sqrt(D_h / (2.0 * R_c_eff))

    # Baselines
    Nu_lam = 4.36  # fully developed, constant heat flux
    # Turbulent baseline via Gnielinski (valid 3e3<Re<5e6, 0.5<Pr<200); we will guard outside
    if Re < 3000:
        # primarily laminar
        e_De = 1.0 + 0.10 * (De ** 0.50)
        e_De = float(np.clip(e_De, 1.0, 3.0))
        Nu = Nu_lam * e_De
    else:
        # friction factor for smooth pipe (Petukhov) and Gnielinski Nu
        f_pet = (0.79 * np.log(Re) - 1.64) ** -2
        Nu_gn = (f_pet / 8.0 * (Re - 1000.0) * Pr) / (1.0 + 12.7 * np.sqrt(f_pet / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))
        Nu_gn = max(Nu_gn, 3.66)  # ensure not below laminar limit
        e_De_t = 1.0 + 0.05 * (De ** 0.20)
        e_De_t = float(np.clip(e_De_t, 1.0, 2.0))
        Nu = Nu_gn * e_De_t

    # Transitional smoothing (optional): if 2000<Re<3000, blend
    if 2000.0 < Re < 3000.0:
        # compute laminar and turbulent and blend
        e_De_l = float(np.clip(1.0 + 0.10 * (De ** 0.50), 1.0, 3.0))
        Nu_l = Nu_lam * e_De_l
        f_pet = (0.79 * np.log(3000.0) - 1.64) ** -2
        Nu_gn_ref = (f_pet / 8.0 * (3000.0 - 1000.0) * Pr) / (1.0 + 12.7 * np.sqrt(f_pet / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))
        e_De_t = float(np.clip(1.0 + 0.05 * (De ** 0.20), 1.0, 2.0))
        Nu_t = max(Nu_gn_ref * e_De_t, 3.66)
        w = (Re - 2000.0) / 1000.0
        Nu = (1 - w) * Nu_l + w * Nu_t

    return float(max(Nu, 3.0))


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
# CHF and multiphase diagnostics removed for single-phase model
# -------------------------

# -------------------------
# Core march() for multiple channels
# -------------------------
def march_annulus_multichannel(mdot_total, P_in, r_in, r_out, A_module, W_total,
                               L=L_fixed, n_channels=5,
                               t_sin=t_sin_default, k_sin=k_sin_default,
                               t_si=t_si_default,
                               T_in=T_INLET,
                               N=N_SLICES,
                               wall_cap_enabled: bool = WALL_CAP_ENABLED,
                               max_superheat: float = MAX_SUPERHEAT):
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
    T_b = np.zeros(N+1)     # bulk fluid temperature (K)
    P = np.zeros(N+1)       # pressure (Pa)
    alpha = np.zeros(N+1)   # void fraction (0..1)
    x_qual = np.zeros(N+1)  # vapor quality (mass fraction of vapor, 0..1)
    qpp_arr = np.zeros(N)  # actual heat flux per unit heater area (W/m^2_heater) for each axial slice

    # initial conditions
    T_b[0] = T_in
    P[0] = P_in
    alpha[0] = 0.0

    # Heater flux per heater area (heater area = A_module total)
    if A_module <= 0:
        raise ValueError("A_module must be > 0")
    qpp_heater_global = W_total / A_module   # W per m^2_heater

    # Heater area per axial length (m^2_heater per m_axial)
    a_per_length = A_module / L

    # Total wetted perimeter for all channels (m per channel * n_channels)
    P_wet_total = P_wet_channel * n_channels

    # Instantiate the correlation helper
    correlation_helper = SerpentineCorrelation()

    T_wall_max = -np.inf
    # iterate slices
    for i in range(N):
        curr_P = P[i]
        curr_T = T_b[i]
        curr_x = x_qual[i]
        # Update to use temperature-dependent k_si
        k_si_local = get_temp_dependent_k(curr_T)
        Rpp_si = t_si / k_si_local if k_si_local > 0 else 1e12
        Rpp_sin = t_sin / k_sin if k_sin > 0 else 1e12

        # Local saturation temperature at current pressure
        T_sat_curr = get_T_sat(curr_P)
        # Saturated liquid/vapor properties at current pressure
        rho_l_curr = rho_l
        rho_v_curr = max(1e-3, curr_P / (R_v * T_sat_curr)) if (R_v > 0 and T_sat_curr > 0) else 1e-3

        # Determine regime and mixture properties (homogeneous-equilibrium model in two-phase)
        if curr_T < T_sat_curr and curr_x <= 1e-6:
            # Subcooled liquid
            x_loc = 0.0
            rho_phase = rho_l_curr
            mu_phase = mu_l
            k_phase = k_l
            cp_phase = cp_l
            T_fluid_eff = curr_T
        elif curr_x < 1.0:
            # Two-phase boiling: temperature pinned near saturation
            x_loc = curr_x
            rho_mix = 1.0 / (x_loc / rho_v_curr + (1.0 - x_loc) / rho_l_curr)
            mu_mix = (1.0 - x_loc) * mu_l + x_loc * mu_v
            k_mix = (1.0 - x_loc) * k_l + x_loc * k_v
            cp_mix = (1.0 - x_loc) * cp_l + x_loc * cp_v
            rho_phase = rho_mix
            mu_phase = mu_mix
            k_phase = k_mix
            cp_phase = cp_mix
            T_fluid_eff = T_sat_curr
        else:
            # Superheated vapor
            x_loc = 1.0
            rho_phase = max(1e-3, curr_P / (R_v * max(curr_T, 1.0)))
            mu_phase = mu_v
            k_phase = k_v
            cp_phase = cp_v
            T_fluid_eff = curr_T

        # velocity and Re per channel (use per-channel area)
        u_m = mdot_channel / (rho_phase * A_cs) if (rho_phase * A_cs) > 0 else 0.0
        Re_m = rho_phase * u_m * D_h / mu_phase if mu_phase > 0 else 0.0

        # Get enhancement factors from the correlation helper (already clamped inside)
        eNu, ef = correlation_helper.get_enhancement_factors(Re_m)

        # Darcy friction factor with regime awareness (smooth pipe baseline)
        if Re_m <= 0:
            f_s = 1e3
        elif Re_m < 2300.0:
            f_s = 64.0 / Re_m
        elif Re_m < 1.0e5:
            f_s = 0.3164 / (Re_m ** 0.25)  # Blasius
        else:
            # Haaland (smooth pipe: roughness/D ~ 0)
            f_s = (-1.8 * np.log10(6.9 / Re_m)) ** -2
        # Two-phase friction multiplier (bounded, modest increase around mid qualities)
        tp_mult = 1.0 + 10.0 * (x_loc * (1.0 - x_loc))  # in [1, 3.5]
        f = max(1e-6, ef * f_s * tp_mult)
        dP_dx_raw = -f * rho_phase * u_m**2 / (2.0 * D_h)
        # Clamp maximum fractional pressure loss per slice to avoid non-physical collapse in very low-density vapor
        max_frac_drop = 0.05  # 5% per slice upper bound
        max_drop_allowed = -max_frac_drop * curr_P / dx  # Pa/m equivalent
        dP_dx = max(dP_dx_raw, max_drop_allowed)

        # Prandtl number using local phase properties
        Pr_phase = (cp_phase * mu_phase / max(1e-12, k_phase)) if (cp_phase > 0 and mu_phase > 0 and k_phase > 0) else 1.0

        # Nusselt number using Dean-number enhanced model for serpentine curvature
        Nu_dean = compute_nusselt_dean(Re_m, Pr_phase, D_h, R_c_eff=R_c)
        # Apply thermal entrance-length development factor to reduce Nu if not fully developed
        if ENTRANCE_EFFECTS_ENABLED:
            if Re_m < 2300.0:
                L_th = 0.05 * max(Re_m, 1.0) * Pr_phase * D_h
            else:
                L_th = 10.0 * D_h
            dev_factor = min(1.0, L / max(L_th, 1e-12))
        else:
            dev_factor = 1.0
        # Combine with any user-provided serpentine enhancement (from data files), then clamp reasonably
        Nu_raw = eNu * Nu_dean * dev_factor
        Nu = float(np.clip(Nu_raw, 3.0, 200.0))
        h_l_curr = max(1e-12, Nu * k_phase / D_h)
        # Mild boiling enhancement in two-phase region to mimic nucleate boiling effects without full Chen model
        if 0.0 < x_loc < 1.0:
            e_boil = 1.0 + 4.0 * x_loc * (1.0 - x_loc)  # max 2.0 at x=0.5
            h_l_curr *= e_boil
            # Reduce effective h due to bubble coverage / intermittent contact; Dean mixing partially mitigates
            if BOILING_CONTACT_ENABLED:
                f_bubble = 1.0 - 0.50 * x_loc * (1.0 - x_loc)  # up to 50% area loss at x=0.5
                De_local = Re_m * np.sqrt(D_h / (2.0 * R_c)) if R_c > 0 else 0.0
                dean_mix_factor = 0.9 + 0.1 * np.tanh(De_local / 100.0)  # 0.9..1.0
                f_contact = max(CONTACT_MIN, f_bubble * dean_mix_factor)
                h_l_curr *= f_contact

        # convective conductance per heater area uses total wetted perimeter and heater area per length:
        conv_cond_per_heater_area = h_l_curr * (P_wet_total / a_per_length)   # [W/(m2_heater K)]

        # series conduction resistance per heater area (TBRs + SiN + Si + fouling + convection)
        R_conv = (1.0 / conv_cond_per_heater_area if conv_cond_per_heater_area > 0 else 1e12)
        R_foul = FOULING_RPP if FOULING_ENABLED else 0.0
        Rpp_total = (Rpp_tbr_mosin + Rpp_sin + Rpp_tbr_sinsi + Rpp_si + R_foul + R_conv)

        if wall_cap_enabled:
            # Legacy cap mode (kept for compatibility). Not recommended when using physical wall model.
            deltaT_avail = max(0.0, T_w - T_fluid_eff)
            qpp_cap = deltaT_avail / Rpp_total if Rpp_total > 0 else 0.0
            qpp_to_fluid = max(0.0, min(qpp_heater_global, qpp_cap))
            T_wall_local = T_fluid_eff + qpp_to_fluid * Rpp_total
        else:
            # Physical wall energy balance: solve for T_wall such that
            # q_elec = q_to_fluid + q_back + q_rad + q_ext
            # where terms are per heater area.
            # Backside resistance per area (substrate + contact)
            k_back = get_temp_dependent_k(max(T_fluid_eff, T_BASE_K))
            Rpp_back_sub = T_BACK_THICK / max(1e-12, k_back)
            Rpp_back_total = Rpp_back_sub + RPP_BACK_CONTACT

            qpp_elec = max(0.0, qpp_heater_global)

            def energy_balance(Tw):
                # to fluid via heater stack
                q_to_fluid = max(0.0, (Tw - T_fluid_eff) / max(1e-12, Rpp_total))
                # to backside sink
                q_back = max(0.0, (Tw - T_BASE_K) / max(1e-12, Rpp_back_total))
                # radiation from top surface
                q_rad = EPSILON_RAD * SIGMA_SB * (Tw**4 - T_AMBIENT_K**4)
                # external convection from top surface
                q_ext = H_EXT * (Tw - T_AMBIENT_K)
                return (q_to_fluid + q_back + q_rad + q_ext) - qpp_elec

            Tw_low = max(T_fluid_eff, T_AMBIENT_K)
            Tw_high = 2000.0
            try:
                T_wall_local = brentq(energy_balance, Tw_low, Tw_high, maxiter=100, xtol=1e-6)
            except ValueError:
                # Expand bracket if needed
                try:
                    T_wall_local = brentq(energy_balance, Tw_low, 4000.0, maxiter=100, xtol=1e-6)
                except Exception:
                    # Fallback (should rarely happen): assume all power to fluid
                    T_wall_local = T_fluid_eff + qpp_heater_global * Rpp_total

            # Once T_wall is known, only the portion to the fluid contributes to fluid heating
            qpp_to_fluid = max(0.0, (T_wall_local - T_fluid_eff) / max(1e-12, Rpp_total))
            # Apply maldistribution inefficiency so not all wall-to-fluid coupling is effective
            if MALDISTRIBUTION_ENABLED:
                qpp_to_fluid *= MALDISTRIBUTION_FACTOR
            qpp_elec = max(0.0, qpp_heater_global)
            # Clamp small numerical overshoot
            qpp_to_fluid = float(min(qpp_to_fluid, qpp_elec))
            # Use this as the heat that actually enters the fluid
            qpp_to_fluid = float(qpp_to_fluid)
            qpp_used = qpp_to_fluid
        # track wall temperature peak for diagnostics
        if np.isfinite(T_wall_local):
            T_wall_max = max(T_wall_max, T_wall_local)

        # energy available in this slice (W) from heater through thermal stack
        E_available = qpp_used * a_per_length * dx

        # Energy bookkeeping for subcooled -> saturation -> boiling -> superheat
        T_next = curr_T
        x_next = curr_x

        if curr_T < T_sat_curr and curr_x <= 1e-6:
            # Need sensible heat to reach saturation
            E_to_sat = mdot_total * cp_l * max(0.0, (T_sat_curr - curr_T))
            if E_available < E_to_sat:
                # remain subcooled
                dT = E_available / (mdot_total * cp_l) if (mdot_total > 0 and cp_l > 0) else 0.0
                T_next = curr_T + dT
                x_next = 0.0
            else:
                # reach saturation within slice; use excess for latent heat
                E_excess = E_available - E_to_sat
                dx_mass = E_excess / (mdot_total * h_fg) if (mdot_total > 0 and h_fg > 0) else 0.0
                x_next = float(np.clip(dx_mass, 0.0, 1.0))
                T_next = T_sat_curr
        elif curr_x < 1.0:
            # Two-phase: all energy to latent heat until dryout
            dx_mass = E_available / (mdot_total * h_fg) if (mdot_total > 0 and h_fg > 0) else 0.0
            x_next = float(np.clip(curr_x + dx_mass, 0.0, 1.0))
            T_next = T_sat_curr
        else:
            # Superheated vapor sensible heating
            dT = E_available / (mdot_total * cp_v) if (mdot_total > 0 and cp_v > 0) else 0.0
            T_target = curr_T + dT
            T_sat_local = T_sat_curr  # approximate using current pressure
            # Use dynamic wall temperature limit if cap disabled; else fixed HEATER_WALL_T
            wall_limit = T_w if wall_cap_enabled else T_wall_local
            if REMOVE_SUPERHEAT_CAP:
                T_next = min(T_target, wall_limit)
            else:
                T_next = min(T_target, T_sat_local + max_superheat, wall_limit)
            x_next = 1.0

        qpp_arr[i] = float(qpp_used)

        # update
        # Void fraction estimate from quality (homogeneous model): alpha = x * rho_m / rho_v
        if x_next <= 0.0:
            alpha_val = 0.0
        elif x_next >= 1.0:
            alpha_val = 1.0
        else:
            rho_mix_tmp = 1.0 / (x_next / rho_v_curr + (1.0 - x_next) / rho_l_curr)
            alpha_val = float(np.clip(x_next * (rho_mix_tmp / rho_v_curr), 0.0, 1.0))

        alpha[i+1] = alpha_val
        x_qual[i+1] = x_next
        T_b[i+1] = T_next
        # Avoid non-physical pressure collapse; small floor to keep numerics stable
        P[i+1] = max(5.0e4, curr_P + dP_dx * dx)  # Floor at 0.5 bar to avoid unrealistic near-vacuum unless modeled compressible expansion

        # Debug prints removed

    # total absorbed heat (W) across all heater area and axial integration
    # Use modern trapezoidal integration (np.trapezoid) to avoid deprecation warning
    Q_total = a_per_length * np.trapezoid(qpp_arr, x[:-1])
    return x, T_b, P, alpha, Q_total, x_qual, T_wall_max

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
    l_nd = params.get('l_nd', 0.0) * um
    A_t = w_t * depth
    A_e = w_nd * depth
    # Planar nozzle divergence half-angle approximation (flow discharge angle)
    if l_nd > 0 and w_nd > w_t:
        theta_e_rad = np.arctan((w_nd - w_t) / (2.0 * l_nd))
    else:
        theta_e_rad = 0.0
    theta_e_deg = np.degrees(theta_e_rad)
    # Per-nozzle discharge coefficient (optional in params)
    Cd = params.get('Cd', NOZZLE_CD_DEFAULT)
    return {'A_t': A_t, 'A_e': A_e, 'theta_e_rad': theta_e_rad, 'theta_e_deg': theta_e_deg, 'Cd': Cd}

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

def _mdot_choked_isentropic(P1, T1, A_t, gamma=1.33, R=R_v, Cd=NOZZLE_CD_DEFAULT):
    """
    Choked (sonic) mass flow at the throat for ideal gas isentropic model with discharge coefficient Cd.
    """
    if A_t <= 0 or P1 <= 0 or T1 <= 0:
        return 0.0
    base = (2.0 / (gamma + 1.0))**((gamma + 1.0) / (gamma - 1.0))
    mdot_star = (A_t * P1 * gamma * np.sqrt(base)) / np.sqrt(gamma * R * T1)
    return Cd * mdot_star

def isentropic_nozzle_performance(P1, T1, A_t, A_e, gamma=1.33, R=R_v, p_ambient=P_AMBIENT,
                                  mdot_override=None, Cd=NOZZLE_CD_DEFAULT,
                                  theta_e_rad: float = 0.0):
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
        return {'F':0.0,'Isp':0.0,'p_e':np.nan,'T_e':np.nan,'Ve':0.0,'mdot':0.0,'Me':np.nan,'mdot_choked':0.0,'choked':False,
                'theta_e_deg': np.degrees(theta_e_rad), 'Cd': Cd}

    # Compute area ratio from geometry; depth cancels, so this equals width_exit/width_throat
    Ae_At = A_e / A_t

    # Solve for supersonic exit Mach number using area-M relation
    Me = _solve_M_from_area_ratio(Ae_At, gamma, supersonic=True)

    # Choked mass flow with discharge coefficient
    mdot_choked = _mdot_choked_isentropic(P1, T1, A_t, gamma=gamma, R=R, Cd=Cd)

    # If override mass flow is provided (e.g., only vapor portion, or to avoid choking), use the lesser of override and choked
    if mdot_override is None:
        mdot = mdot_choked
    else:
        mdot = min(max(0.0, mdot_override), mdot_choked)

    # Exit temperature and pressure using isentropic relations
    Te = T1 * (1.0 + (gamma - 1.0) / 2.0 * Me**2)**-1.0
    pe = P1 * (1.0 + (gamma - 1.0) / 2.0 * Me**2)**(-gamma / (gamma - 1.0))

    # Exit velocity should use static temperature Te (ideal gas relation for local speed of sound times Mach)
    Ve = Me * np.sqrt(gamma * R * Te)

    # (Moved computation above to allow use for velocity)

    # Axial thrust reduction due to discharge angle (non-axial exhaust)
    axial_factor = np.cos(theta_e_rad)
    # Thrust including pressure term; only momentum term projected axially
    F = mdot * Ve * axial_factor + (pe - p_ambient) * A_e

    g0 = 9.80665
    Isp = F / (mdot * g0) if mdot > 0 else 0.0

    return {'F': F, 'Isp': Isp, 'p_e': pe, 'T_e': Te, 'Ve': Ve, 'mdot': mdot, 'Me': Me,
        'mdot_choked': mdot_choked, 'choked': abs(mdot - mdot_choked) < 1e-12,
        'theta_e_deg': np.degrees(theta_e_rad), 'Cd': Cd}

def nozzle_params_catalog():
        """Return the three nozzle types provided by user, with dimensions in micrometers."""
        return {
        # Assign per-nozzle Cd values (can be tuned with experiments or CFD). These are illustrative defaults.
        # Longer, smoother convergent/divergent (L) -> slightly higher Cd; wide/steeper (W) -> slightly lower Cd.
        'L': {'w_nd':500, 'l_nd':645, 'w_nc':3000, 'l_nc':2600, 'w_t':45, 'Cd': 0.92},
        'W': {'w_nd':780, 'l_nd':660, 'w_nc':3000, 'l_nc':1500, 'w_t':45, 'Cd': 0.88},
        'B': {'w_nd':500, 'l_nd':500, 'w_nc':3000, 'l_nc':1600, 'w_t':45, 'Cd': 0.90},
        }

# -------------------------
# Example run for both cases
# -------------------------
# Power selection now runs heater 1 and heater 2 separately (12 cases total)
V_applied = V_APPLIED
nozzle_types = nozzle_params_catalog()
results = []
# New: container for axial distributions and vaporization locations
dist_rows = []

case_id = 1
for heater_id in [1, 2]:
    W_heater = heater_power_from_voltage(V_applied, heater_id=heater_id)
    heater_label = f"H{heater_id}"
    print(f"\n=== Heater {heater_label} (Power={W_heater:.3f} W) ===")
    for A_module, r_in, r_out in zip(As_list, r_in_list, r_out_list):
        L_calc = L_fixed
        for nozzle_name, params in nozzle_types.items():
            mdot = MDOT_TOTAL  # total inlet mass flow stays constant

            x, T, P_arr, alpha, Q, x_qual, T_wall_max = march_annulus_multichannel(
                mdot, P_INLET, r_in, r_out, A_module, W_heater,
                L=L_calc, n_channels=n_channels, N=N_SLICES, T_in=T_INLET,
                wall_cap_enabled=WALL_CAP_ENABLED
            )

            P0 = P_arr[-1]
            T0 = T[-1]
            areas = nozzle_geometry_areas(params, depth=nozzle_depth_default)

            mdot_vapor = float(np.clip(x_qual[-1], 0.0, 1.0)) * mdot
            perf = isentropic_nozzle_performance(
                P1=P0, T1=T0, A_t=areas['A_t'], A_e=areas['A_e'], gamma=1.33, R=R_v, p_ambient=P_AMBIENT,
                mdot_override=mdot_vapor, Cd=areas['Cd'], theta_e_rad=areas['theta_e_rad']
            )

            P_in_W = W_heater
            mdot_mgs = perf['mdot'] * 1e6
            F_uN = perf['F'] * 1e6
            F_mN = perf['F'] * 1e3
            Isp_s = perf['Isp']
            p_bar = P0 / 1e5
            T_K = T0
            tau_uN_per_W = F_uN / P_in_W if P_in_W > 0 else 0.0
            tau_mN_per_W = F_mN / P_in_W if P_in_W > 0 else 0.0
            # Heating utilization: fraction of electrical power actually conducted to the fluid
            util = (Q / P_in_W) if P_in_W > 0 else 0.0
            limit_str = 'heater-limited' if util > 0.98 else 'thermal-stack-limited'

            A_mm2 = A_module * 1e6
            t_wall_um = r_in * 1e6
            D_o_um = r_out * 1e6

            print(f"Case {case_id}: Heater={heater_label}, A={A_mm2:.2f} mm², t_wall={t_wall_um:.1f} µm, D_o={D_o_um:.1f} µm, Nozzle={nozzle_name} (Cd={areas['Cd']:.2f}, theta={areas['theta_e_deg']:.1f}°)")
            print(f"  P={P_in_W:.2f} W, mdot_total={mdot*1e6:.2f} mg/s, x_exit={x_qual[-1]:.3f}, alpha_exit={alpha[-1]:.3f}, mdot_noz={mdot_mgs:.2f} mg/s, choked={perf['choked']}")
            print(f"  F={F_uN:.1f} µN, Isp={Isp_s:.1f} s, p={p_bar:.3f} bar, T={T_K:.1f} K, tau={tau_uN_per_W:.1f} µN/W")
            print(f"  Q_absorbed={Q:.2f} W ({util*100:.1f}% of heater power) -> {limit_str}")
            dT_wf_est = max(0.0, T_wall_max - T0)
            print(f"  chamber T0={T0:.1f} K, P0={P0 / 1e5:.2f} bar, T_wall_max≈{T_wall_max:.1f} K, ΔT_wall-fluid≈{dT_wf_est:.1f} K")

            # capture current case id before increment for per-slice rows
            current_case_id = case_id
            results.append({
                'case': case_id,
                'heater': heater_label,
                'nozzle': nozzle_name,
                'P_W': P_in_W,
                'mdot_total_mgs': mdot*1e6,
                'alpha_exit': alpha[-1],
                'x_exit': x_qual[-1],
                'mdot_nozzle_mgs': mdot_mgs,
                'F_mN': F_mN,
                'Isp_s': Isp_s,
                'p_bar': p_bar,
                'T_K': T_K,
                'tau_mN_per_W': tau_mN_per_W,
                'Me': perf['Me'],
                'choked': perf['choked'],
                'theta_deg': perf['theta_e_deg'],
                'Cd': perf['Cd'],
                'dP_kPa': float(P_arr[0] - P_arr[-1]) / 1e3,
                'A_mm2': A_mm2,
                't_wall_um': t_wall_um,
                'D_o_um': D_o_um,
                'Q_absorbed_W': Q,
                'utilization': util,
                'T_wall_max_K': T_wall_max,
                'dT_wall_fluid_est_K': dT_wf_est
            })
            # Compute vaporization region (where 0 < quality < 1). Use linear interpolation for boundaries.
            vap_start_m = np.nan
            vap_end_m = np.nan
            yq = np.asarray(x_qual, dtype=float)
            # Start: first crossing from <=0 to >0
            for ii in range(len(yq) - 1):
                if (yq[ii] <= 0.0) and (yq[ii+1] > 0.0):
                    dy = yq[ii+1] - yq[ii]
                    dx_seg = x[ii+1] - x[ii]
                    vap_start_m = x[ii] + (0.0 - yq[ii]) * dx_seg / (dy if dy != 0 else 1.0)
                    break
            # If not found but we do have two-phase samples, fallback to first in-slice position
            mask_tp = (yq > 0.0) & (yq < 1.0)
            if np.isnan(vap_start_m) and np.any(mask_tp):
                vap_start_m = float(x[np.argmax(mask_tp)])

            # End: first crossing from <1 to >=1
            for ii in range(len(yq) - 1):
                if (yq[ii] < 1.0) and (yq[ii+1] >= 1.0):
                    dy = yq[ii+1] - yq[ii]
                    dx_seg = x[ii+1] - x[ii]
                    vap_end_m = x[ii] + (1.0 - yq[ii]) * dx_seg / (dy if dy != 0 else 1.0)
                    break
            # If not found but we do have two-phase samples that never reach dryout, use last in-slice position
            if np.isnan(vap_end_m) and np.any(mask_tp):
                vap_end_m = float(x[np.where(mask_tp)[0][-1]])

            # Append per-position distribution rows for this case
            geom_label = 'Large' if D_o_um >= 200.0 else 'Small'
            for ii in range(len(x)):
                xi = float(x[ii])
                Ti = float(T[ii])
                Pi = float(P_arr[ii])
                qi = float(x_qual[ii])
                ai = float(alpha[ii])
                Tsati = float(get_T_sat(Pi))
                in_tp = (qi > 0.0) and (qi < 1.0)
                dist_rows.append({
                    'case': current_case_id,
                    'heater': heater_label,
                    'geom': geom_label,
                    'nozzle': nozzle_name,
                    'A_mm2': A_mm2,
                    't_wall_um': t_wall_um,
                    'D_o_um': D_o_um,
                    'x_m': xi,
                    'x_mm': xi * 1e3,
                    'T_K': Ti,
                    'T_sat_K': Tsati,
                    'P_Pa': Pi,
                    'p_bar': Pi / 1e5,
                    'quality': qi,
                    'alpha': ai,
                    'in_two_phase': in_tp,
                    'vap_start_m': vap_start_m,
                    'vap_end_m': vap_end_m,
                    'T_wall_max_K': T_wall_max
                })
            # Plot and save T(x) and P(x) for this case
            if MATPLOTLIB_AVAILABLE:
                try:
                    os.makedirs('plots', exist_ok=True)
                    x_mm = np.array(x) * 1e3
                    P_bar_arr = np.array(P_arr) / 1e5
                    fig, axT = plt.subplots(figsize=(7.5, 4.5))
                    axP = axT.twinx()
                    # Temperature (K)
                    axT.plot(x_mm, T, color='tab:red', label='Temperature (K)', linewidth=2)
                    axT.set_xlabel('Axial position x (mm)')
                    axT.set_ylabel('Temperature (K)', color='tab:red')
                    axT.tick_params(axis='y', labelcolor='tab:red')
                    # Pressure (bar)
                    axP.plot(x_mm, P_bar_arr, color='tab:blue', label='Pressure (bar)', linewidth=2, linestyle='--')
                    axP.set_ylabel('Pressure (bar)', color='tab:blue')
                    axP.tick_params(axis='y', labelcolor='tab:blue')
                    # Vaporization markers
                    if not np.isnan(vap_start_m):
                        xvs = vap_start_m * 1e3
                        axT.axvline(x=xvs, color='green', linestyle=':', linewidth=2, label='Boiling start')
                    if not np.isnan(vap_end_m):
                        xve = vap_end_m * 1e3
                        axT.axvline(x=xve, color='purple', linestyle=':', linewidth=2, label='Boiling end')
                        # Shade two-phase region if both are valid and start<end
                        if not np.isnan(vap_start_m) and vap_end_m > vap_start_m:
                            axT.axvspan(vap_start_m*1e3, vap_end_m*1e3, color='orange', alpha=0.15)
                    title = f"Case {current_case_id}: {heater_label}, {geom_label}, nozzle {nozzle_name}"
                    axT.set_title(title)
                    # Build a combined legend with case-specific placement
                    lines_T, labels_T = axT.get_legend_handles_labels()
                    lines_P, labels_P = axP.get_legend_handles_labels()
                    legend_loc = 'best'
                    bbox_anchor = None
                    # Cases 4-6: keep right side, move to upper right but below title
                    if 4 <= current_case_id <= 6:
                        legend_loc = 'upper right'
                        bbox_anchor = (0.6, 1)  # slightly below top of axes
                    # Cases 7-12: keep right side, place vertically centered
                    elif 7 <= current_case_id <= 12:
                        legend_loc = 'center right'
                        bbox_anchor = (1.0, 0.5)
                    if bbox_anchor is None:
                        axT.legend(lines_T + lines_P, labels_T + labels_P, loc=legend_loc)
                    else:
                        axT.legend(lines_T + lines_P, labels_T + labels_P, loc=legend_loc, bbox_to_anchor=bbox_anchor)
                    fname = f"plots/case_{current_case_id:02d}_{heater_label}_{geom_label}_{nozzle_name}.png"
                    plt.tight_layout()
                    fig.savefig(fname, dpi=200)
                    plt.close(fig)
                except Exception as e:
                    print(f"Warning: plotting failed for case {current_case_id}: {e}")
            case_id += 1

df = pd.DataFrame(results)

# Round and tidy columns for clearer printing
cols_order = ['case','heater','nozzle','P_W','mdot_total_mgs','alpha_exit','x_exit','mdot_nozzle_mgs',
              'F_mN','Isp_s','p_bar','T_K','tau_mN_per_W','Me','choked','theta_deg','Cd',
              'A_mm2','t_wall_um','D_o_um','dP_kPa','Q_absorbed_W','utilization','T_wall_max_K','dT_wall_fluid_est_K']
for c in cols_order:
    if c in df.columns:
        if c in ['P_W','A_mm2','t_wall_um','D_o_um','dP_kPa']:
            df[c] = df[c].astype(float).round(1)
        elif c in ['F_mN','tau_mN_per_W']:
            df[c] = df[c].astype(float).round(2)
        elif c in ['mdot_total_mgs','mdot_nozzle_mgs','p_bar']:
            df[c] = df[c].astype(float).round(2)
        elif c in ['T_K','Isp_s']:
            df[c] = df[c].astype(float).round(1)
        elif c in ['dT_wall_fluid_est_K']:
            df[c] = df[c].astype(float).round(1)
        elif c in ['Me','alpha_exit','x_exit']:
            df[c] = df[c].astype(float).round(3)

# Additional rounding for theta_deg to prevent line wrapping
if 'theta_deg' in df.columns:
    df['theta_deg'] = df['theta_deg'].astype(float).round(1)

# Add a simple geometry label for readability
df['geom'] = np.where(df['D_o_um'] >= 200.0, 'Large', 'Small')

# Order rows: Heater -> geom (Large first) -> nozzle (L,W,B) -> case
nozzle_cat = pd.CategoricalDtype(['L','W','B'], ordered=True)
df['nozzle'] = df['nozzle'].astype(nozzle_cat)
df['geom'] = pd.Categorical(df['geom'], categories=['Large','Small'], ordered=True)
df = df.sort_values(['heater','geom','nozzle','case']).reset_index(drop=True)

# Configure display to reduce wrapping in console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

# Compact columns to print in console clearly
cols_print = ['case','heater','geom','nozzle','P_W','Q_absorbed_W','utilization','T_wall_max_K','dT_wall_fluid_est_K',
              'mdot_total_mgs','mdot_nozzle_mgs','F_mN','tau_mN_per_W','Isp_s','p_bar','T_K','choked']

print("\nSummary (12 cases, grouped by heater) [P=W, mdot=mg/s, F=mN, p=bar, T=K, tau=mN/W]:")
for h in ['H1','H2']:
    dfh = df[df['heater'] == h]
    if not dfh.empty:
        print(f"\n-- Heater {h} --")
        print(dfh[cols_print].to_string(index=False))

# Save full and compact summaries to CSV for unambiguous review
df.to_csv('summary_full.csv', index=False)
df[cols_print].to_csv('summary_compact.csv', index=False)
print("\nSaved: summary_full.csv and summary_compact.csv")

# Save axial pressure and temperature distributions only to CSV
if len(dist_rows) > 0:
    df_profiles = pd.DataFrame(dist_rows)
    # Minimal columns: identifiers + axial position + pressure + temperature
    cols_profiles = ['case','heater','geom','nozzle','x_m','x_mm','P_Pa','p_bar','T_K']
    cols_profiles = [c for c in cols_profiles if c in df_profiles.columns]
    df_profiles = df_profiles[cols_profiles]
    df_profiles.to_csv('profiles_PT.csv', index=False)
    print("Saved: profiles_PT.csv (axial pressure and temperature distributions only)")