"""
Sensitivity Analysis: Number of Axial Slices

This script runs the micropropulsion solver with varying numbers of axial slices
to determine when the solution converges. Convergence is assessed based on:
  - Exit temperature (T_exit)
  - Exit pressure (P_exit)
  - Exit vapor quality (x_exit)
  - Total heat absorbed (Q_absorbed)

The analysis stops when the relative change in all key metrics falls below
a specified tolerance for consecutive refinements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import os

# Import the solver functions from the main file
# We need to modify sys.path to import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required functions and constants from the solver
from importlib import import_module

# Since the filename has spaces, we need to use importlib
import importlib.util
solver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver working V3.py")
spec = importlib.util.spec_from_file_location("solver", solver_path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

# =============================================================================
# SENSITIVITY ANALYSIS CONFIGURATION
# =============================================================================
# Range of slice counts to test (start small, increase geometrically)
N_SLICES_LIST = [10, 20, 50, 100, 150, 200, 300, 400, 500, 750, 1000]

# Convergence tolerance (relative change threshold)
CONVERGENCE_TOL = 0.005  # 0.5% relative change

# Number of consecutive points that must satisfy convergence
CONSECUTIVE_CONVERGED = 2

# Which geometry case to use for sensitivity (index into width_list)
# 0 = Large (212 µm), 1 = Small (40 µm)
GEOMETRY_CASE = 1

# Which heater to use (1 or 2)
HEATER_ID = 1

# =============================================================================
# RUN SENSITIVITY ANALYSIS
# =============================================================================
def run_sensitivity_analysis():
    """
    Run the solver with different numbers of slices and track key outputs.
    """
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: Number of Axial Slices")
    print("=" * 70)
    
    # Get geometry and operating parameters from solver
    width = solver.CHANNEL_WIDTH_LIST[GEOMETRY_CASE]
    depth = solver.CHANNEL_DEPTH
    A_module = solver.HEATER_A_LIST[GEOMETRY_CASE]
    W_heater = solver.heater_power_from_voltage(solver.V_APPLIED, heater_id=HEATER_ID)
    mdot = solver.MDOT_TOTAL
    
    geom_label = "Large" if GEOMETRY_CASE == 0 else "Small"
    print(f"\nConfiguration:")
    print(f"  Geometry: {geom_label} (width={width*1e6:.1f} µm, depth={depth*1e6:.1f} µm)")
    print(f"  Heater: H{HEATER_ID} (Power={W_heater:.2f} W)")
    print(f"  Mass flow: {mdot*1e6:.2f} mg/s")
    print(f"  Convergence tolerance: {CONVERGENCE_TOL*100:.2f}%")
    print("-" * 70)
    
    # Storage for results
    results = []
    
    for N in N_SLICES_LIST:
        print(f"  Running with N_slices = {N:4d} ... ", end="", flush=True)
        
        # Run the solver
        x, T_b, P_arr, alpha, Q_total, x_qual, T_wall_max = solver.march_annulus_multichannel(
            mdot_total=mdot,
            P_in=solver.P_INLET,
            width=width,
            depth=depth,
            A_module=A_module,
            W_total=W_heater,
            L=solver.L_fixed,
            n_channels=solver.N_CHANNELS,
            t_sin=solver.t_sin_default,
            k_sin=solver.k_sin_default,
            t_si=solver.t_si_default,
            T_in=solver.T_INLET,
            N=N,
            wall_cap_enabled=solver.WALL_CAP_ENABLED,
            max_superheat=solver.MAX_SUPERHEAT,
            debug=False
        )
        
        # Extract key metrics
        T_exit = T_b[-1]
        P_exit = P_arr[-1]
        x_exit = x_qual[-1]
        
        results.append({
            'N_slices': N,
            'T_exit_K': T_exit,
            'P_exit_Pa': P_exit,
            'P_exit_bar': P_exit / 1e5,
            'x_exit': x_exit,
            'Q_absorbed_W': Q_total,
            'T_wall_max_K': T_wall_max
        })
        
        print(f"T_exit={T_exit:.2f} K, P_exit={P_exit/1e5:.3f} bar, x_exit={x_exit:.4f}, Q={Q_total:.3f} W")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate relative changes from previous value
    df['dT_rel'] = df['T_exit_K'].pct_change().abs()
    df['dP_rel'] = df['P_exit_Pa'].pct_change().abs()
    df['dx_rel'] = df['x_exit'].pct_change().abs()
    df['dQ_rel'] = df['Q_absorbed_W'].pct_change().abs()
    
    # Maximum relative change across all metrics
    df['max_rel_change'] = df[['dT_rel', 'dP_rel', 'dx_rel', 'dQ_rel']].max(axis=1)
    
    # Check convergence
    df['converged'] = df['max_rel_change'] < CONVERGENCE_TOL
    
    # Find first point where convergence is achieved (with consecutive check)
    converged_idx = None
    for i in range(1, len(df) - CONSECUTIVE_CONVERGED + 1):
        if all(df['converged'].iloc[i:i + CONSECUTIVE_CONVERGED]):
            converged_idx = i
            break
    
    print("-" * 70)
    if converged_idx is not None:
        N_converged = df.iloc[converged_idx]['N_slices']
        print(f"\n✓ CONVERGENCE ACHIEVED at N_slices = {int(N_converged)}")
        print(f"  (All metrics changed < {CONVERGENCE_TOL*100:.2f}% for {CONSECUTIVE_CONVERGED} consecutive refinements)")
    else:
        print(f"\n✗ Convergence NOT achieved within tested range.")
        print(f"  Consider extending N_SLICES_LIST or relaxing tolerance.")
    
    # Save results to CSV
    output_csv = "sensitivity_slices_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    return df, converged_idx


def create_convergence_plot(df, converged_idx):
    """
    Create an attractive, multi-panel convergence plot.
    """
    # Set up the figure with a clean style and a tighter aspect to avoid extra white space
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 8))
    
    # Color palette (colorblind-friendly)
    colors = {
        'T': '#E74C3C',      # Red
        'P': '#3498DB',      # Blue
        'x': '#2ECC71',      # Green
        'Q': '#9B59B6',      # Purple
        'max': '#34495E'     # Dark gray
    }
    
    N = df['N_slices'].values
    
    # =========================================================================
    # Panel 1: Absolute values (2x2 subplot in top half)
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(N, df['T_exit_K'], 'o-', color=colors['T'], linewidth=2, markersize=8, label='Exit Temperature')
    ax1.set_xlabel('Number of Axial Slices', fontsize=11)
    ax1.set_ylabel('Exit Temperature (K)', fontsize=11, color=colors['T'])
    ax1.tick_params(axis='y', labelcolor=colors['T'])
    ax1.set_title('(a) Exit Temperature Convergence', fontsize=12, fontweight='bold')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    if converged_idx is not None:
        ax1.axvline(x=df.iloc[converged_idx]['N_slices'], color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Converged (N={int(df.iloc[converged_idx]["N_slices"])})')
        ax1.legend(loc='best', fontsize=9)
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(N, df['P_exit_bar'], 's-', color=colors['P'], linewidth=2, markersize=8, label='Exit Pressure')
    ax2.set_xlabel('Number of Axial Slices', fontsize=11)
    ax2.set_ylabel('Exit Pressure (bar)', fontsize=11, color=colors['P'])
    ax2.tick_params(axis='y', labelcolor=colors['P'])
    ax2.set_title('(b) Exit Pressure Convergence', fontsize=12, fontweight='bold')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    if converged_idx is not None:
        ax2.axvline(x=df.iloc[converged_idx]['N_slices'], color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.legend(loc='best', fontsize=9)
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(N, df['x_exit'], '^-', color=colors['x'], linewidth=2, markersize=8, label='Exit Vapor Quality')
    ax3.set_xlabel('Number of Axial Slices', fontsize=11)
    ax3.set_ylabel('Exit Vapor Quality (-)', fontsize=11, color=colors['x'])
    ax3.tick_params(axis='y', labelcolor=colors['x'])
    ax3.set_title('(c) Exit Vapor Quality Convergence', fontsize=12, fontweight='bold')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    if converged_idx is not None:
        ax3.axvline(x=df.iloc[converged_idx]['N_slices'], color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.legend(loc='best', fontsize=9)
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(N, df['Q_absorbed_W'], 'D-', color=colors['Q'], linewidth=2, markersize=8, label='Heat Absorbed')
    ax4.set_xlabel('Number of Axial Slices', fontsize=11)
    ax4.set_ylabel('Heat Absorbed (W)', fontsize=11, color=colors['Q'])
    ax4.tick_params(axis='y', labelcolor=colors['Q'])
    ax4.set_title('(d) Heat Absorbed Convergence', fontsize=12, fontweight='bold')
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    if converged_idx is not None:
        ax4.axvline(x=df.iloc[converged_idx]['N_slices'], color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax4.legend(loc='best', fontsize=9)
    
    # Tight layout without reserving extra bottom margin (no lower panel here)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Main title
    geom_label = "Large" if GEOMETRY_CASE == 0 else "Small"
    fig.suptitle(f'Mesh Sensitivity Analysis — {geom_label} Geometry, Heater H{HEATER_ID}', 
                fontsize=14, fontweight='bold', y=0.985)
    
    # Save figure
    output_png = "plots/sensitivity_slices_analysis.png"
    os.makedirs("plots", exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_png}")
    
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n")
    df_results, conv_idx = run_sensitivity_analysis()
    print("\n")
    create_convergence_plot(df_results, conv_idx)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    display_cols = ['N_slices', 'T_exit_K', 'P_exit_bar', 'x_exit', 'Q_absorbed_W', 'max_rel_change']
    df_display = df_results[display_cols].copy()
    df_display['max_rel_change'] = df_display['max_rel_change'].apply(lambda x: f"{x*100:.3f}%" if pd.notna(x) else "—")
    df_display.columns = ['N_slices', 'T_exit (K)', 'P_exit (bar)', 'x_exit', 'Q (W)', 'Max Δ (%)']
    print(df_display.to_string(index=False))
    print("=" * 70)
