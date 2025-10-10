"""
Generate all validation plots for the radiation data example.
Run this to create publication-quality figures for your dissertation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.wrapper.ipd_wrapper import (
    preprocess, 
    extract_ipd_single, 
    validate_reconstruction
)

from src.wrapper.plotting import (
    plot_km_overlay,
    plot_error_over_time,
    plot_validation_dashboard
)

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

print("=" * 70)
print("GENERATING VALIDATION PLOTS")
print("=" * 70)

# Load data
print("\n1. Loading data...")
curve = pd.read_csv('data/examples/radio_curve.csv')
at_risk = pd.read_csv('data/examples/at_risk_radio.csv')

# Reconstruct IPD
print("2. Reconstructing IPD...")
prep = preprocess(
    curve_data=curve,
    at_risk_times=at_risk['time'].tolist(),
    at_risk_counts=at_risk['n_risk'].tolist(),
    survival_scale=100
)
ipd = extract_ipd_single(prep, arm_id=0)

# Run validation
print("3. Running validation...")
validation = validate_reconstruction(
    original_curve=curve,
    reconstructed_ipd=ipd
)

# Get reconstructed curve
reconstructed_curve = validation['reconstructed_curve']

print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

# Figure 1: KM Curve Overlay
print("\n📊 Figure 1: KM Curve Overlay...")
fig1, ax1 = plot_km_overlay(
    original_curve=curve,
    reconstructed_curve=reconstructed_curve,
    validation_metrics=validation,
    at_risk_data=at_risk,
    title="Kaplan-Meier Curve Reconstruction: Radiation Data",
    save_path='figures/km_overlay_radio.png'
)
plt.close(fig1)

# Figure 2: Error Over Time
print("📊 Figure 2: Error Over Time...")
fig2, ax2 = plot_error_over_time(
    original_curve=curve,
    reconstructed_curve=reconstructed_curve,
    title="Reconstruction Error Over Time: Radiation Data",
    save_path='figures/error_over_time_radio.png'
)
plt.close(fig2)

# Figure 3: Comprehensive Dashboard
print("📊 Figure 3: Validation Dashboard...")
fig3 = plot_validation_dashboard(
    original_curve=curve,
    reconstructed_curve=reconstructed_curve,
    reconstructed_ipd=ipd,
    validation_metrics=validation,
    at_risk_data=at_risk,
    save_path='figures/validation_dashboard_radio.png'
)
plt.close(fig3)

print("\n" + "=" * 70)
print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nSaved figures:")
print("  1. figures/km_overlay_radio.png")
print("  2. figures/error_over_time_radio.png")
print("  3. figures/validation_dashboard_radio.png")
print("\nThese are publication-ready at 300 DPI!")
print("=" * 70)

# Optional: Show one of the plots
print("\nShowing KM overlay plot...")
fig, ax = plot_km_overlay(
    original_curve=curve,
    reconstructed_curve=reconstructed_curve,
    validation_metrics=validation,
    at_risk_data=at_risk,
    title="Kaplan-Meier Curve Reconstruction: Radiation Data"
)
plt.show()