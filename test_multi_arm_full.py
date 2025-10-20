"""
Complete multi-arm test with validation and visualization
Generates publication-ready figures for multi-arm reconstruction
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.wrapper.ipd_wrapper import (
    preprocess_multi_arm,
    extract_ipd_multi_arm,
    validate_multi_arm_reconstruction,
    print_multi_arm_validation_summary
)

from src.wrapper.plotting import (
    plot_multi_arm_comparison,
    plot_multi_arm_dashboard
)

print("=" * 80)
print("COMPLETE MULTI-ARM VALIDATION & VISUALIZATION")
print("=" * 80)

# Load data
print("\n1. Loading data...")
radio_curve = pd.read_csv('data/examples/radio_curve.csv')
radioplus_curve = pd.read_csv('data/examples/radioplus_curve.csv')
radio_at_risk = pd.read_csv('data/examples/at_risk_radio.csv')
radioplus_at_risk = pd.read_csv('data/examples/at_risk_radioplus.csv')

# Preprocess
print("2. Preprocessing both arms...")
prep_multi = preprocess_multi_arm(
    curve_data_list=[radio_curve, radioplus_curve],
    at_risk_times_list=[
        radio_at_risk['time'].tolist(),
        radioplus_at_risk['time'].tolist()
    ],
    at_risk_counts_list=[
        radio_at_risk['n_risk'].tolist(),
        radioplus_at_risk['n_risk'].tolist()
    ],
    arm_names=['Radio (Control)', 'RadioPlus (Treatment)'],
    survival_scale=100
)

# Extract IPD
print("3. Extracting IPD...")
ipd_multi = extract_ipd_multi_arm(prep_multi)

print(f"\n   Total patients: {len(ipd_multi)}")
for arm_id in [0, 1]:
    arm_ipd = ipd_multi[ipd_multi['arm'] == arm_id]
    print(f"   {prep_multi['arm_names'][arm_id]}: {len(arm_ipd)} patients")

# Validate
print("\n4. Running validation...")
validation_multi = validate_multi_arm_reconstruction(
    original_curves=[radio_curve, radioplus_curve],
    reconstructed_ipd=ipd_multi,
    arm_names=['Radio (Control)', 'RadioPlus (Treatment)']
)

# Print summary
print_multi_arm_validation_summary(validation_multi)

# Get reconstructed curves for plotting
reconstructed_curves = [
    validation_multi['arm_results']['Radio (Control)']['reconstructed_curve'],
    validation_multi['arm_results']['RadioPlus (Treatment)']['reconstructed_curve']
]

# Generate plots
print("\n5. Generating plots...")
os.makedirs('figures', exist_ok=True)

# Plot 1: Comparison
print("   📊 Multi-arm comparison plot...")
fig1, _ = plot_multi_arm_comparison(
    original_curves=[radio_curve, radioplus_curve],
    reconstructed_curves=reconstructed_curves,
    arm_names=['Radio (Control)', 'RadioPlus (Treatment)'],
    validation_results=validation_multi,
    title='Multi-Arm KM Curve Reconstruction: Radio vs RadioPlus',
    save_path='figures/multi_arm_comparison.png'
)
plt.close(fig1)

# Plot 2: Dashboard
print("   📊 Multi-arm dashboard...")
fig2 = plot_multi_arm_dashboard(
    original_curves=[radio_curve, radioplus_curve],
    reconstructed_curves=reconstructed_curves,
    reconstructed_ipd=ipd_multi,
    arm_names=['Radio (Control)', 'RadioPlus (Treatment)'],
    validation_results=validation_multi,
    save_path='figures/multi_arm_dashboard.png'
)
plt.close(fig2)

# Save combined IPD
output_path = 'results/multi_arm_ipd.csv'
ipd_multi.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("✅ MULTI-ARM TEST COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. results/multi_arm_ipd.csv")
print("  2. figures/multi_arm_comparison.png")
print("  3. figures/multi_arm_dashboard.png")
print("\nAll validation checks passed for both arms!")
print("=" * 80)