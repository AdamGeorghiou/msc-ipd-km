"""
Test validation metrics implementation
Run this after adding the validation functions to ipd_wrapper.py
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.wrapper.ipd_wrapper import (
    preprocess, 
    extract_ipd_single, 
    validate_reconstruction,
    print_validation_summary
)

print("=" * 70)
print("TESTING VALIDATION METRICS")
print("=" * 70)

# Load input data
print("\n1. Loading data...")
curve = pd.read_csv('data/examples/radio_curve.csv')
at_risk = pd.read_csv('data/examples/at_risk_radio.csv')

print(f"   Original curve: {len(curve)} points")
print(f"   At-risk table: {len(at_risk)} time points")

# Reconstruct IPD
print("\n2. Reconstructing IPD...")
prep = preprocess(
    curve_data=curve,
    at_risk_times=at_risk['time'].tolist(),
    at_risk_counts=at_risk['n_risk'].tolist(),
    survival_scale=100
)

ipd = extract_ipd_single(prep, arm_id=0)
print(f"   Reconstructed: {len(ipd)} patients")

# Run validation
print("\n3. Running validation metrics...")
validation = validate_reconstruction(
    original_curve=curve,
    reconstructed_ipd=ipd
)

# Print results
print_validation_summary(validation)

# Show first few points of comparison
print("\n--- Sample Comparison (first 10 points) ---")
print(f"{'Time':>8} | {'Original':>10} | {'Reconstructed':>14} | {'Error':>8}")
print("-" * 50)

original_times = curve.iloc[:, 0].values[:10]
original_surv = curve.iloc[:, 1].values[:10]

# Interpolate reconstructed at these points
from src.wrapper.ipd_wrapper import interpolate_survival
reconstructed_surv = interpolate_survival(
    validation['reconstructed_curve'], 
    original_times
)

for t, orig, recon in zip(original_times, original_surv, reconstructed_surv):
    error = abs(orig - recon)
    print(f"{t:8.2f} | {orig:10.2f} | {recon:14.2f} | {error:8.4f}")

print("\n✅ Validation test complete!")

# Save reconstructed curve for plotting later
validation['reconstructed_curve'].to_csv(
    'results/reconstructed_curve_radio.csv',
    index=False
)
print("   Saved: results/reconstructed_curve_radio.csv")