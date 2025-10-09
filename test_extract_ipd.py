"""Test full pipeline: preprocess -> extract_ipd_single"""

import pandas as pd
from src.wrapper.ipd_wrapper import preprocess, extract_ipd_single

# Load data
curve = pd.read_csv('data/examples/radio_curve.csv')
at_risk = pd.read_csv('data/examples/at_risk_radio.csv')

print("=" * 60)
print("Testing Full Pipeline")
print("=" * 60)

# Step 1: Preprocess
print("\n1. Preprocessing...")
prep = preprocess(
    curve_data=curve,
    at_risk_times=at_risk['time'].tolist(),
    at_risk_counts=at_risk['n_risk'].tolist(),
    survival_scale=100
)
print("✅ Preprocessing complete")

# Step 2: Extract IPD
print("\n2. Extracting IPD...")
ipd = extract_ipd_single(
    preprocessed_data=prep,
    arm_id=0,
    total_events=None
)
print("✅ IPD extraction complete")

# Display results
print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"Reconstructed patients: {len(ipd)}")
print(f"Events: {(ipd['status'] == 1).sum()}")
print(f"Censored: {(ipd['status'] == 0).sum()}")

print("\nFirst 10 patient records:")
print(ipd.head(10))

print("\nData types:")
print(ipd.dtypes)

print("\nSummary statistics:")
print(ipd.describe())

# Save output
output_path = 'results/python_radio.csv'
ipd.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")