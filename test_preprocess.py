"""Quick test of preprocess function"""

import pandas as pd
from src.wrapper.ipd_wrapper import preprocess

# Load data
curve = pd.read_csv('data/examples/radio_curve.csv')
at_risk = pd.read_csv('data/examples/at_risk_radio.csv')

print("Input data:")
print(f"  Curve shape: {curve.shape}")
print(f"  At-risk shape: {at_risk.shape}")

# Call preprocess
try:
    result = preprocess(
        curve_data=curve,
        at_risk_times=at_risk['time'].tolist(),
        at_risk_counts=at_risk['n_risk'].tolist(),
        survival_scale=100
    )
    
    print("\n✅ preprocess() successful!")
    print(f"  Curve points: {result['n_curve_points']}")
    print(f"  Intervals: {result['n_intervals']}")
    print(f"  R object type: {type(result['r_object'])}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()