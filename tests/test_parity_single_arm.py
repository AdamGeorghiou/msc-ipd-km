"""
CRITICAL TEST: Parity between Python wrapper and R oracle.

This test proves the Python wrapper produces identical results to R.
This is the foundation of the entire dissertation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.wrapper.ipd_wrapper import preprocess, extract_ipd_single


def test_single_arm_parity():
    """
    Compare Python wrapper output vs R oracle output.
    They must match exactly (within numerical tolerance).
    """
    
    # Load R oracle result (ground truth)
    oracle_df = pd.read_csv('results/oracle_radio.csv')
    
    # Load input data
    curve = pd.read_csv('data/examples/radio_curve.csv')
    at_risk = pd.read_csv('data/examples/at_risk_radio.csv')
    
    # Generate Python result using wrapper
    prep = preprocess(
        curve_data=curve,
        at_risk_times=at_risk['time'].tolist(),
        at_risk_counts=at_risk['n_risk'].tolist(),
        survival_scale=100
    )
    
    python_df = extract_ipd_single(
        preprocessed_data=prep,
        arm_id=0,
        total_events=None
    )
    
    # CRITICAL ASSERTIONS
    
    # 1. Same number of patients
    assert len(oracle_df) == len(python_df), (
        f"Row count mismatch: Oracle={len(oracle_df)}, Python={len(python_df)}"
    )
    
    # 2. Same columns
    assert list(oracle_df.columns) == list(python_df.columns), (
        f"Column mismatch: Oracle={oracle_df.columns.tolist()}, "
        f"Python={python_df.columns.tolist()}"
    )
    
    # 3. Time values match (within numerical tolerance)
    time_match = np.allclose(
        oracle_df['time'].values,
        python_df['time'].values,
        atol=1e-6,
        rtol=1e-9
    )
    assert time_match, "Time values don't match between Oracle and Python"
    
    # 4. Status values match exactly (integer: 0 or 1)
    status_match = (oracle_df['status'].values == python_df['status'].values).all()
    assert status_match, "Status values don't match between Oracle and Python"
    
    # 5. Arm values match exactly
    arm_match = (oracle_df['arm'].values == python_df['arm'].values).all()
    assert arm_match, "Arm values don't match between Oracle and Python"
    
    # Additional checks
    
    # 6. Event counts match
    oracle_events = (oracle_df['status'] == 1).sum()
    python_events = (python_df['status'] == 1).sum()
    assert oracle_events == python_events, (
        f"Event count mismatch: Oracle={oracle_events}, Python={python_events}"
    )
    
    # 7. Censored counts match
    oracle_censored = (oracle_df['status'] == 0).sum()
    python_censored = (python_df['status'] == 0).sum()
    assert oracle_censored == python_censored, (
        f"Censored count mismatch: Oracle={oracle_censored}, Python={python_censored}"
    )
    
    max_diff = np.max(np.abs(oracle_df['time'].values - python_df['time'].values))

    print("\n" + "=" * 60)
    print("🎉 PARITY TEST PASSED!")
    print("=" * 60)
    print(f"Patients reconstructed: {len(python_df)}")
    print(f"Events: {python_events}")
    print(f"Censored: {python_censored}")
    print(f"Max time difference: {max_diff:.2e}")  # Scientific notation
    print("=" * 60)


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])