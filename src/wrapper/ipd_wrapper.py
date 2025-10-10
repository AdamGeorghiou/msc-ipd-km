"""
Python wrapper for IPDfromKM R package
"""
#ipd_wrapper.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Import R package
IPD_PKG = importr('IPDfromKM')


def preprocess(
    curve_data: pd.DataFrame,
    at_risk_times: list,
    at_risk_counts: list,
    survival_scale: int = 100
) -> dict:
    """
    Preprocess KM curve data for IPD reconstruction.
    
    Args:
        curve_data: DataFrame with 2 columns [time, survival_prob]
        at_risk_times: List of time points where n_risk is reported
        at_risk_counts: List of patients at risk at each time point
        survival_scale: 100 if survival in %, 1 if in decimal (0-1)
        
    Returns:
        dict: Contains preprocessed R object and metadata
    """
    
    # Input validation
    if not isinstance(curve_data, pd.DataFrame):
        raise TypeError("curve_data must be a pandas DataFrame")
    
    if curve_data.shape[1] != 2:
        raise ValueError(f"curve_data must have 2 columns, got {curve_data.shape[1]}")
    
    if len(at_risk_times) != len(at_risk_counts):
        raise ValueError(
            f"at_risk_times ({len(at_risk_times)}) and "
            f"at_risk_counts ({len(at_risk_counts)}) must have same length"
        )
    
    # Convert pandas DataFrame to R dataframe using modern API
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_curve_data = ro.conversion.py2rpy(curve_data)
    
    # Convert Python lists to R vectors
    r_trisk = ro.FloatVector(at_risk_times)
    r_nrisk = ro.FloatVector(at_risk_counts)
    
    # Call R preprocess function
    try:
        preprocessed = IPD_PKG.preprocess(
            dat=r_curve_data,
            trisk=r_trisk,
            nrisk=r_nrisk,
            maxy=survival_scale
        )
    except Exception as e:
        raise RuntimeError(f"R preprocess() failed: {str(e)}")
    
    # Return the R object wrapped in a dict with metadata
    result = {
        'r_object': preprocessed,
        'n_curve_points': curve_data.shape[0],
        'n_intervals': len(at_risk_times),
        'survival_scale': survival_scale
    }
    
    return result


def extract_ipd_single(
    preprocessed_data: dict,
    arm_id: int = 0,
    total_events: int = None
) -> pd.DataFrame:
    """
    Extract individual patient data from preprocessed curve.
    
    Args:
        preprocessed_data: Output from preprocess()
        arm_id: Treatment arm identifier (0, 1, etc.)
        total_events: Total events (optional, improves accuracy)
        
    Returns:
        DataFrame with columns [time, status, arm]
    """
    
    # Input validation
    if not isinstance(preprocessed_data, dict):
        raise TypeError("preprocessed_data must be dict from preprocess()")
    
    if 'r_object' not in preprocessed_data:
        raise ValueError("preprocessed_data missing 'r_object' key")
    
    # Extract the R preprocessed object
    r_prep = preprocessed_data['r_object']
    
    # Convert arm_id to R integer
    r_arm_id = ro.IntVector([arm_id])[0]
    
    # Convert total_events to R (NULL if None)
    if total_events is None:
        r_tot_events = ro.NULL
    else:
        r_tot_events = ro.IntVector([total_events])[0]
    
    # Call R getIPD function
    try:
        ipd_result = IPD_PKG.getIPD(
            prep=r_prep,
            armID=r_arm_id,
            tot_events=r_tot_events
        )
    except Exception as e:
        raise RuntimeError(f"R getIPD() failed: {str(e)}")
    
    # Extract the IPD data frame from the result
    # R returns a list with $IPD element
    r_ipd_df = ipd_result.rx2('IPD')
    
    # Convert R dataframe to pandas
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        ipd_df = ro.conversion.rpy2py(r_ipd_df)
    
    # Ensure column names are correct
    # R returns: time, status, arm (or treatment)
    if ipd_df.shape[1] == 3:
        ipd_df.columns = ['time', 'status', 'arm']
    
    # Ensure correct data types
    ipd_df['time'] = ipd_df['time'].astype(float)
    ipd_df['status'] = ipd_df['status'].astype(int)
    ipd_df['arm'] = ipd_df['arm'].astype(int)
    
    return ipd_df


import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_km_curve(ipd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Kaplan-Meier survival curve from individual patient data.
    
    Args:
        ipd_df: DataFrame with columns [time, status, arm]
                status: 1=event, 0=censored
    
    Returns:
        DataFrame with columns [time, survival_prob]
    """
    # Sort by time
    sorted_data = ipd_df.sort_values('time').copy()
    
    # Get unique event times (only times where events occurred)
    event_times = sorted_data[sorted_data['status'] == 1]['time'].unique()
    event_times = np.sort(event_times)
    
    # Initialize
    n_patients = len(sorted_data)
    survival_prob = 1.0
    km_curve = []
    
    # Add starting point (time=0, survival=1.0)
    km_curve.append({'time': 0.0, 'survival_prob': 100.0})
    
    for t in event_times:
        # Number at risk just before time t
        at_risk = (sorted_data['time'] >= t).sum()
        
        # Number of events at exactly time t
        events = ((sorted_data['time'] == t) & (sorted_data['status'] == 1)).sum()
        
        # Kaplan-Meier formula: S(t) = S(t-1) * (1 - d_t/n_t)
        # where d_t = deaths at time t, n_t = at risk at time t
        if at_risk > 0:
            survival_prob *= (1 - events / at_risk)
        
        km_curve.append({
            'time': t,
            'survival_prob': survival_prob * 100  # Convert to percentage
        })
    
    return pd.DataFrame(km_curve)


def interpolate_survival(km_curve: pd.DataFrame, time_points: np.ndarray) -> np.ndarray:
    """
    Interpolate survival probabilities at specific time points.
    Uses step-function interpolation (appropriate for KM curves).
    
    Args:
        km_curve: DataFrame with [time, survival_prob]
        time_points: Array of times where we want survival estimates
    
    Returns:
        Array of survival probabilities at requested time points
    """
    # Use 'previous' method for step function (KM is right-continuous)
    interpolated = np.interp(
        time_points,
        km_curve['time'].values,
        km_curve['survival_prob'].values,
        left=100.0,  # Before first time point, survival = 100%
        right=km_curve['survival_prob'].iloc[-1]  # After last point, use last value
    )
    
    return interpolated


def validate_reconstruction(
    original_curve: pd.DataFrame,
    reconstructed_ipd: pd.DataFrame,
    at_risk_reported: pd.DataFrame = None
) -> dict:
    """
    Calculate validation metrics for reconstructed IPD.
    
    Compares the original digitized KM curve against a KM curve
    calculated from the reconstructed IPD.
    
    Args:
        original_curve: Original digitized curve points [time, survival_prob]
        reconstructed_ipd: IPD from extract_ipd_single() [time, status, arm]
        at_risk_reported: Reported at-risk numbers (optional, for future use)
        
    Returns:
        dict with keys:
            - rmse: Root Mean Square Error
            - mae: Mean Absolute Error  
            - max_error: Maximum absolute error
            - ks_statistic: Kolmogorov-Smirnov test statistic
            - ks_pvalue: KS test p-value
            - n_comparison_points: Number of points compared
            - reconstructed_curve: The KM curve calculated from IPD
    """
    
    # Input validation
    if not isinstance(original_curve, pd.DataFrame):
        raise TypeError("original_curve must be a pandas DataFrame")
    
    if not isinstance(reconstructed_ipd, pd.DataFrame):
        raise TypeError("reconstructed_ipd must be a pandas DataFrame")
    
    if original_curve.shape[1] < 2:
        raise ValueError("original_curve must have at least 2 columns [time, survival_prob]")
    
    required_cols = ['time', 'status']
    if not all(col in reconstructed_ipd.columns for col in required_cols):
        raise ValueError(f"reconstructed_ipd must contain columns: {required_cols}")
    
    # Step 1: Calculate KM curve from reconstructed IPD
    reconstructed_curve = calculate_km_curve(reconstructed_ipd)
    
    # Step 2: Get time points from original curve
    original_times = original_curve.iloc[:, 0].values  # First column is time
    original_surv = original_curve.iloc[:, 1].values   # Second column is survival
    
    # Step 3: Interpolate reconstructed curve at original time points
    reconstructed_surv = interpolate_survival(reconstructed_curve, original_times)
    
    # Step 4: Calculate metrics
    
    # RMSE - Root Mean Square Error
    # Target: ≤ 0.05 (on 0-100 scale, so ≤ 5% error)
    rmse = np.sqrt(mean_squared_error(original_surv, reconstructed_surv))
    
    # MAE - Mean Absolute Error
    # Target: ≤ 0.02 (on 0-100 scale, so ≤ 2% error)
    mae = mean_absolute_error(original_surv, reconstructed_surv)
    
    # Max Absolute Error
    # Target: ≤ 0.05 (on 0-100 scale, so ≤ 5% error)
    max_error = np.max(np.abs(original_surv - reconstructed_surv))
    
    # Kolmogorov-Smirnov Test
    # Tests if two distributions are the same
    # Target: p-value > 0.05 (fail to reject null hypothesis = distributions are similar)
    ks_statistic, ks_pvalue = ks_2samp(original_surv, reconstructed_surv)
    
    # Compile results
    results = {
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue,
        'n_comparison_points': len(original_times),
        'reconstructed_curve': reconstructed_curve,
        'passes_rmse': rmse <= 5.0,  # ≤ 5% on 0-100 scale
        'passes_mae': mae <= 2.0,     # ≤ 2% on 0-100 scale
        'passes_max_error': max_error <= 5.0,  # ≤ 5% on 0-100 scale
        'passes_ks_test': ks_pvalue > 0.05
    }
    
    return results


def print_validation_summary(validation_results: dict):
    """
    Print a nicely formatted summary of validation results.
    
    Args:
        validation_results: Output from validate_reconstruction()
    """
    results = validation_results
    
    print("\n" + "=" * 70)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 70)
    
    print(f"\nComparison points: {results['n_comparison_points']}")
    
    print("\n--- Accuracy Metrics ---")
    print(f"RMSE:              {results['rmse']:.4f} % {'✅ PASS' if results['passes_rmse'] else '❌ FAIL'} (target: ≤ 5.0%)")
    print(f"MAE:               {results['mae']:.4f} % {'✅ PASS' if results['passes_mae'] else '❌ FAIL'} (target: ≤ 2.0%)")
    print(f"Max Absolute Error: {results['max_error']:.4f} % {'✅ PASS' if results['passes_max_error'] else '❌ FAIL'} (target: ≤ 5.0%)")
    
    print("\n--- Statistical Test ---")
    print(f"KS Statistic:      {results['ks_statistic']:.6f}")
    print(f"KS p-value:        {results['ks_pvalue']:.6f} {'✅ PASS' if results['passes_ks_test'] else '❌ FAIL'} (target: > 0.05)")
    
    # Overall pass/fail
    all_pass = all([
        results['passes_rmse'],
        results['passes_mae'],
        results['passes_max_error'],
        results['passes_ks_test']
    ])
    
    print("\n" + "=" * 70)
    if all_pass:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
    else:
        print("⚠️  Some validation checks failed - review metrics above")
    print("=" * 70 + "\n")