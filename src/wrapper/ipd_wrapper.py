"""
Python wrapper for IPDfromKM R package
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

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


def validate_reconstruction(
    original_curve: pd.DataFrame,
    reconstructed_ipd: pd.DataFrame,
    at_risk_reported: pd.DataFrame = None
) -> dict:
    """
    Calculate validation metrics for reconstructed IPD.
    
    Args:
        original_curve: Original digitized curve points
        reconstructed_ipd: IPD from extract_ipd_single()
        at_risk_reported: Reported at-risk numbers (optional)
        
    Returns:
        dict with keys: rmse, mae, max_error, ks_test_pvalue
    """
    # TODO: Implement later
    pass