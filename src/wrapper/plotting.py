"""
Plotting functions for visualizing KM curve reconstruction accuracy.

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional 


def plot_km_overlay(
    original_curve: pd.DataFrame,
    reconstructed_curve: pd.DataFrame,
    validation_metrics: Optional[Dict] = None,
    at_risk_data: Optional[pd.DataFrame] = None,
    title: str = "Kaplan-Meier Curve: Original vs Reconstructed",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Create overlay plot comparing original and reconstructed KM curves.
    
    Args:
        original_curve: DataFrame with [time, survival_prob]
        reconstructed_curve: DataFrame with [time, survival_prob] from calculate_km_curve()
        validation_metrics: Optional dict from validate_reconstruction()
        at_risk_data: Optional DataFrame with [time, n_risk] for numbers at risk
        title: Plot title
        save_path: Path to save figure (e.g., 'figures/km_overlay.png')
        figsize: Figure size (width, height)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original curve (digitized from paper)
    ax.plot(
        original_curve.iloc[:, 0], 
        original_curve.iloc[:, 1],
        'o-',
        color='#2E86AB',
        linewidth=2,
        markersize=3,
        label='Original (Digitized)',
        alpha=0.7
    )
    
    # Plot reconstructed curve (from IPD)
    ax.plot(
        reconstructed_curve['time'],
        reconstructed_curve['survival_prob'],
        '--',
        color='#A23B72',
        linewidth=2.5,
        label='Reconstructed (from IPD)',
        alpha=0.8
    )
    
    # Styling
    ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    # Add validation metrics text box if provided
    if validation_metrics:
        metrics_text = (
            f"Validation Metrics:\n"
            f"RMSE: {validation_metrics['rmse']:.3f}%\n"
            f"MAE: {validation_metrics['mae']:.3f}%\n"
            f"Max Error: {validation_metrics['max_error']:.3f}%\n"
            f"KS p-value: {validation_metrics['ks_pvalue']:.4f}"
        )
        
        # Determine pass/fail color
        all_pass = all([
            validation_metrics.get('passes_rmse', False),
            validation_metrics.get('passes_mae', False),
            validation_metrics.get('passes_max_error', False),
            validation_metrics.get('passes_ks_test', False)
        ])
        
        box_color = '#D4EDDA' if all_pass else '#F8D7DA'
        text_color = '#155724' if all_pass else '#721C24'
        
        ax.text(
            0.98, 0.02,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor=text_color),
            color=text_color,
            family='monospace'
        )
    
    # Add numbers at risk table if provided
    if at_risk_data is not None:
        add_at_risk_table(fig, ax, at_risk_data)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    return fig, ax


def add_at_risk_table(fig, ax, at_risk_data: pd.DataFrame):
    """
    Add numbers at risk table below the main plot.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        at_risk_data: DataFrame with [time, n_risk]
    """
    # Get the time points and counts
    times = at_risk_data['time'].values
    n_risk = at_risk_data['n_risk'].values
    
    # Create table data
    table_data = [
        [f"{int(n)}" for n in n_risk]
    ]
    col_labels = [f"{t:.1f}" for t in times]
    row_labels = ['At Risk']
    
    # Add table below plot
    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='bottom',
        bbox=[0, -0.15, 1, 0.08],
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color header
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(weight='bold')
        if j == -1:  # Row label
            cell.set_facecolor('#F0F0F0')
            cell.set_text_props(weight='bold')


def plot_error_over_time(
    original_curve: pd.DataFrame,
    reconstructed_curve: pd.DataFrame,
    title: str = "Reconstruction Error Over Time",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot the absolute error between original and reconstructed curves over time.
    
    Args:
        original_curve: DataFrame with [time, survival_prob]
        reconstructed_curve: DataFrame with [time, survival_prob]
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    # Interpolate reconstructed curve at original time points
    from scipy.interpolate import interp1d
    
    interp_func = interp1d(
        reconstructed_curve['time'],
        reconstructed_curve['survival_prob'],
        kind='previous',  # Step function
        bounds_error=False,
        fill_value=(100, reconstructed_curve['survival_prob'].iloc[-1])
    )
    
    original_times = original_curve.iloc[:, 0].values
    original_surv = original_curve.iloc[:, 1].values
    reconstructed_surv = interp_func(original_times)
    
    # Calculate errors
    errors = np.abs(original_surv - reconstructed_surv)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(original_times, errors, 'o-', color='#E63946', linewidth=2, markersize=4)
    ax.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, label='5% threshold')
    ax.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5, label='2% threshold')
    
    ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Add statistics
    stats_text = (
        f"Mean Error: {np.mean(errors):.3f}%\n"
        f"Max Error: {np.max(errors):.3f}%\n"
        f"Std Dev: {np.std(errors):.3f}%"
    )
    
    ax.text(
        0.98, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        family='monospace'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    return fig, ax


def plot_validation_dashboard(
    original_curve: pd.DataFrame,
    reconstructed_curve: pd.DataFrame,
    reconstructed_ipd: pd.DataFrame,
    validation_metrics: Dict,
    at_risk_data: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive dashboard with multiple validation plots.
    
    Args:
        original_curve: Original digitized curve
        reconstructed_curve: Reconstructed KM curve
        reconstructed_ipd: The individual patient data
        validation_metrics: Validation metrics dict
        at_risk_data: Optional at-risk table
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. KM curve overlay (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(original_curve.iloc[:, 0], original_curve.iloc[:, 1], 
             'o-', color='#2E86AB', linewidth=2, markersize=3, label='Original', alpha=0.7)
    ax1.plot(reconstructed_curve['time'], reconstructed_curve['survival_prob'],
             '--', color='#A23B72', linewidth=2.5, label='Reconstructed', alpha=0.8)
    ax1.set_xlabel('Time (months)', fontweight='bold')
    ax1.set_ylabel('Survival Probability (%)', fontweight='bold')
    ax1.set_title('KM Curve Overlay', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. Error over time (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    from scipy.interpolate import interp1d
    interp_func = interp1d(
        reconstructed_curve['time'], reconstructed_curve['survival_prob'],
        kind='previous', bounds_error=False, fill_value=(100, reconstructed_curve['survival_prob'].iloc[-1])
    )
    original_times = original_curve.iloc[:, 0].values
    original_surv = original_curve.iloc[:, 1].values
    errors = np.abs(original_surv - interp_func(original_times))
    ax2.plot(original_times, errors, 'o-', color='#E63946', linewidth=2, markersize=4)
    ax2.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (months)', fontweight='bold')
    ax2.set_ylabel('Absolute Error (%)', fontweight='bold')
    ax2.set_title('Reconstruction Error', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Event distribution histogram (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    events = reconstructed_ipd[reconstructed_ipd['status'] == 1]['time']
    censored = reconstructed_ipd[reconstructed_ipd['status'] == 0]['time']
    ax3.hist(events, bins=20, alpha=0.7, color='#E63946', label=f'Events (n={len(events)})')
    ax3.hist(censored, bins=20, alpha=0.7, color='#2E86AB', label=f'Censored (n={len(censored)})')
    ax3.set_xlabel('Time (months)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Event and Censoring Distribution', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Metrics summary (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    metrics_text = f"""
    VALIDATION METRICS
    {'='*40}
    
    Accuracy Metrics:
      • RMSE:          {validation_metrics['rmse']:>8.4f} %  {'✓' if validation_metrics['passes_rmse'] else '✗'}
      • MAE:           {validation_metrics['mae']:>8.4f} %  {'✓' if validation_metrics['passes_mae'] else '✗'}
      • Max Error:     {validation_metrics['max_error']:>8.4f} %  {'✓' if validation_metrics['passes_max_error'] else '✗'}
    
    Statistical Test:
      • KS Statistic:  {validation_metrics['ks_statistic']:>8.6f}
      • KS p-value:    {validation_metrics['ks_pvalue']:>8.6f}  {'✓' if validation_metrics['passes_ks_test'] else '✗'}
    
    Reconstruction Summary:
      • Patients:      {len(reconstructed_ipd):>8} 
      • Events:        {(reconstructed_ipd['status']==1).sum():>8}
      • Censored:      {(reconstructed_ipd['status']==0).sum():>8}
      • Curve Points:  {validation_metrics['n_comparison_points']:>8}
    
    {'='*40}
    Status: {'ALL CHECKS PASSED ✓' if all([validation_metrics['passes_rmse'], validation_metrics['passes_mae'], validation_metrics['passes_max_error'], validation_metrics['passes_ks_test']]) else 'SOME CHECKS FAILED ✗'}
    """
    
    ax4.text(0.1, 0.95, metrics_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.9, edgecolor='#CED4DA'))
    
    plt.suptitle('IPD Reconstruction Validation Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved: {save_path}")
    
    return fig




def plot_multi_arm_comparison(
    original_curves: List[pd.DataFrame],
    reconstructed_curves: List[pd.DataFrame],
    arm_names: List[str],
    validation_results: Optional[Dict] = None,
    title: str = "Multi-Arm KM Curve Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8)
):
    """
    Plot comparison of original vs reconstructed curves for multiple arms.
    
    Args:
        original_curves: List of original digitized curves
        reconstructed_curves: List of reconstructed curves
        arm_names: List of arm names
        validation_results: Optional validation results dict
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Left panel: Original curves
    for i, (curve, name) in enumerate(zip(original_curves, arm_names)):
        color = colors[i % len(colors)]
        ax1.plot(
            curve.iloc[:, 0], 
            curve.iloc[:, 1],
            'o-',
            color=color,
            linewidth=2,
            markersize=3,
            label=name,
            alpha=0.7
        )
    
    ax1.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Survival Probability (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Original Digitized Curves', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=11)
    ax1.set_ylim(0, 105)
    
    # Right panel: Overlay (original + reconstructed)
    for i, (orig, recon, name) in enumerate(zip(original_curves, reconstructed_curves, arm_names)):
        color = colors[i % len(colors)]
        
        # Original (solid line with markers)
        ax2.plot(
            orig.iloc[:, 0], 
            orig.iloc[:, 1],
            'o-',
            color=color,
            linewidth=2,
            markersize=3,
            label=f'{name} (Original)',
            alpha=0.6
        )
        
        # Reconstructed (dashed line)
        ax2.plot(
            recon['time'],
            recon['survival_prob'],
            '--',
            color=color,
            linewidth=2.5,
            label=f'{name} (Reconstructed)',
            alpha=0.9
        )
    
    ax2.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Survival Probability (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Original vs Reconstructed', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10, ncol=2)
    ax2.set_ylim(0, 105)
    
    # Add validation metrics if provided
    if validation_results:
        overall = validation_results['overall']
        metrics_text = (
            f"Overall Metrics:\n"
            f"Mean RMSE: {overall['mean_rmse']:.3f}%\n"
            f"Mean MAE: {overall['mean_mae']:.3f}%\n"
            f"Status: {'✓ PASS' if overall['all_arms_pass'] else '✗ FAIL'}"
        )
        
        ax2.text(
            0.98, 0.02,
            metrics_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8),
            family='monospace'
        )
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    return fig, (ax1, ax2)


def plot_multi_arm_dashboard(
    original_curves: List[pd.DataFrame],
    reconstructed_curves: List[pd.DataFrame],
    reconstructed_ipd: pd.DataFrame,
    arm_names: List[str],
    validation_results: Dict,
    save_path: Optional[str] = None
):
    """
    Create comprehensive multi-arm validation dashboard.
    
    Args:
        original_curves: List of original curves
        reconstructed_curves: List of reconstructed curves
        reconstructed_ipd: Combined IPD dataframe
        arm_names: List of arm names
        validation_results: Validation results dict
        save_path: Path to save figure
    """
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. KM curve overlay (top row, span both columns)
    ax1 = fig.add_subplot(gs[0, :])
    for i, (orig, recon, name) in enumerate(zip(original_curves, reconstructed_curves, arm_names)):
        color = colors[i % len(colors)]
        ax1.plot(orig.iloc[:, 0], orig.iloc[:, 1], 'o-', color=color, 
                linewidth=2, markersize=3, label=f'{name} (Orig)', alpha=0.6)
        ax1.plot(recon['time'], recon['survival_prob'], '--', color=color,
                linewidth=2.5, label=f'{name} (Recon)', alpha=0.9)
    
    ax1.set_xlabel('Time (months)', fontweight='bold')
    ax1.set_ylabel('Survival Probability (%)', fontweight='bold')
    ax1.set_title('Multi-Arm KM Curve Comparison', fontweight='bold', fontsize=13)
    ax1.legend(loc='best', fontsize=10, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. Per-arm error plots (middle row)
    for i, name in enumerate(arm_names[:2]):  # Show first 2 arms
        ax = fig.add_subplot(gs[1, i])
        
        # Calculate errors for this arm
        orig = original_curves[i]
        recon = reconstructed_curves[i]
        
        from scipy.interpolate import interp1d
        interp_func = interp1d(
            recon['time'], recon['survival_prob'],
            kind='previous', bounds_error=False, 
            fill_value=(100, recon['survival_prob'].iloc[-1])
        )
        
        orig_times = orig.iloc[:, 0].values
        orig_surv = orig.iloc[:, 1].values
        errors = np.abs(orig_surv - interp_func(orig_times))
        
        color = colors[i % len(colors)]
        ax.plot(orig_times, errors, 'o-', color=color, linewidth=2, markersize=4)
        ax.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (months)', fontweight='bold', fontsize=10)
        ax.set_ylabel('Absolute Error (%)', fontweight='bold', fontsize=10)
        ax.set_title(f'{name} - Reconstruction Error', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # 3. Event distribution by arm (bottom left)
    ax3 = fig.add_subplot(gs[2, 0])
    for i, name in enumerate(arm_names):
        arm_ipd = reconstructed_ipd[reconstructed_ipd['arm'] == i]
        events = arm_ipd[arm_ipd['status'] == 1]['time']
        color = colors[i % len(colors)]
        ax3.hist(events, bins=20, alpha=0.6, color=color, label=f'{name} (n={len(events)})')
    
    ax3.set_xlabel('Time (months)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Event Distribution by Arm', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Validation metrics table (bottom right)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    arm_results = validation_results['arm_results']
    overall = validation_results['overall']
    
    metrics_text = "VALIDATION METRICS\n" + "="*45 + "\n\n"
    
    for name in arm_names:
        results = arm_results[name]
        metrics_text += f"{name}:\n"
        metrics_text += f"  RMSE:    {results['rmse']:6.3f}% {'✓' if results['passes_rmse'] else '✗'}\n"
        metrics_text += f"  MAE:     {results['mae']:6.3f}% {'✓' if results['passes_mae'] else '✗'}\n"
        metrics_text += f"  KS p-val: {results['ks_pvalue']:6.4f} {'✓' if results['passes_ks_test'] else '✗'}\n\n"
    
    metrics_text += "-"*45 + "\n"
    metrics_text += f"OVERALL:\n"
    metrics_text += f"  Mean RMSE: {overall['mean_rmse']:.3f}%\n"
    metrics_text += f"  Mean MAE:  {overall['mean_mae']:.3f}%\n"
    metrics_text += f"  Status: {'ALL PASS ✓' if overall['all_arms_pass'] else 'FAIL ✗'}\n"
    
    # Patient counts
    metrics_text += "\n" + "="*45 + "\n"
    for i, name in enumerate(arm_names):
        arm_ipd = reconstructed_ipd[reconstructed_ipd['arm'] == i]
        metrics_text += f"{name}: {len(arm_ipd)} patients\n"
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.9))
    
    plt.suptitle('Multi-Arm IPD Reconstruction Dashboard', 
                fontsize=16, fontweight='bold', y=0.99)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved: {save_path}")
    
    return fig