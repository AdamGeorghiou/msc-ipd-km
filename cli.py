#!/usr/bin/env python3
"""
Command-line interface for IPDfromKM Python wrapper
Allows easy IPD reconstruction without writing Python code
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.wrapper.ipd_wrapper import (
    preprocess,
    extract_ipd_single,
    validate_reconstruction,
    print_validation_summary
)

from src.wrapper.plotting import (
    plot_km_overlay,
    plot_validation_dashboard
)


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct IPD from Kaplan-Meier curves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic reconstruction
  python cli.py --curve data/curve.csv --at-risk data/at_risk.csv --output results/ipd.csv

  # With validation plots
  python cli.py --curve data/curve.csv --at-risk data/at_risk.csv \\
                --output results/ipd.csv --validate --plot

  # Specify survival scale
  python cli.py --curve data/curve.csv --at-risk data/at_risk.csv \\
                --output results/ipd.csv --scale 1  # For 0-1 scale

Input file formats:
  curve.csv: Two columns [time, survival_probability]
  at_risk.csv: Two columns [time, n_at_risk]
        '''
    )
    
    # Required arguments
    parser.add_argument(
        '--curve',
        type=str,
        required=True,
        help='Path to curve data CSV file [time, survival_prob]'
    )
    
    parser.add_argument(
        '--at-risk',
        type=str,
        required=True,
        help='Path to at-risk data CSV file [time, n_risk]'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save reconstructed IPD CSV file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--scale',
        type=int,
        default=100,
        choices=[1, 100],
        help='Survival scale: 100 for percentage (0-100), 1 for decimal (0-1). Default: 100'
    )
    
    parser.add_argument(
        '--arm-id',
        type=int,
        default=0,
        help='Treatment arm identifier. Default: 0'
    )
    
    parser.add_argument(
        '--total-events',
        type=int,
        default=None,
        help='Total number of events (optional, improves accuracy if known)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation metrics and print summary'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate validation plots (requires --validate)'
    )
    
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='figures',
        help='Directory to save plots. Default: figures/'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("IPDfromKM Python Wrapper - Command Line Interface")
    print("=" * 70)
    
    try:
        # Load data
        if args.verbose:
            print(f"\n1. Loading data...")
            print(f"   Curve: {args.curve}")
            print(f"   At-risk: {args.at_risk}")
        
        curve = pd.read_csv(args.curve)
        at_risk = pd.read_csv(args.at_risk)
        
        # Validate input
        if curve.shape[1] != 2:
            raise ValueError(f"Curve file must have 2 columns, got {curve.shape[1]}")
        
        if at_risk.shape[1] != 2:
            raise ValueError(f"At-risk file must have 2 columns, got {at_risk.shape[1]}")
        
        if args.verbose:
            print(f"   ✅ Loaded {len(curve)} curve points")
            print(f"   ✅ Loaded {len(at_risk)} at-risk time points")
        
        # Preprocess
        if args.verbose:
            print(f"\n2. Preprocessing (survival scale: {args.scale})...")
        
        prep = preprocess(
            curve_data=curve,
            at_risk_times=at_risk.iloc[:, 0].tolist(),
            at_risk_counts=at_risk.iloc[:, 1].tolist(),
            survival_scale=args.scale
        )
        
        if args.verbose:
            print(f"   ✅ Preprocessing complete")
        
        # Extract IPD
        if args.verbose:
            print(f"\n3. Extracting IPD (arm ID: {args.arm_id})...")
        
        ipd = extract_ipd_single(
            preprocessed_data=prep,
            arm_id=args.arm_id,
            total_events=args.total_events
        )
        
        # Print summary
        print(f"\n{'='*70}")
        print("RECONSTRUCTION SUMMARY")
        print(f"{'='*70}")
        print(f"Patients reconstructed: {len(ipd)}")
        print(f"Events (deaths):        {(ipd['status'] == 1).sum()}")
        print(f"Censored:               {(ipd['status'] == 0).sum()}")
        print(f"{'='*70}")
        
        # Save output
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        ipd.to_csv(args.output, index=False)
        print(f"\n✅ IPD saved to: {args.output}")
        
        # Validation
        if args.validate:
            if args.verbose:
                print(f"\n4. Running validation...")
            
            validation = validate_reconstruction(
                original_curve=curve,
                reconstructed_ipd=ipd
            )
            
            print_validation_summary(validation)
            
            # Generate plots
            if args.plot:
                if args.verbose:
                    print(f"\n5. Generating plots...")
                
                Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
                
                # Overlay plot
                overlay_path = Path(args.plot_dir) / 'km_overlay.png'
                plot_km_overlay(
                    original_curve=curve,
                    reconstructed_curve=validation['reconstructed_curve'],
                    validation_metrics=validation,
                    save_path=str(overlay_path)
                )
                
                # Dashboard
                dashboard_path = Path(args.plot_dir) / 'validation_dashboard.png'
                plot_validation_dashboard(
                    original_curve=curve,
                    reconstructed_curve=validation['reconstructed_curve'],
                    reconstructed_ipd=ipd,
                    validation_metrics=validation,
                    save_path=str(dashboard_path)
                )
                
                print(f"\n✅ Plots saved to: {args.plot_dir}/")
        
        print(f"\n{'='*70}")
        print("✅ SUCCESS! IPD reconstruction complete")
        print(f"{'='*70}\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        return 1
    
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())