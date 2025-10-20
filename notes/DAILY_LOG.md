# MSc Dissertation Daily Log

## Format

- **Date**:
- **Hours worked**:
- **Main goal**:
- **What I accomplished**:
- **What I learned**:
- **Blockers/Issues**:
- **Tomorrow's priority**:
- **Mood/Energy**:

---

## Week 1: Foundation + Wrapper v0.1

### Tuesday, October 8, 2024

- **Hours worked**:
- **Main goal**: Environment setup + understand R package
- **What I accomplished**:
  - [ ] Created project structure
  - [ ] Set up GitHub repo
  - [ ] Installed R + Python dependencies
  - [ ] Downloaded sample data
  - [ ] Ran first R example
  - [ ]
- **What I learned**:
- **Blockers/Issues**:
- **Tomorrow's priority**: Build R oracle script
- **Mood/Energy**: /10

---

### Wednesday, October 9, 2024

- **Hours worked**:
- **Main goal**:
- **What I accomplished**:
- **What I learned**:
- **Blockers/Issues**:
- **Tomorrow's priority**:
- **Mood/Energy**: /10

---

<!-- Copy template below for each day -->

### Tuesday, October 8, 2024 - SETUP DAY ✅

- **Hours worked**: 3.5
- **Main goal**: Environment setup + understand R package ✅ COMPLETE
- **What I accomplished**:
  - [x] Created project structure
  - [x] Set up GitHub repo + first commits
  - [x] Installed R + Python + rpy2 (working despite warnings)
  - [x] Downloaded sample data: 145 curve points, 6 at-risk times
  - [x] Fixed dimension mismatch in at-risk data
  - [x] Created working R oracle script
  - [x] Explored R package thoroughly in console
  - [x] Understand data structures completely
  - [x] Designed Python wrapper API
- **What I learned**:
  - Radiationdata structure: radio/radioplus curves + at-risk tables
  - preprocess() takes: (dat, trisk, nrisk, maxy) → cleaned data object
  - getIPD() takes: (prep, armID, tot.events) → DataFrame[time, status, arm]
  - status column: 1 = event (death), 0 = censored
  - At-risk reporting stops when few patients remain (6 of 8 time points)
- **Blockers/Issues**:
  - rpy2 API/ABI mode warning → ignored, works fine
  - At-risk dimension mismatch → fixed by handling arrays separately
- **Tomorrow's priority**:
  1. Implement preprocess() wrapper function
  2. Implement extract_ipd_single() wrapper function
  3. Write parity test comparing Python vs R oracle
  4. Get first test PASSING
- **Mood/Energy**: 9/10 - Excellent progress! Foundation is solid! 🚀

**Files created today**:

- data/examples/radio_curve.csv (145 rows)
- data/examples/at_risk_radio.csv (6 rows)
- src/r_oracle/oracle_single_arm.R (working!)
- results/oracle_radio.csv (213 patients reconstructed)
- src/wrapper/ipd_wrapper.py (skeleton)

**Repository**: All committed and pushed to GitHub ✅

### Wednesday, October 9, 2024 - PARITY TEST PASSING! 🎉🎉🎉

- **Hours worked**: 5.5
- **Main goal**: Implement wrapper and pass parity test ✅ **ACHIEVED!**

**MAJOR MILESTONE**: Parity test is PASSING!

- **What I accomplished**:
  - [x] Implemented preprocess() function with modern rpy2 API
  - [x] Implemented extract_ipd_single() function
  - [x] Fixed pandas2ri deprecation (using localconverter)
  - [x] Fixed column name mismatch (treat → arm)
  - [x] **PARITY TEST PASSING** ✅✅✅
  - [x] Python wrapper produces IDENTICAL results to R oracle
  - [x] 213 patients reconstructed correctly
  - [x] All validation checks pass
- **Technical achievements**:
  - **Reconstructed patients**: 213
  - **Events**: 134
  - **Censored**: 79
  - **Time value accuracy**: Within numerical precision
  - **Status values**: Exact match
  - **Arm values**: Exact match
- **What I learned**:
  - Modern rpy2 uses localconverter context manager
  - R package outputs 'treat' column, renamed to 'arm' for consistency
  - numpy.allclose() perfect for numerical comparison
  - Parity testing validates the entire pipeline
- **Blockers/Issues**:
  - pandas2ri.activate() deprecated → Fixed with localconverter
  - Column name mismatch → Fixed in both R and Python
  - All resolved! ✅
- **Tomorrow's priorities**:
  - Validation metrics (RMSE, MAE, KS test)
  - Overlay plotting function
  - Multi-arm support
  - Edge case testing
  - CLI tool
- **Mood/Energy**: 11/10 - THIS IS THE BIG WIN! 🚀🎉

---

**CRITICAL MILESTONE ACHIEVED**:
The Python wrapper is validated. It produces identical results to the R package.
This is the foundation for the entire dissertation. Everything from here is building on proven functionality.

**Files working**:

- src/wrapper/ipd_wrapper.py ✅
- tests/test_parity_single_arm.py ✅ PASSING
- src/r_oracle/oracle_single_arm.R ✅
- results/oracle_radio.csv (ground truth)
- results/python_radio.csv (validated match)

**Multi-arm testing complete**:

- Radio arm: 213 patients (RMSE: 0.68%, MAE: 0.46%) ✅
- RadioPlus arm: 211 patients (RMSE: 0.67%, MAE: 0.47%) ✅
- Combined: 424 patients reconstructed
- All validation checks passing
- Multi-arm plots generated

**TIER 1 STATUS: 100% FUNCTIONALLY COMPLETE** 🎉
