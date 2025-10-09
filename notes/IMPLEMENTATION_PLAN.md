# Implementation Plan - Day 2 (Oct 9)

## preprocess() function

**What it needs to do**:

1. Load pandas DataFrame (curve data)
2. Convert to R dataframe using rpy2
3. Convert lists (trisk, nrisk) to R vectors
4. Call IPDfromKM::preprocess()
5. Return result (keep as R object for now)

**Key challenge**: pandas2ri conversion

## extract_ipd_single() function

**What it needs to do**:

1. Take preprocessed R object
2. Convert arm_id and tot_events to R format
3. Call IPDfromKM::getIPD()
4. Extract IPD from result
5. Convert back to pandas DataFrame
6. Ensure columns: [time, status, arm]

**Key challenge**: Extracting nested R list objects

## Parity test

**Compare**:

- results/oracle_radio.csv (from R)
- results/python_radio.csv (from Python wrapper)

**Assertions**:

- Same number of rows
- time values match (atol=1e-6)
- status values match exactly
- arm values match exactly
