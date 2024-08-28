# Subgroup Discovery Test Suite
## Overview

This test suite is designed to validate the correctness and robustness of the Subgroup Discovery implementations using PySubgroup and Gurobi. The tests cover various aspects, including data preprocessing, handling missing values, outlier detection, and the overall optimization pipeline.

## Test Structure

- test_pipeline: Runs the full pipeline with selected CSV files to ensure the entire process, from data loading to subgroup discovery, is functioning correctly.
- test_missing_values_in_test_files: Checks specific test files for missing values and ensures that the preprocessing steps handle them appropriately.
- test_handle_outliers: Verifies the correct handling and removal of outliers in the data Z-score.

## Test Files

Test files are stored in the test/test_data directory and include various scenarios to test the robustness of the implementation:

- test1.csv to test6.csv: General test cases for subgroup discovery.
- test7missing.csv to test10missing.csv: Test files designed to check the handling of missing values.
- test11outlier.csv and test12outlier.csv: Test files to validate outlier detection and removal.