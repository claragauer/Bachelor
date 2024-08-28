# Repository Structure

- run_pysubgroup.py: Implements Subgroup Discovery using PySubgroup.
- run_gurobi.py: Implements Subgroup Discovery using Gurobi optimization solver.
- preprocess_data.py: Contains data preprocessing functions like loading data, handling missing values, and handling outliers.

# Difference between create_search_space and define_selectors

Both methods, create_search_space in PySubgroup and define_selectors in Gurobi, pursue the same overarching goal: to define and optimize subgroups that meet certain conditions and maximize or minimize a target variable. However, they approach this in different ways.

Similarity:
Both functions are designed to define conditions that can be applied to the dataset to identify interesting subgroups.

Difference:
PySubgroup operates at a higher level of abstraction, working with objects and rules directly within the domain of Subgroup Discovery. In contrast, Gurobi uses an explicit mathematical formulation with binary decision variables and constraints, requiring a precise definition of the conditions.

# User instructions
1. Requirements
Ensure you have Python installed along with the necessary packages. You can install the required packages using the following command:

pip install pandas gurobipy pysubgroup scikit-learn

2. Pysubgroup Script 
To use different datasets, certain parameters are required for modification: 
- File Path: Update the file_path variable in the main() function to point to your dataset.
- Target Definition: In the define_target function, ensure the target variable ('Label') and TARGET_VALUE are correctly set for your dataset.
- Search Space: Modify the create_search_space function to specify the attributes and values that define your search space.

3. Gurobi Script 
To use different datasets or perform hyperparameter tuning, certain paramterers can be modified: 
- File Path: Update the file_path variable in the main() function to point to your dataset.
- Selectors: In the define_selectors function, ensure the correct attributes are being used to create binary selectors. The script dynamically creates selectors based on unique values in the dataset's columns.
- Constraints: Adjust the THETA_DC, THETA_CC, and THETA_MAX_RATIO constants to change the constraints of the optimization problem.