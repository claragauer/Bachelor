import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Constants
Z_SCORE_THRESHOLD = 1.5  # Common threshold for identifying outliers, experimental value
IQR_MULTIPLIER = 1.5     # Constant for IQR calculation


def load_data(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def handle_missing_values(df, method='ffill'):
    """
    Handle missing values in the dataset.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        method (str): Method to handle missing values (default: 'ffill').
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    if method == 'ffill':
        return df.ffill()  # Fill forward to replace missing values with the last valid entry
    elif method == 'bfill':
        return df.bfill()  # Fill backward to replace missing values with the next valid entry
    else:
        raise ValueError("Invalid method. Use 'ffill' or 'bfill'.")


def encode_categorical_columns(df, columns):
    """
    Encode categorical data into numerical format using LabelEncoder.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to encode.
        
    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
        dict: Dictionary of LabelEncoders for each column.
    """
    label_encoders = {}
    for column in columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders


def balance_data(df, target, method='oversample'):
    """
    Balance the dataset using oversampling or undersampling.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        target (str): Target column for balancing.
        method (str): Method for balancing data ('oversample', 'undersample').
        
    Returns:
        pd.DataFrame: Balanced DataFrame.
    """
    if method == 'oversample':
        sm = SMOTE()
        X, y = sm.fit_resample(df.drop(columns=[target]), df[target])
        df_balanced = pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=[target])], axis=1)
    elif method == 'undersample':
        rus = RandomUnderSampler()
        X, y = rus.fit_resample(df.drop(columns=[target]), df[target])
        df_balanced = pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=[target])], axis=1)
    else:
        raise ValueError("Invalid method. Use 'oversample' or 'undersample'.")
    
    return df_balanced


def handle_outliers(df, columns, method='z-score', threshold=Z_SCORE_THRESHOLD):
    """
    Handle outliers in the dataset.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to check for outliers.
        method (str): Method to handle outliers ('z-score' or 'iqr').
        threshold (float): Threshold for identifying outliers.
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if method == 'z-score':
        z_scores = zscore(df[columns].dropna())  # Compute z-scores, ignoring NaNs
        return df[(z_scores < threshold).all(axis=1)]  # Remove outliers using Z-score
    elif method == 'iqr':
        Q1 = df[columns].quantile(0.25)
        Q3 = df[columns].quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df[columns] < (Q1 - IQR_MULTIPLIER * IQR)) | (df[columns] > (Q3 + IQR_MULTIPLIER * IQR))).any(axis=1)]  # Remove outliers using IQR
    else:
        raise ValueError("Invalid method. Use 'z-score' or 'iqr'.")


def preprocess_data(file_path, categorical_columns):
    """
    Load and preprocess data.

    Parameters:
        file_path (str): Path to the CSV file.
        categorical_columns (list): List of categorical columns to encode.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
        dict: Dictionary of LabelEncoders.
    """
    # Load data
    df = load_data(file_path)
    print("Data loaded successfully.")
    
    # Handle missing values
    df = handle_missing_values(df)
    print("Missing values handled.")
    
    # Encode categorical columns
    df, label_encoders = encode_categorical_columns(df, categorical_columns)
    print("Categorical columns encoded.")
    
    return df, label_encoders
