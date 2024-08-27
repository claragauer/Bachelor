import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

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
    return df.fillna(method=method)

def encode_categorical_columns(df, columns):
    """
    Encode categorical columns using LabelEncoder.
    
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