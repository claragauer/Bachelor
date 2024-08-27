import unittest
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocess_data import (
    load_data,
    handle_missing_values,
    encode_categorical_columns,
    preprocess_data
)
import os

class TestDataPreprocessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Dictionary with the test data
        """
        # Create a temporary DataFrame for testing
        cls.data = pd.DataFrame({
            'Color': ['Red', 'Blue', None, 'Green', 'Red'],
            'Shape': ['Circle', 'Square', 'Circle', None, 'Square'],
            'Label': [1, 0, 1, 0, None]
        })
        # Save the temporary DataFrame to a CSV file for load_data testing
        cls.test_csv_path = 'test_data.csv'
        cls.data.to_csv(cls.test_csv_path, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up any state.
        """
        # Remove the temporary CSV file after all tests
        if os.path.exists(cls.test_csv_path):
            os.remove(cls.test_csv_path)

    def test_load_data(self):
        """
        Test loading data from a CSV file.
        """
        df = load_data(self.test_csv_path)
        self.assertIsInstance(df, pd.DataFrame) # type of the object returned by load_data
        self.assertEqual(len(df), len(self.data)) # rows match
        self.assertListEqual(list(df.columns), list(self.data.columns)) # columns match

    def test_handle_missing_values(self):
        """
        Test handling missing values.
        """
        df = handle_missing_values(self.data.copy())
        # Check that there are no missing values
        self.assertFalse(df.isnull().values.any())
        # Check specific values after forward fill
        self.assertEqual(df.loc[2, 'Color'], 'Blue')
        self.assertEqual(df.loc[3, 'Shape'], 'Circle')

    def test_encode_categorical_columns(self):
        """
        Test encoding categorical columns.
        """
        df, encoders = encode_categorical_columns(self.data.copy(), ['Color', 'Shape'])
        # Check if categorical columns are encoded
        self.assertTrue(df['Color'].dtype == 'int32' or df['Color'].dtype == 'int64')
        self.assertTrue(df['Shape'].dtype == 'int32' or df['Shape'].dtype == 'int64')
        # Check if label encoders are returned
        self.assertIn('Color', encoders)
        self.assertIn('Shape', encoders)

    def test_preprocess_data(self):
        """
        Test full data preprocessing pipeline.
        """
        df, encoders = preprocess_data(self.test_csv_path, ['Color', 'Shape'])
        # Check if preprocessing was successful
        self.assertFalse(df.isnull().values.any())
        self.assertTrue(df['Color'].dtype == 'int32' or df['Color'].dtype == 'int64')
        self.assertTrue(df['Shape'].dtype == 'int32' or df['Shape'].dtype == 'int64')
        self.assertIn('Color', encoders)
        self.assertIn('Shape', encoders)

if __name__ == '__main__':
    unittest.main()
