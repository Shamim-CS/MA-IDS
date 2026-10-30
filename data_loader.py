"""
Loading and cleaning of NetFlow CSV data.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from logger import IDSLogger


class DataLoader:
    """Handles loading and preprocessing of network flow data"""
    
    def __init__(self, logger: IDSLogger):
        self.logger = logger
    
    def load_data(
        self, 
        file_path: str, 
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Load CSV data with preprocessing
        
        Args:
            file_path: Path to CSV file
            sample_size: Optional number of rows to sample
            
        Returns:
            Tuple of (DataFrame, label_column_name)
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Loading data from: {file_path}")
        
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            self.logger.info(f"   -> Loaded {len(df)} rows")
            
            # Handle infinity and NaN values
            df = df.replace([np.inf, -np.inf], 0).fillna(0)
            self.logger.info("   -> Cleaned infinity and NaN values")
            
            # Identify label column
            label_col = self._identify_label_column(df)
            self.logger.info(f"   -> Identified Target Column: {label_col}")
            
            # Sample if requested
            if sample_size and len(df) > sample_size:
                self.logger.info(f"   -> Dataset large, randomly sampling {sample_size} rows...")
                df = df.sample(n=sample_size, random_state=42)
                #print(len(df))
                #print(sample_size)
                print("-----------------hi------------------")
            
            return df, label_col
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _identify_label_column(self, df: pd.DataFrame) -> str:
        """
        Identify the label column in the dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Name of the label column
        """
        # Common label column names
        label_candidates = ['Attack', 'Label', 'Class', 'attack', 'label', 'class']
        
        for candidate in label_candidates:
            if candidate in df.columns:
                return candidate
        
        # Default to last column
        return df.columns[-1]
    
    def get_data_info(self, df: pd.DataFrame, label_col: str) -> dict:
        """
        Get information about the loaded data
        
        Args:
            df: Input dataframe
            label_col: Name of label column
            
        Returns:
            Dictionary with data statistics
        """
        info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "label_column": label_col,
            "label_distribution": df[label_col].value_counts().to_dict()
        }
        
        self.logger.info("\nData Information:")
        self.logger.info(f"   Total Rows: {info['total_rows']}")
        self.logger.info(f"   Total Columns: {info['total_columns']}")
        self.logger.info(f"   Label Column: {info['label_column']}")
        self.logger.info("   Label Distribution:")
        
        for label, count in info['label_distribution'].items():
            percentage = (count / info['total_rows']) * 100
            self.logger.info(f"      {label}: {count} ({percentage:.2f}%)")
        
        return info
