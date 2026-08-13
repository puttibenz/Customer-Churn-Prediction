import os
import pandas as pd
from pathlib import Path
from typing import Union
from scripts.config import RAW_DATA_PATH, SAMPLE_DATA_PATH, TARGET_COL

def load_data(filepath: Union[str, Path] = RAW_DATA_PATH, sheet_name: str = "E Comm") -> pd.DataFrame:
    """
    Load raw dataset from Excel or CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the dataset file (.xlsx or .csv)
    sheet_name : str
        Sheet name if loading from Excel (default: 'E Comm')
        
    Returns
    -------
    pd.DataFrame
        Loaded raw dataframe
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found at: {filepath}")
        
    if filepath.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    elif filepath.suffix.lower() == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
    validate_data(df)
    return df

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that essential columns exist in the DataFrame.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from input dataset.")
    if len(df) == 0:
        raise ValueError("Loaded dataset is empty.")
    return True
