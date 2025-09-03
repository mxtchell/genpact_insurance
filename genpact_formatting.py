"""
Genpact-specific formatting functions for numeric values.
"""

import pandas as pd


def genpact_format_number(value, add_dollar_sign=False):
    """
    Format a number according to Genpact standards:
    - Values >= 1B: 1.1B
    - Values >= 1M: 1.1M  
    - Values >= 100K: 900.5K
    - Values >= 1K: 1K (no decimal)
    - Values < 1K: whole number
    """
    if pd.isna(value) or not isinstance(value, (int, float)):
        return str(value)
    
    abs_value = abs(value)
    
    if abs_value >= 1_000_000_000:
        formatted = f"{value / 1_000_000_000:.1f}B"
    elif abs_value >= 1_000_000:
        formatted = f"{value / 1_000_000:.1f}M"
    elif abs_value >= 100_000:
        formatted = f"{value / 1_000:.1f}K"
    elif abs_value >= 1_000:
        formatted = f"{value / 1_000:.0f}K"
    else:
        formatted = f"{value:.0f}"
    
    if add_dollar_sign:
        formatted = f"${formatted}"
    
    return formatted


def apply_genpact_formatting_to_dataframe(df, numeric_columns):
    """
    Apply Genpact formatting to specified numeric columns in a DataFrame.
    
    Args:
        df: pandas DataFrame to format
        numeric_columns: list of column names to apply formatting to
    
    Returns:
        DataFrame with formatted values
    """
    if df is None or df.empty:
        return df
    
    formatted_df = df.copy()
    
    for col in numeric_columns:
        if col in formatted_df.columns:
            col_lower = col.lower()
            # Skip formatting for percentages, change columns, and ranks
            is_percentage = '%' in col_lower or 'percent' in col_lower or 'change' in col_lower
            is_rank = 'rank' in col_lower or 'ranking' in col_lower
            
            if is_percentage or is_rank:
                # For percentage and rank columns, don't apply Genpact formatting - keep original values
                continue
            else:
                # Apply Genpact formatting with dollar sign for monetary columns
                formatted_df[col] = formatted_df[col].apply(lambda x: genpact_format_number(x, add_dollar_sign=True))
    
    return formatted_df