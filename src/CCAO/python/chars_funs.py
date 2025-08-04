import pandas as pd
import numpy as np
from typing import List, Dict, Union

## ---------------------------------------------------------------------------
## Placeholder Dependencies
## ---------------------------------------------------------------------------
# The R code depends on external objects and functions (like `town_get_assmnt_year`
# and `chars_cols`). We must define placeholders for them to make the code run.

def town_get_assmnt_year(
    town: Union[str, pd.Series],
    year: Union[int, pd.Series],
    round_type: str = "floor"
) -> Union[int, pd.Series]:
    """
    Placeholder for the R function `ccao::town_get_assmnt_year`.
    
    Calculates the last (floor) or next (ceiling) assessment year for a given
    township based on Cook County's triennial cycle. This is a simplified model.
    """
    # Simplified logic: North townships assessed in 2022, 2025, etc.
    # South in 2023, 2026, etc. West (City) in 2024, 2027, etc.
    town_series = town if isinstance(town, pd.Series) else pd.Series([town])
    year_series = year if isinstance(year, pd.Series) else pd.Series([year])

    # Assign each township to a group (0, 1, 2)
    # This is a mock mapping. A real one would be more detailed.
    town_map = {
        'Evanston': 0, 'Niles': 0, 'Northfield': 0, # North
        '25': 2, '77': 0, '10': 1 # Assuming 25 is city, 77 north, 10 south
    }
    
    # Get the remainder to find the group's most recent assessment year relative to 2023
    # For example, a South town (group 1) in 2025 would be 2025 % 3 = 1. `2025 - 2 = 2023`.
    # A North town (group 0) in 2025 would be 2025 % 3 = 1. `2025 - 1 = 2024`.
    # A West town (group 2) in 2025 would be 2025 % 3 = 1. `2025 - (1-3) = 2027?` Wait...
    # Let's try a simpler base year approach.
    base_year = 2023 # South/Suburbs Tri
    town_group = town_series.map(town_map).fillna(2) # Default to City group
    
    year_mod = (year_series - town_group - base_year) % 3
    
    if round_type == "floor":
        assessment_year = year_series - year_mod
    elif round_type == "ceiling":
        # If year_mod is 0, it's an assessment year. Otherwise, find the next one.
        assessment_year = np.where(year_mod == 0, year_series, year_series - year_mod + 3)
    else:
        raise ValueError("round_type must be 'floor' or 'ceiling'")
        
    return assessment_year if isinstance(town, pd.Series) else assessment_year.iloc[0]


# This dictionary must be defined, as it's a critical dependency.
# It maps source columns from HIE data to target columns in the main dataset.
chars_cols = {
    'add': {
        'source': ['qu_sqft_bld'], 'target': ['char_bldg_sf']
    },
    'replace': {
        'source': ['qu_rooms'], 'target': ['char_rooms']
    }
}

# Helper function to get the last non-zero value from a Series.
def last_nonzero_element(series: pd.Series) -> any:
    """Finds the last element in a Series that is not 0 or '0'."""
    if series.dtype == 'object':
        valid_values = series[series.notna() & (series != '0')]
        return valid_values.iloc[-1] if not valid_values.empty else '0'
    else:
        valid_values = series[series.notna() & (series != 0)]
        return valid_values.iloc[-1] if not valid_values.empty else 0
        
# Helper to find the source column for a given target column.
def chars_get_col(target_col: str) -> str:
    """Looks up the HIE source column for a given target characteristics column."""
    for group in ['add', 'replace']:
        try:
            idx = chars_cols[group]['target'].index(target_col)
            return chars_cols[group]['source'][idx]
        except (ValueError, IndexError):
            continue
    raise ValueError(f"No source column found for target '{target_col}'")


## ---------------------------------------------------------------------------
## Main Function Translations
## ---------------------------------------------------------------------------

def chars_fix_age(
    age: Union[int, pd.Series],
    year: Union[int, pd.Series],
    town: Union[str, pd.Series]
) -> Union[int, pd.Series]:
    """
    Calculates the correct age of a property for a given year and township.
    The AGE variable in CCAO data often only updates upon reassessment.

    Args:
        age: A numeric vector of ages.
        year: A numeric vector of tax years.
        town: A character vector of town codes or names.

    Returns:
        A numeric vector of "true" ages.
    """
    # Convert scalars to Series to handle broadcasting easily
    if not isinstance(age, pd.Series): age = pd.Series([age])
    if not isinstance(year, pd.Series): year = pd.Series([year])
    if not isinstance(town, pd.Series): town = pd.Series([town])
    
    # Basic input validation
    if not pd.api.types.is_numeric_dtype(age) or not pd.api.types.is_numeric_dtype(year):
        raise TypeError("age and year must be numeric.")
    if not pd.api.types.is_string_dtype(town) and not pd.api.types.is_object_dtype(town):
        raise TypeError("town must be a character/string.")
    if (age < 0).any():
        raise ValueError("age must be >= 0.")

    # Calculate the year offset to add to age
    # Broadcasting in pandas/numpy handles mismatched lengths automatically.
    year_diff = year - town_get_assmnt_year(town, year, round_type="floor")
    
    return age + year_diff

# ---

def chars_288_active(
    start_year: Union[int, pd.Series],
    town: Union[str, pd.Series]
) -> List[np.ndarray]:
    """
    Calculates the active years for a Home Improvement Exemption (HIE or "288").
    
    A "288" exemption lasts for 4 years or until the next triennial
    reassessment, whichever is longer.

    Args:
        start_year: A numeric vector of HIE start years.
        town: A character vector of town codes or names.

    Returns:
        A list of numpy arrays, where each array contains the active years for an HIE.
    """
    # Convert scalars to Series for consistent processing
    start_years = start_year if isinstance(start_year, pd.Series) else pd.Series([start_year])
    towns = town if isinstance(town, pd.Series) else pd.Series([town])
    
    if not pd.api.types.is_numeric_dtype(start_years):
        raise TypeError("start_year must be numeric.")
    
    results = []
    # Use zip to iterate over inputs simultaneously, similar to R's mapply
    # We must handle broadcasting manually if lengths differ.
    if len(start_years) == 1 and len(towns) > 1:
        start_years = pd.Series(np.repeat(start_years.iloc[0], len(towns)))
    if len(towns) == 1 and len(start_years) > 1:
        towns = pd.Series(np.repeat(towns.iloc[0], len(start_years)))

    for y_start, twn in zip(start_years, towns):
        if pd.isna(y_start) or pd.isna(twn):
            results.append(np.array([np.nan]))
            continue
            
        y_start = int(y_start)
        
        # Determine the end year of the exemption
        end_year_statute = y_start + 4
        end_year_assessment = town_get_assmnt_year(twn, end_year_statute, round_type="ceiling")
        end_year = max(end_year_statute, end_year_assessment)
        
        # Active years are from the start year up to (but not including) the end year
        active_years = np.arange(y_start, end_year)
        results.append(active_years)
        
    return results

# ---

def chars_sparsify(
    data: pd.DataFrame,
    pin_col: str,
    year_col: str,
    town_col: str,
    upload_date_col: str,
    additive_source: List[str],
    replacement_source: List[str]
) -> pd.DataFrame:
    """
    Transforms HIE data into a sparse format (one row per PIN per active year).

    Args:
        data: DataFrame containing HIE records.
        pin_col: Name of the PIN column.
        year_col: Name of the HIE start year column.
        town_col: Name of the township column.
        upload_date_col: Name of the upload date column (for tie-breaking).
        additive_source: List of source columns with additive values (e.g., sqft).
        replacement_source: List of source columns with replacement values (e.g., rooms).

    Returns:
        A sparsified DataFrame with characteristic updates per PIN per year.
    """
    # 1. Consolidate improvements within the same start year for each PIN.
    # We sort by upload date so `last_nonzero_element` gets the latest value.
    agg_dict = {col: 'sum' for col in additive_source}
    agg_dict.update({col: last_nonzero_element for col in replacement_source})
    agg_dict[town_col] = 'first'
    
    consolidated = (
        data.sort_values(upload_date_col)
            .groupby([pin_col, year_col])
            .agg(agg_dict)
            .reset_index()
    )
    
    # 2. Determine active years for each HIE record.
    consolidated['active_years'] = chars_288_active(
        consolidated[year_col].astype(int),
        consolidated[town_col].astype(str)
    )
    
    # 3. Expand the DataFrame to have one row per active year.
    # This is the equivalent of R's `tidyr::unnest`.
    expanded = consolidated.explode('active_years').rename(columns={'active_years': 'year'})
    
    if expanded.empty:
        return pd.DataFrame()

    # 4. Combine overlapping HIEs.
    # If a PIN has multiple HIEs active in the same year, we combine their effects.
    final_agg_dict = {col: 'sum' for col in additive_source}
    final_agg_dict.update({col: last_nonzero_element for col in replacement_source})
    final_agg_dict['hie_num_active'] = ('year', 'size') # Count how many HIEs are active
    
    sparsified = (
        expanded.sort_values('year')
                .groupby([pin_col, 'year'])
                .agg(final_agg_dict)
                .reset_index()
    )
    
    return sparsified

# ---

def chars_update(
    data: pd.DataFrame,
    additive_target: List[str],
    replacement_target: List[str]
) -> pd.DataFrame:
    """
    Updates property characteristics using joined HIE data.

    This function expects a DataFrame that has been left-joined with the output
    of `chars_sparsify`, so both target (e.g., `char_rooms`) and source
    (e.g., `qu_rooms`) columns are present.

    Args:
        data: The merged DataFrame.
        additive_target: List of target columns for additive updates.
        replacement_target: List of target columns for replacement updates.

    Returns:
        The DataFrame with updated characteristic columns.
    """
    df = data.copy()

    # Process additive columns
    for target in additive_target:
        source = chars_get_col(target)
        # Fill NaNs with 0 before summing to avoid propagating NaNs
        df[target] = df[[target, source]].fillna(0).sum(axis=1)

    # Process replacement columns
    for target in replacement_target:
        source = chars_get_col(target)
        # Condition where the source value is present and not zero
        # The mask updates the target column only where the condition is True
        is_valid_source = df[source].notna() & (df[source] != 0) & (df[source] != '0')
        df[target] = df[target].mask(is_valid_source, df[source])

    return df