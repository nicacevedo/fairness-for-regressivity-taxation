import pandas as pd
import pyreadr # read .rda data


vars_dict = pyreadr.read_r('data_ingest/vars_dict.rda')["vars_dict"]
col_type_dict = (
    pd.Series(vars_dict.var_data_type.values, index=vars_dict.var_name_model)
    .dropna()
    .to_dict()
)

def recode_column_type(series, series_name, type_map=col_type_dict):
    """Applies the correct data type to a pandas Series."""
    if series_name not in type_map:
        return series

    target_type = type_map[series_name]
    if target_type == 'numeric':
        return pd.to_numeric(series, errors='coerce')
    if target_type == 'character':
        return series.astype(str)
    if target_type == 'logical':
        # Handles 0/1, True/False, and missing values
        return pd.to_numeric(series, errors='coerce').astype('boolean')
    if target_type == 'categorical':
        return pd.Categorical(series)
    if target_type == 'date':
        return pd.to_datetime(series, errors='coerce').dt.date
    return series
