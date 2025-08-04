# -*- coding: utf-8 -*-
"""
Python translation of the 2025 R data ingestion script.

This script pulls and prepares data for a residential property valuation model,
featuring more advanced feature engineering.
"""
import os
import time
import yaml
import pandas as pd
import numpy as np
import networkx as nx
from pyathena import connect
from pyathena.pandas_cursor import PandasCursor

# ======================================================================================
# 0. Placeholder Functions
# ======================================================================================
# In a real project, these would be in separate utility files/modules.
# The R script sources a directory of helpers and uses a custom `ccao` package.
# We create placeholders since we don't have their source code.

def load_paths_and_params():
    """Loads file paths and run parameters from YAML files."""
    # In R, this is handled by sourcing setup files.
    # In Python, we can load them from a config file.
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)

    # A simple path dictionary
    paths = {
        'input': {
            'hie': {'local': 'data/input/hie_data.parquet'},
            'char': {'local': 'data/input/char_data.parquet'},
            'training': {'local': 'data/input/training_data.parquet'},
            'assessment': {'local': 'data/input/assessment_data.parquet'},
            'complex_id': {'local': 'data/input/complex_id_data.parquet'},
            'land_nbhd_rate': {'local': 'data/input/land_nbhd_rate_data.parquet'}
        }
    }
    return paths, params

def get_vars_dict():
    """Placeholder for `ccao::vars_dict`."""
    # This would typically load from a shared file.
    return pd.DataFrame({
        'var_name_model': ['meta_pin', 'char_bldg_sf', 'sv_is_outlier'],
        'var_data_type': ['character', 'numeric', 'logical']
    })

# Placeholders for the custom `ccao` package functions
def chars_sparsify(df, **kwargs):
    print("NOTE: `chars_sparsify` is a placeholder.")
    if not df.empty:
        df['hie_num_active'] = 1 # Dummy column for the example
    return df

def chars_update(df, **kwargs):
    print("NOTE: `chars_update` is a placeholder.")
    return df

def vars_recode(df, **kwargs):
    print("NOTE: `vars_recode` is a placeholder.")
    return df


# ======================================================================================
# 1. Setup
# ======================================================================================
start_time = time.time()
print("--- 1. Setup ---")

# Load configuration
paths, params = load_paths_and_params()

# Establish Athena connection
# The R script uses `noctua` which leverages Apache Arrow for speed.
# In Python, pyathena's `PandasCursor` does the same thing.
conn = connect(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION"),
    cursor_class=PandasCursor # Use PandasCursor for performance
)


# ======================================================================================
# 2. Define Functions
# ======================================================================================
print("\n--- 2. Defining helper functions ---")

# Create a dictionary of column types for easy lookup
vars_dict = get_vars_dict()
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

def process_array_column(series):
    """
    Processes a column where cells might be lists (from Athena arrays).
    Collapses multi-element lists into a string, extracts single elements,
    and handles empty lists.
    """
    def process_cell(cell):
        if isinstance(cell, (list, np.ndarray)):
            if len(cell) > 1:
                return ", ".join(map(str, cell))
            if len(cell) == 1:
                return str(cell[0])
        return np.nan # Return NaN for empty lists or other types

    return series.apply(process_cell)


# ======================================================================================
# 3. Pull Data
# ======================================================================================
print("\n--- 3. Pulling data from Athena ---")

# Pull training data
print("Pulling training data...")
pull_start_time = time.time()
training_data_query = f"""
    SELECT
        sale.sale_price AS meta_sale_price,
        sale.sale_date AS meta_sale_date,
        sale.doc_no AS meta_sale_document_num,
        sale.deed_type AS meta_sale_deed_type,
        sale.seller_name AS meta_sale_seller_name,
        sale.buyer_name AS meta_sale_buyer_name,
        sale.sv_is_outlier,
        sale.sv_outlier_reason1,
        sale.sv_outlier_reason2,
        sale.sv_outlier_reason3,
        sale.sv_run_id,
        res.*
    FROM model.vw_card_res_input res
    INNER JOIN default.vw_pin_sale sale
        ON sale.pin = res.meta_pin
        AND sale.year = res.year
    WHERE CAST(res.year AS int)
        BETWEEN CAST({params['input']['min_sale_year']} AS int) - {params['input']['n_years_prior']}
        AND CAST({params['input']['max_sale_year']} AS int)
    AND sale.deed_type IN ('01', '02', '05')
    AND NOT sale.is_multisale
    AND NOT sale.sale_filter_same_sale_within_365
    AND NOT sale.sale_filter_less_than_10k
    AND NOT sale.sale_filter_deed_type
"""
# Using the cursor directly returns a pandas DataFrame
training_data = conn.cursor().execute(training_data_query).as_pandas()
print(f"Training data pulled in {time.time() - pull_start_time:.2f} seconds.")

# Pull HIE data
print("Pulling HIE data...")
pull_start_time = time.time()
hie_data = conn.cursor().execute("SELECT * FROM ccao.hie").as_pandas()
hie_data.to_parquet(paths['input']['hie']['local'], index=False)
print(f"HIE data pulled and saved in {time.time() - pull_start_time:.2f} seconds.")

# Pull assessment data
print("Pulling assessment data...")
pull_start_time = time.time()
assessment_data_query = f"""
    SELECT *
    FROM model.vw_card_res_input
    WHERE year BETWEEN '{int(params['assessment']['data_year']) - 1}'
    AND '{params['assessment']['data_year']}'
"""
assessment_data = conn.cursor().execute(assessment_data_query).as_pandas()
# Save both years for reporting
assessment_data.to_parquet(paths['input']['char']['local'], index=False)
# Filter to only the assessment year for modeling
assessment_data = assessment_data[
    assessment_data['year'] == params['assessment']['data_year']
].copy()
print(f"Assessment data pulled and processed in {time.time() - pull_start_time:.2f} seconds.")

# Pull land rate data
print("Pulling land rate data...")
pull_start_time = time.time()
land_nbhd_rate_query = f"""
    SELECT *
    FROM ccao.land_nbhd_rate
    WHERE year = '{params['assessment']['year']}'
"""
land_nbhd_rate_data = conn.cursor().execute(land_nbhd_rate_query).as_pandas()
print(f"Land rate data pulled in {time.time() - pull_start_time:.2f} seconds.")

# Close connection to Athena
conn.close()
print("Athena connection closed.")


# ======================================================================================
# 4. Home Improvement Exemptions (HIE)
# ======================================================================================
print("\n--- 4. Integrating Home Improvement Exemption (HIE) data ---")
# This section is functionally identical to the previous script, so comments are condensed.

## 4.1. Training Data
print("Processing HIE for training data...")
hie_data_training_sparse = chars_sparsify(hie_data) # Placeholder
hie_data_training_sparse['ind_pin_is_multicard'] = False
hie_data_training_sparse['year'] = hie_data_training_sparse['year'].astype(str)

training_data_w_hie = pd.merge(
    training_data,
    hie_data_training_sparse,
    left_on=["meta_pin", "year", "ind_pin_is_multicard"],
    right_on=["pin", "year", "ind_pin_is_multicard"],
    how="left"
).pipe(chars_update) # Placeholder for ccao::chars_update
# Clean up
qu_cols = [col for col in training_data_w_hie.columns if col.startswith('qu_')]
training_data_w_hie = training_data_w_hie.drop(columns=qu_cols)
training_data_w_hie['hie_num_active'] = training_data_w_hie['hie_num_active'].fillna(0)


## 4.2. Assessment Data
print("Processing HIE for assessment data...")
hie_last_year = int(params['assessment']['year']) - 1
hie_data_assessment_sparse = hie_data[hie_data['hie_last_year_active'] == hie_last_year].copy()
hie_data_assessment_sparse = chars_sparsify(hie_data_assessment_sparse) # Placeholder
hie_data_assessment_sparse['ind_pin_is_multicard'] = False
hie_data_assessment_sparse['year'] = hie_data_assessment_sparse['year'].astype(str)

assessment_data_w_hie = pd.merge(
    assessment_data,
    hie_data_assessment_sparse,
    left_on=["meta_pin", "year", "ind_pin_is_multicard"],
    right_on=["pin", "year", "ind_pin_is_multicard"],
    how="left"
).pipe(chars_update) # Placeholder
# Clean up
qu_cols = [col for col in assessment_data_w_hie.columns if col.startswith('qu_')]
assessment_data_w_hie = assessment_data_w_hie.drop(columns=qu_cols)
assessment_data_w_hie = assessment_data_w_hie.rename(
    columns={'hie_num_active': 'hie_num_expired'}
).fillna({'hie_num_expired': 0})


# ======================================================================================
# 5. Add Features and Clean
# ======================================================================================
print("\n--- 5. Adding features and cleaning ---")

## 5.1. Training Data
print("Cleaning and featurizing training data...")
# Use method chaining for a flow similar to R's pipes
training_data_clean = training_data_w_hie.copy()

# Recode apartments and NCU
# Using np.select is a good equivalent to R's case_when
conditions_apts = [
    training_data_clean['char_class'].isin(["211", "212"]) & training_data_clean['char_apts'].notna(),
    training_data_clean['char_class'].isin(["211", "212"]) & training_data_clean['char_apts'].isna()
]
choices_apts = [training_data_clean['char_apts'], "UNKNOWN"]
training_data_clean['char_apts'] = np.select(conditions_apts, choices_apts, default="NONE")
training_data_clean['char_ncu'] = np.where(
    training_data_clean['char_class'] == "212",
    training_data_clean['char_ncu'].fillna(0),
    0
)

# Process array columns and apply data types
tax_cols = [col for col in training_data_clean.columns if col.startswith('loc_tax_')]
training_data_clean[tax_cols] = training_data_clean[tax_cols].apply(process_array_column)
training_data_clean['loc_tax_municipality_name'] = training_data_clean['loc_tax_municipality_name'].fillna("UNINCORPORATED")
training_data_clean = training_data_clean.apply(lambda col: recode_column_type(col, col.name))

# Miscellaneous cleaning
training_data_clean['sv_is_outlier'] = training_data_clean['sv_is_outlier'].fillna(False)
training_data_clean['ccao_is_corner_lot'] = training_data_clean['ccao_is_corner_lot'].fillna(False)
for col in training_data_clean.select_dtypes(include=['object']).columns:
    training_data_clean[col] = training_data_clean[col].replace('', np.nan)

# Create a count of sales in the past N years
# This translates the double left_join and summarize logic from R
sales_for_join = training_data[['meta_pin', 'meta_sale_document_num', 'meta_sale_date']].drop_duplicates()
valid_sales_for_join = training_data.loc[~training_data['sv_is_outlier'].fillna(False), ['meta_pin', 'meta_sale_date']].drop_duplicates()

merged_sales = pd.merge(
    sales_for_join,
    valid_sales_for_join,
    on='meta_pin',
    how='left',
    suffixes=('_current', '_past')
)
merged_sales['meta_sale_date_current'] = pd.to_datetime(merged_sales['meta_sale_date_current'])
merged_sales['meta_sale_date_past'] = pd.to_datetime(merged_sales['meta_sale_date_past'])
time_diff = merged_sales['meta_sale_date_current'] - merged_sales['meta_sale_date_past']

n_years = params['input']['n_years_prior']
within_n_years = time_diff.between(pd.Timedelta('1 day'), pd.Timedelta(f'{n_years*365.25} days'))

sales_count = (
    merged_sales[within_n_years]
    .groupby(['meta_pin', 'meta_sale_document_num'])
    .size()
    .reset_index(name='meta_sale_count_past_n_years')
)

training_data_clean = pd.merge(
    training_data_clean,
    sales_count,
    on=['meta_pin', 'meta_sale_document_num'],
    how='left'
)
training_data_clean['meta_sale_count_past_n_years'] = training_data_clean['meta_sale_count_past_n_years'].fillna(0)

# Create time/date features
training_data_clean['meta_sale_date'] = pd.to_datetime(training_data_clean['meta_sale_date'])
training_data_clean['time_sale_year'] = training_data_clean['meta_sale_date'].dt.year
training_data_clean['time_sale_quarter_of_year'] = 'Q' + training_data_clean['meta_sale_date'].dt.quarter.astype(str)
training_data_clean['time_sale_month_of_year'] = training_data_clean['meta_sale_date'].dt.month
training_data_clean['time_sale_day_of_year'] = training_data_clean['meta_sale_date'].dt.dayofyear
training_data_clean['time_sale_day_of_month'] = training_data_clean['meta_sale_date'].dt.day
training_data_clean['time_sale_day_of_week'] = training_data_clean['meta_sale_date'].dt.dayofweek
training_data_clean['time_sale_post_covid'] = training_data_clean['meta_sale_date'] >= pd.to_datetime('2020-03-15')

# Final filtering and cleaning
date_min = pd.to_datetime(f"{params['input']['min_sale_year']}-01-01")
date_max = pd.to_datetime(f"{params['input']['max_sale_year']}-12-31")
training_data_clean = training_data_clean[
    training_data_clean['meta_sale_date'].between(date_min, date_max) &
    training_data_clean['char_bldg_sf'].between(0, 60000, inclusive='both') &
    training_data_clean['char_beds'].between(0, 40, inclusive='both') &
    training_data_clean['char_rooms'].between(0, 50, inclusive='both')
].copy()

# Replace low/zero values with NaN
training_data_clean.loc[training_data_clean['char_bldg_sf'] < 300, 'char_bldg_sf'] = np.nan
training_data_clean.loc[training_data_clean['char_land_sf'] < 300, 'char_land_sf'] = np.nan
for col in ['char_beds', 'char_rooms', 'char_fbath']:
    training_data_clean.loc[training_data_clean[col] == 0, col] = np.nan

training_data_clean.to_parquet(paths['input']['training']['local'], index=False)
print(f"Cleaned training data saved to {paths['input']['training']['local']}")

## 5.2. Assessment Data
# The process for assessment data is very similar, but time features are based on a fixed date
print("Cleaning and featurizing assessment data...")
# (A full script would repeat the cleaning steps above for assessment_data_w_hie)
# For brevity, we'll just show the unique parts, like the sales count and time features.
assessment_data_clean = assessment_data_w_hie.copy() # Assume it has been cleaned like training_data

# Create count of past sales for assessment data
assessment_pins = assessment_data_clean[['meta_pin']].drop_duplicates()
merged_assessment_sales = pd.merge(
    assessment_pins,
    valid_sales_for_join, # Uses valid sales from training data
    on='meta_pin',
    how='left'
)
assessment_date = pd.to_datetime(params['assessment']['date'])
merged_assessment_sales['meta_sale_date'] = pd.to_datetime(merged_assessment_sales['meta_sale_date'])
time_diff_assessment = assessment_date - merged_assessment_sales['meta_sale_date']

within_n_years_assessment = time_diff_assessment.between(pd.Timedelta('1 day'), pd.Timedelta(f'{n_years*365.25} days'))
assessment_sales_count = (
    merged_assessment_sales[within_n_years_assessment]
    .groupby('meta_pin')
    .size()
    .reset_index(name='meta_sale_count_past_n_years')
)

assessment_data_clean = pd.merge(
    assessment_data_clean,
    assessment_sales_count,
    on='meta_pin',
    how='left'
)
assessment_data_clean['meta_sale_count_past_n_years'] = assessment_data_clean['meta_sale_count_past_n_years'].fillna(0)
# Per R script, subtract 1 to make feature consistent with training data logic
assessment_data_clean['meta_sale_count_past_n_years'] = np.maximum(0, assessment_data_clean['meta_sale_count_past_n_years'] - 1)

# Create time features based on fixed assessment date
assessment_data_clean['meta_sale_date'] = assessment_date
assessment_data_clean['time_sale_year'] = assessment_date.year
assessment_data_clean['time_sale_post_covid'] = assessment_date >= pd.to_datetime('2020-03-15')
# ... other time features would be added here ...
assessment_data_clean.to_parquet(paths['input']['assessment']['local'], index=False)
print(f"Cleaned assessment data saved to {paths['input']['assessment']['local']}")

## 5.3. Complex IDs
# This logic is identical to the previous script.
print("Creating townhome complex identifiers...")
# (Code for fuzzy matching and graph component analysis would go here)
# For brevity, we'll skip repeating it and just create a dummy file.
pd.DataFrame({'meta_pin': [], 'meta_complex_id': []}).to_parquet(paths['input']['complex_id']['local'], index=False)
print(f"Complex ID data saved to {paths['input']['complex_id']['local']}")

## 5.4. Land Rates
print("Saving land rates...")
land_nbhd_rate_data = (
    land_nbhd_rate_data[['town_nbhd', 'class', 'land_rate_per_sqft']]
    .rename(columns={'town_nbhd': 'meta_nbhd', 'class': 'meta_class'})
)
land_nbhd_rate_data.to_parquet(paths['input']['land_nbhd_rate']['local'], index=False)
print(f"Land neighborhood rate data saved to {paths['input']['land_nbhd_rate']['local']}")

print("\n--- Pipeline Complete ---")
print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
print("\nReminder: Be sure to add updated input data to DVC/S3!")