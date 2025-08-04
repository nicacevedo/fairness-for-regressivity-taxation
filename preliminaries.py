# Satandard packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

# My scripts
from src.util_functions import *
from src.reweight_and_resample import *

# CCAO's scripts
import ccao
from src.CCAO.pipeline._00_ingest_funs import recode_column_type
from src.CCAO.python.chars_funs import chars_update  

# CCAO's data sources: https://github.com/ccao-data/ccao/tree/master/data
import pyreadr

print("Imported the packages")

def root_mean_squared_error(x,y):
    return np.sqrt(mean_squared_error(x,y))



def CCAO_features(X, y, test_year=2023):
    """
        Description of the variables available on: https://github.com/ccao-data/model-res-avm?tab=readme-ov-file
    """


    # Numerical features
    num_cols = [
        "acs5_percent_age_children",
        "acs5_percent_age_senior",
        "acs5_median_age_total",
        "acs5_percent_mobility_no_move", # 2023
        "acs5_percent_mobility_moved_from_other_state", # 2023
        "acs5_percent_household_family_married", 
        "acs5_percent_household_nonfamily_alone",
        "acs5_percent_education_high_school",
        "acs5_percent_education_bachelor",
        "acs5_percent_education_graduate",
        "acs5_percent_income_below_poverty_level",
        "acs5_median_income_household_past_year",
        "acs5_median_income_per_capita_past_year",
        "acs5_percent_income_household_received_snap_past_year",
        "acs5_percent_employment_unemployed",
        "acs5_median_household_total_occupied_year_built", # Too many NA's
        "acs5_median_household_renter_occupied_gross_rent", # Too many NA's
        "acs5_percent_household_owner_occupied",
        "acs5_percent_household_total_occupied_w_sel_cond", # 2023
        "acs5_percent_mobility_moved_in_county", # 2023
        "char_beds",
        "char_bldg_sf",
        "char_fbath",
        "char_frpl",
        "char_hbath", 
        "char_land_sf",
        "char_ncu",
        "char_rooms",
        "loc_longitude",
        # "loc_latitude",
        "loc_env_flood_fs_factor",
        "loc_env_flood_fs_risk_direction", # 2023
        "loc_access_cmap_walk_nta_score", 
        "loc_access_cmap_walk_total_score",
        "loc_env_airport_noise_dnl", # "prox_airport_dnl_total", # Not present in 2022, but yes on 2023 as "loc_env_airport_noise_dnl"
        "meta_tieback_proration_rate", # 2023
        "other_tax_bill_rate",
        "other_school_district_elementary_avg_rating", # 2023: They contain only NA's in 2022, which ruins the testing
        "other_school_district_secondary_avg_rating", # 2023: They contain only NA's in 2022, which ruins the testing
        "prox_num_pin_in_half_mile",
        "prox_num_bus_stop_in_half_mile",
        "prox_num_foreclosure_per_1000_pin_past_5_years",
        "prox_num_school_in_half_mile", # 2023
        "prox_num_school_with_rating_in_half_mile", # 2023
        "prox_avg_school_rating_in_half_mile", # Too many Na's
        "prox_nearest_bike_trail_dist_ft",
        "prox_nearest_cemetery_dist_ft",
        "prox_nearest_cta_route_dist_ft",
        "prox_nearest_cta_stop_dist_ft",
        "prox_nearest_hospital_dist_ft",
        "prox_lake_michigan_dist_ft",
        "prox_nearest_major_road_dist_ft", # 2023
        "prox_nearest_metra_route_dist_ft",
        "prox_nearest_metra_stop_dist_ft",
        "prox_nearest_park_dist_ft",
        "prox_nearest_railroad_dist_ft",
        "prox_nearest_water_dist_ft",
        "prox_nearest_golf_course_dist_ft", # Not present in 2022, but present in 2023 #  prox_nearest_golf_course_dist_ft

        # Data that is not used/present in 2022 nor 2023 (?)
        # "meta_sale_count_past_n_years", # Not present in 2022
        # "shp_parcel_centroid_dist_ft_sd", # Not present in 2022
        # "shp_parcel_edge_len_ft_sd", # Not present in 2022
        # "shp_parcel_interior_angle_sd", # Not present in 2022
        # "shp_parcel_mrr_area_ratio", # Not present in 2022
        # "shp_parcel_mrr_side_ratio", # Not present in 2022
        # "shp_parcel_num_vertices", # Not present in 2022
        # "prox_nearest_university_dist_ft", # Not present in 2022
        # "prox_nearest_vacant_land_dist_ft", # Not present in 2022
        # "prox_nearest_road_highway_dist_ft", # Not present in 2022
        # "prox_nearest_road_arterial_dist_ft", # Not present in 2022
        # "prox_nearest_road_collector_dist_ft", # Not present in 2022
        # "prox_nearest_road_arterial_daily_traffic", # Not present in 2022
        # "prox_nearest_road_collector_daily_traffic",  # Not present in 2022
        # "prox_nearest_new_construction_dist_ft", # Not present in 2022
        # "prox_nearest_stadium_dist_ft", # Not present in 2022

    ]

    cat_cols = [
        "char_yrblt",
        "char_air",
        "char_apts",
        "char_attic_fnsh",
        "char_attic_type",
        "char_bsmt",
        "char_bsmt_fin",
        # "meta_class", # "char_class", # Not present in 2022 nor 2023
        # "char_ext_wall", # Not 2023
        "char_gar1_area", # 2023
        "char_gar1_att",
        "char_gar1_cnst", # Too many NA's
        "char_gar1_size",
        "char_heat", 
        "char_porch",
        "char_roof_cnst", 
        # "char_tp_dsgn", # Deprecated: Cathedral Ceiling
        "char_tp_plan", # 2023 - Design plan: Too many NA's
        "char_type_resd",
        "char_recent_renovation", # Not present in 2022, but present in 2023
        "loc_tax_municipality_name", # "loc_tax_municipality_name", # Not present in 2022/2023 with that name (also geo_municipality (?))
        "loc_env_flood_fema_sfha", # 2023 : FEMA hazard bool"
        "loc_school_elementary_district_geoid", # 2023. Not sure about how to treat them # Too many NA's
        "loc_school_secondary_district_geoid", # 2023. Not sure about how to treat them # Too many NA's
        "meta_township_code",
        "meta_nbhd_code",
        
        # "time_sale_year", # Not necesary for X?
    ]

    unclear_cols = [
        "loc_census_tract_geoid", # Census Tract GEOID: 11-digit ACS/Census tract GEOID
        # "loc_school_elementary_district_geoid",
        # "loc_school_secondary_district_geoid",
    ]

    # Cols related to sale, so don't use them(?)
    loc_cols=[
        "time_sale_day",
        "time_sale_quarter_of_year",
        "time_sale_month_of_year",
        "time_sale_day_of_year",
        "time_sale_day_of_month",
        "time_sale_day_of_week",
        "time_sale_post_covid",
    ]




    # ====== 0. Features w/ too many NA's ======
    na_threshold = 0.01 # 1%
    # X = X[[""] + num_cols + cat_cols]
    # print("min samples: ", na_threshold * X.shape[0])
    X = X.dropna(axis=1, thresh=(1 - na_threshold) * X.shape[0])
    cat_cols = [col for col in cat_cols if col in X.columns]
    num_cols = [col for col in num_cols if col in X.columns]

    # ======= 1. Categorical to categorical dtype drop too many categories =====
    X[cat_cols] = X[cat_cols].astype('category')
    # X_onehot =  pd.get_dummies(X, columns=cat_cols)
    cat_threshold = 0.001 # 0.1%
    for col in cat_cols:
        unique_values = X[col].unique()
        if unique_values.size >= cat_threshold * X.shape[0]:
            cat_cols.remove(col)
            # print(col, unique_values)
    # exit()


    # ======= 2. Categorical to categorical dtype =====
    print("All shape: ", X.shape)
    X_train = X.loc[~X["time_sale_year"].isin([test_year-1]),num_cols+cat_cols]#.dropna()#.sample(1000, random_state=42)
    print(X_train.isna().sum().sort_values(ascending=False))#.loc[X.isna().sum()])
    # print("Pre-train: ", X.loc[~X["time_sale_year"].isin([test_year-1]),num_cols+cat_cols].shape)
    print("Train shape: ", X_train.shape)
    # print("NA's train: ", X_train.isna().sum().sort_values())
    X_train = X_train.dropna()#.sample(10000)
    print("Train shape dropna: ", X_train.shape)
    X_test = X.loc[X["time_sale_year"].isin([test_year-1]),num_cols+cat_cols]#.dropna()#.sample(1000, random_state=42)
    # print("Pre-test: ", X.loc[X["time_sale_year"].isin([test_year-1]),num_cols+cat_cols].shape)
    print("Test shape: ", X_test.shape)
    # print("NA's test: ", X_test.isna().sum().sort_values())
    X_test = X_test.dropna()
    y_train, y_test = y[X_train.index], y[X_test.index]
    # X["time_sale_year"].isin([2019, 2020, 2021])
    # exit()

    # ====== Categorical Data ======
    # Check the number of cat cols that are potentially created with one-hot-encoder
    for col in cat_cols:
        print(col, X_train[col].unique().size)#, X_train[col].unique())
    # One hot encoded data
    combined_df = pd.concat([X_train, X_test], ignore_index=False)
    print("combined: ", combined_df.shape)
    combined_df = pd.get_dummies(combined_df, columns=cat_cols)
    X_train_onehot = combined_df.loc[X_train.index, :]
    X_test_onehot = combined_df.loc[X_test.index, :]





    # ===== Model Selection ===== 
    # 1. General parameters
    model_names = ["UnconstrainedNN", "LightGBM", "LinearRegression", "ConstrainedNN"]
    model_name = model_names[1]
    print("MODEL SELECTED: ", model_name)

    # inputation = False
    scaling = False


    print("First solution on a dataset of: ", X_train.shape)
    print("One hot endoded dataset of: ", X_train_onehot.shape)
    # exit()
    from time import time
    import lightgbm as lgb
    # import torch
    # import torch.nn as nn
    # import torch.optim as optim
    # from torch.utils.data import DataLoader, TensorDataset
    # from sklearn.model_selection import train_test_split
    # from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics import mean_squared_error
    # import numpy as np


    from src.nn_unconstrained import FeedForwardNNRegressor
    from src.nn_constrained_cpu_v2 import ConstrainedNNRegressor#ConstrainedMLPRegressor
    from sklearn.ensemble import IsolationForest
    # from sklearn.ensemble import ExtraTreesRegressor

    # from sklearn.neural_network import MLPRegressor
    # model = MLPRegressor(
    #     # loss="squared_error",
    #     hidden_layer_sizes=(2*X_train_onehot.shape[1],),  # Double the number of features
    #     activation="relu",
    #     solver="adam",
    #     alpha=0.001,
    #     batch_size=16,#"auto", # auto: batch_size=min(200, n_samples)
    #     learning_rate="constant",
    #     max_iter=100,
    #     random_state=42,
    #     verbose=True,
    #     # early_stopping=False,
    #     # validation_fraction=0.1,
    # )#LinearRegression(fit_intercept=True)

    if model_name == "LightGBM":
        # ============================
        # LightGBM Model (sklearn API)
        # ============================
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.1,
            random_state=42,
            # CCAO's: in between their range
            num_leaves=1000,
            add_to_linked_depth=4,
            feature_fraction=0.5,
            min_gain_to_split=10,
            min_data_in_leaf=100,
            ax_cat_threshold=100,
            min_data_per_group=100,
            cat_smooth=100,
            cat_l2=1,
            lambda_l1=1,
            lambda_l2=1,
        )
    elif model_name == "UnconstrainedNN":


        model = FeedForwardNNRegressor( # output size 1 for regression
            input_features=X_train_scaled.shape[1], 
            output_size=1, batch_size=16, learning_rate=0.001, 
            num_epochs=30, hidden_sizes=[1024]
        )

    elif model_name == "ConstrainedNN":
        model = ConstrainedNNRegressor(
            input_dim=X_train_scaled.shape[1], 
            output_size=1, batch_size=16, learning_rate=0.001, 
            num_epochs=10, hidden_layers=[1024]#, dropout_rate=0.2
        )

    elif model_name == "LinearRegression":
        model = LinearRegression(fit_intercept=True)


    # Impute the nan values
    # imputer = KNNImputer(
    #     n_neighbors=5, weights='uniform', metric='nan_euclidean', 
    #     copy=True, add_indicator=False, keep_empty_features=False
    # )
    # imputer.fit_transform(X)


    # # Sample the data
    # sample_size = 100000
    # X_train_onehot = X_train_onehot.sample(sample_size, replace=False, random_state=42)
    # y_train = y_train.loc[X_train_onehot.index]

    print("First solution on a dataset of: ", X_train.shape)
    print("One hot endoded dataset (sample) of: ", X_train_onehot.shape)

    # Fit and predict

    print("model: ", str(model))

    t0 = time()
    if model_name != "LightGBM":
        if scaling:
            # Scale features for better performance
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_onehot)
            X_test_scaled = scaler.transform(X_test_onehot)

            model.fit(X_train_scaled, y_train)

            # Prediction
            y_pred_test = model.predict(X_test_scaled)
            y_pred_train = model.predict(X_train_scaled)
        else:
            model.fit(X_train_onehot, y_train)

            # Prediction
            y_pred_test = model.predict(X_test_onehot)
            y_pred_train = model.predict(X_train_onehot)

        print(f"Fitting time {model_name}: ", time()-t0)



    else:
        print("Fitting the LightGBM model...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse', 
            callbacks=[
                lgb.early_stopping(stopping_rounds=10),  # Early stopping here
                lgb.log_evaluation(0)  # Suppress logging (use 1 for logging every round)
            ]
        )
        print(f"Fitting time {model_name}: ", time()-t0)

        # Prediction
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

    # --- Evaluate Performance ---
    # For regression, we can use metrics like Mean Squared Error (MSE).
    mse = mean_squared_error(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)

    # ratio
    n_groups_, alpha_ = 3, 2 
    r_pred_test = y_pred_test / y_test
    r_pred_train = y_pred_train / y_train
    f_metrics_test = compute_haihao_F_metrics(r_pred_test, y_test, n_groups=n_groups_, alpha=alpha_)
    f_metrics_train = compute_haihao_F_metrics(r_pred_train, y_train, n_groups=n_groups_, alpha=alpha_)

    print(f"RMSE train: {np.sqrt(train_mse):.3f}")
    print(f"RMSE val: {np.sqrt(mse):.3f}")
    print(fr"$R^2$ train: {r2_score(y_train, y_pred_train):.3f}")
    print(fr"$R^2$ val: {r2_score(y_test, y_pred_test):.3f}")
    print(fr"$F_dev$ ({alpha_}) train: {f_metrics_train.f_dev:.3f}")
    print(fr"$F_dev$ ({alpha_}) test: {f_metrics_test.f_dev:.3f}")
    print(fr"$F_grp$ ({n_groups_}) train: {f_metrics_train.f_grp:.3f}")
    print(fr"$F_grp$ ({n_groups_}) test: {f_metrics_test.f_grp:.3f}")


    # PLotting
    np.random.seed(42)
    train_sample = np.random.choice(range(y_train.size), size=1000, replace=False)
    test_sample  = np.random.choice(range(y_test.size), size=1000, replace=False)

    train_test_scatter_plots(
        y_train=y_train.iloc[train_sample], 
        y_test=y_test.iloc[test_sample], 
        y_pred_train=y_pred_train[train_sample],
        y_pred_test=y_pred_test[test_sample],
        save_plot=True,
        suffix=f"_{model_name}",
    )
    print("Plots saved (First part)!!")


    # Re-sampled version   
    # 1. Outlier scoring 
    iso = IsolationForest(
        n_estimators=100,
        contamination='auto',  # or a float like 0.05 if you know the expected outlier fraction
        # behaviour='new',       # for newest scikit‑learn versions; remove if deprecated
        random_state=42
    )
    iso.fit(X_train_onehot) 
    score_train = -iso.decision_function(X_train_onehot)

    # 2. Get scoring bins 
    n_bins = 25
    outlier_bin_edges = compute_bin_edges(score_train, num_bins=n_bins +1)
    outlier_bin_indices = get_bin_indices_from_edges(score_train, outlier_bin_edges)

    # 3. Re-sampler
    csmote = OutlierSmoteResampler(
        bin_indices=outlier_bin_indices, 
        k_neighbors=20, 
        metric="minkowski", 
        p=2, 
        random_state=45, 
        bin_size_ratio=0.05, # Ratio of the max-min count to define as goal 
        undersampling_policy="random"
    )  # New one
    X_train_resampled, y_train_resampled = csmote.fit_resample(X_train_onehot, y_train)

    print("Second solution on a dataset of: ", X_train_onehot.shape)

    # 4. Re-train model
    t0 = time()
    if model!= "LightGBM":
        if scaling:
            X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
            # X_train_scaled = scaler.transform(X_train_onehot) # Also changes when scaling the resampled (?)
            X_test_scaled = scaler.transform(X_test_onehot) # Also changes when scaling the resampled (?)
        
            model.fit(
                X_train_resampled_scaled, pd.Series(y_train_resampled),
            )
            print("Fitting time 2: ", time()-t0) 

            # Predict
            t0 = time()
            y_pred_test = model.predict(X_test_scaled)
            y_pred_train = model.predict(X_train_resampled_scaled)
            print("Predictive time 2: ", time()-t0)
        
        else:

            model.fit(X_train_resampled, pd.Series(y_train_resampled))
            print("Fitting time 2: ", time()-t0) 

            # Predict
            t0 = time()
            y_pred_test = model.predict(X_test_onehot)
            y_pred_train = model.predict(X_train_resampled)
            print("Predictive time 2: ", time()-t0)



        
    else:
        model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_test_onehot, y_test)],
            eval_metric='rmse', 
            callbacks=[
                lgb.early_stopping(stopping_rounds=10),  # Early stopping here
                lgb.log_evaluation(0)  # Suppress logging (use 1 for logging every round)
            ]
        )
        print("Fitting time 2: ", time()-t0) 

        t0 = time()
        # Predict
        y_pred_test = model.predict(X_test_onehot)
        y_pred_train = model.predict(X_train_resampled)
        print("Predictive time 2: ", time()-t0)

    # --- Evaluate Performance ---
    # For regression, we can use metrics like Mean Squared Error (MSE).
    mse = mean_squared_error(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)

    print(" == Resampled Metrics == ")
    # ratio
    n_groups_, alpha_ = 3, 2 
    r_pred_test = y_pred_test / y_test
    r_pred_train = y_pred_train / y_train
    f_metrics_test = compute_haihao_F_metrics(r_pred_test, y_test, n_groups=n_groups_, alpha=alpha_)
    f_metrics_train = compute_haihao_F_metrics(r_pred_train, y_train, n_groups=n_groups_, alpha=alpha_)

    print(f"RMSE train: {np.sqrt(train_mse):.3f}")
    print(f"RMSE val: {np.sqrt(mse):.3f}")
    print(fr"$R^2$ train: {r2_score(y_train, y_pred_train):.3f}")
    print(fr"$R^2$ val: {r2_score(y_test, y_pred_test):.3f}")
    print(fr"$F_dev({alpha_})^2$ train: {f_metrics_train.f_dev:.3f}")
    print(fr"$F_dev({alpha_})^2$ test: {f_metrics_test.f_dev:.3f}")
    print(fr"$F_grp({n_groups_})^2$ train: {f_metrics_train.f_grp:.3f}")
    print(fr"$F_grp({n_groups_})^2$ test: {f_metrics_test.f_grp:.3f}")


    
    train_test_scatter_plots(
        y_train=y_train.iloc[train_sample], 
        y_test=y_test.iloc[test_sample], 
        y_pred_train=y_pred_train[train_sample],
        y_pred_test=y_pred_test[test_sample],
        save_plot=True,
        suffix=f"_{model_name}_resampled",
    )
    print("Plots (Part 2) saved!!")
    








































def main():
    source = "county" # "toy_data"
    year = 2025

    if source == "toy_data":
        # Toy dataset
        df = pd.read_csv("data/toy_data.csv")
        df.head()

        y = df["Price"]
        X = df.drop(columns=["Price"])

    elif source == "House":
        # Kaggle House Pricing dataset
        df = pd.read_csv("data/Housing.csv")

        # Get dummies of categorial
        df = pd.get_dummies(df, drop_first=True)

        # Add a constant columns
        # df["intercept"] = 1

        # display(df.head())

        y = df["price"]
        X = df.drop(columns=["price"])
    elif source == "California":
        # Data from Google Colab samples
        df = pd.read_csv("data/california_housing_train.csv")
        # Add a constant columns
        # df["intercept"] = 1

        # Drop outliers at y.max() (too many to be true. Must be a threshold)
        df = df.loc[df["median_house_value"] < df["median_house_value"].max(),:]

        y = df["median_house_value"]
        X = df.drop(columns=["median_house_value"])
    elif source == "county":
        # Kaggle training dataset
        training_data = pd.read_parquet(f"data_county/{year}/training_data.parquet")
        assessment_data = pd.read_parquet(f"data_county/{year}/assessment_data.parquet")
        

        # # ===== Complementary datasets: 2023 =====
        # complex_id_data = pd.read_parquet(f"data_county/{year}/complex_id_data.parquet")
        # land_nbhd_rate_data = pd.read_parquet(f"data_county/{year}/land_nbhd_rate_data.parquet")
        # land_site_rate_data = pd.read_parquet(f"data_county/{year}/land_site_rate_data.parquet")

        # ===== Complementary datasets: 2025 =====
        complex_id_data = pd.read_parquet(f"data_county/{year}/complex_id_data.parquet")
        char_data = pd.read_parquet(f"data_county/{year}/char_data.parquet")
        hie_data_training_sparse = pd.read_parquet(f"data_county/{year}/hie_data.parquet")
        land_nbhd_rate_data = pd.read_parquet(f"data_county/{year}/land_nbhd_rate_data.parquet")
        
        # ===== Dictionary of variable names 
        # chars_cols = pyreadr.read_r('data_ingest/chars_cols.rda')
        df_vars_dict = ccao.vars_dict
        # print(ccao.vars_dict)



        update_hie = False
        if update_hie:
            vars_dict_1 = dict(zip(df_vars_dict['var_name_hie'], df_vars_dict['var_name_athena'])) # reversed to get rid of nan values
            vars_dict_2 = dict(zip(df_vars_dict['var_name_hie'], df_vars_dict['var_name_model']))
            for dict_ in [vars_dict_1, vars_dict_2]:
                if np.nan in dict_:
                    dict_.pop(np.nan)
                dict_ = {value: key for key, value in dict_.items()} # swap the dictionary

            vars_dict_1["year"] = "year"  
            # print(vars_dict_1)
            vars_dict_2["pin"] = "meta_pin"
            # vars_dict_1["qu_pin"] = "meta_pin"
            
            # print(vars_dict_1)
            # print("-"*100)
            # print(vars_dict_2)
            hie_data_training_sparse = hie_data_training_sparse.rename(columns=vars_dict_2)
            # print("AAA: ", hie_data_training_sparse.shape)
            # print(hie_data_training_sparse.head())
            # print(training_data.loc[training_data["ind_pin_is_multicard"],:])
            # print(hie_data_training_sparse["ind_pin_is_multicard"])
            # print(vars_dict_1["meta_pin"])
            # exit()

            # Filter the columns that are not in training_data
            cols_to_keep = [col for col in hie_data_training_sparse.columns if col in training_data.columns]
            # print(cols_to_keep)
            # print(hie_data_training_sparse[cols_to_keep].head())
            # head_pins = hie_data_training_sparse.head(10)["pin"] 
            # print(training_data.loc[training_data["meta_pin"].isin(head_pins), ["meta_pin", "char_bsmt", "char_bsmt_fin"]])
            # print(hie_data_training_sparse.head(10)[["pin", "qu_basement_type", "qu_basement_finish"]])

            print("nans in char_fbath")
            print(training_data["char_fbath"].isna().sum())

            # Merge datasets on the translated keys
            # print(training_data.head()["char_hbath"])
            training_data = pd.merge(training_data, hie_data_training_sparse[cols_to_keep], how='left', on=["meta_pin", "year"], suffixes=('_old', '_new'))#, "ind_pin_is_multicard"])


            print("nans in char_fbath")
            print(training_data["char_fbath_old"].isna().sum())

            
            print("-"*100)
            print(training_data["char_fbath_old"].unique())
            print(training_data["char_fbath_new"].unique())
            print("-"*100)

            # Update info of new columns from HIE
            for col in cols_to_keep:
                new_col = col + "_new"
                old_col = col + "_old"
                if new_col in training_data.columns and old_col in training_data.columns:
                    training_data[[old_col, new_col]] = training_data[[old_col, new_col]].astype(object)
                    print(col, "new", training_data[new_col].unique())
                    print(col, "old", training_data[old_col].unique())
                    index_to_update = ~training_data[new_col].isna()
                    # print(col, index_to_update[index_to_update].index)
                    training_data.loc[index_to_update, old_col] = training_data.loc[index_to_update, new_col]
                    del training_data[new_col]
                    training_data.rename(columns={old_col: col}, inplace=True)
                    
            print(training_data.head())        




        
        # # ===== Faile try to translate R code ===== (To be revised again)
        # # Step 1: Apply recode_column_type to target columns
        # for col in chars_cols['add']['target']:
        #     training_data[col] = recode_column_type(training_data[col], col)
        # # Step 2: Left join with hie_data_training_sparse on specified keys
        # training_data = training_data.merge(
        #     hie_data_training_sparse,
        #     how='left',
        #     left_on=['meta_pin', 'year', 'ind_pin_is_multicard'],
        #     right_on=['pin', 'year', 'ind_pin_is_multicard']
        # )
        # # Step 3: Replace 'qu_class' with 'meta_class' where qu_class == "288"
        # training_data['qu_class'] = training_data.apply(
        #     lambda row: row['meta_class'] if row['qu_class'] == "288" else row['qu_class'],
        #     axis=1
        # )
        # # Step 4: Apply ccao::chars_update equivalent
        # training_data = chars_update(
        #     training_data,
        #     additive_target=chars_cols['add']['target'],
        #     replacement_target=chars_cols['replace']['target']
        # )
        # # Step 5: Drop columns starting with 'qu_'
        # training_data = training_data.drop(columns=[col for col in training_data.columns if col.startswith('qu_')])
        # # Step 6: Replace NA in hie_num_active with 0
        # training_data['hie_num_active'] = training_data['hie_num_active'].fillna(0)
        # # Step 7: Recode char_porch values: "3" -> "0"
        # training_data['char_porch'] = training_data['char_porch'].replace("3", "0")
        # # Step 8: Relocate hie_num_active before meta_cdu column
        # cols = list(training_data.columns)
        # cols.remove('hie_num_active')
        # idx = cols.index('meta_cdu')
        # cols = cols[:idx] + ['hie_num_active'] + cols[idx:]
        # training_data = training_data[cols]

            



            # WARNING: Check the definition of the features of the dataset 
        
        # ===== Split in objective vector and features =====
        y = training_data["meta_sale_price"]
        X = training_data.drop(columns=["meta_sale_price"])




        # print(X["time_sale_year"].unique())
        # print(X.loc[X["time_sale_year"].isin([2023-1]),:])
        # exit()
        # print(X.isna().sum())
        # # Filter to use only the features CCAO is using, according to: https://github.com/ccao-data/model-res-avm/blob/master/docs/data-dict.csv
        # THIS APPEAR TO BE NEWER ONES, SO DON'T USE THEM FOR 2023, FOR EXAMPLE
        # used_feature_names = pd.read_csv(f"data_county/{year}/data-dict.csv",  usecols=['variable_name'])
        # X = X[used_feature_names['variable_name'].to_list()]

        # print(X.head())
        # exit()



    # Preprocessing 
    # print(X)
    # print(X.isna().sum())
    # print(y.isna().sum())
    CCAO_features(X, y, test_year=2023)

if __name__ == "__main__":
    main()