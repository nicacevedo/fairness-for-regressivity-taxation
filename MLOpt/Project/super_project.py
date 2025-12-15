import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sklearn processing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge, PoissonRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR

# Sklearn models

# Sklearn evaluation
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

# Custom models
from src.fair_models import ConstantModel, ModelStacking, ContextualFairStacking, StableRegression, LeastAbsoluteDeviationRegression, LogResidualRegression, LexicographicFairRegressor, ProportionalFairRegressor
from src.fair_models import StableCovarianceUpperBoundLADRegressor, StableRegressionOld

# Custom utils
from src.fair_metrics import fairness_metrics, plot_grouped_bar, plot_heatmap, plot_radar_chart
from src.general_utils import plotting_names, plotting_shapes

data_source = "student"
seed = 42

if data_source == "adult":
    # Adult
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'
    ]

    df = pd.read_csv("./datasets/adult.data", 
                        header=None, 
                        names=columns, 
                        skipinitialspace=True)
    df_test = pd.read_csv("./datasets/adult.test", 
                        header=None, 
                        names=columns, 
                        skipinitialspace=True,
                        skiprows=1,
                        )
    df.replace(' ?', np.nan, inplace=True)
    df_test.replace(' ?', np.nan, inplace=True)
    target_name = "hours-per-week"
    sensitive_feature = "sex"

    print("Original dataset:")
    print(df.head())
    print(df.shape)

    # Data to datatframe
    # df_train = pd.DataFrame(train_set.data, columns=train_set.columns)
    # df_test = pd.DataFrame(test_set.data, columns=test_set.columns)
    seed = 42

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # print("Numerical:", num_cols)
    try:
        num_cols.remove("education-num")
        num_cols.remove(target_name)
        num_cols.remove(sensitive_feature)
    except Exception as e:
        pass
    for col in num_cols:
        df[col] = df[col].astype(float)
        df_test[col] = df_test[col].astype(float)
    # sensitive_feature = "sex"
    X, y = df.drop(columns=[sensitive_feature, target_name]), df[target_name]
    X_test, y_test = df_test.drop(columns=[sensitive_feature, target_name]), df_test[target_name]

    oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop="first")
    X_ = oh.fit_transform(X.drop(columns=num_cols, errors='ignore'))
    X_ = pd.DataFrame(X_, columns=oh.get_feature_names_out(), index=X.index)
    X = pd.concat([X_, X[num_cols]], axis=1)

    X_ = oh.fit_transform(X_test.drop(columns=num_cols, errors='ignore'))
    X_ = pd.DataFrame(X_, columns=oh.get_feature_names_out(), index=X_test.index)
    X_test = pd.concat([X_, X_test[num_cols]], axis=1)


elif data_source == "student":
    # Student
    df = pd.read_csv("datasets/student-mat.csv", sep=";")
    df = df.loc[df.G3 > 0,:]
    print(df.head())

    target_name = "G3"
    sensitive_feature = "sex"
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove(target_name)
    if sensitive_feature in num_cols:
        num_cols.remove(sensitive_feature)


    X, y = df.drop(columns=[sensitive_feature, target_name]), df[target_name]
    # X_test, y_test = df_test.drop(columns=[sensitive_feature, target_name]), df_test[target_name]

    oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop="first")
    X_ = oh.fit_transform(X.drop(columns=num_cols, errors='ignore'))
    X_ = pd.DataFrame(X_, columns=oh.get_feature_names_out(), index=X.index)
    X = pd.concat([X_, X[num_cols]], axis=1)





# Train test split
if data_source != "adult":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)
else:
    X_train, y_train = X.copy(), y.copy()

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=seed)

print("Size of the training set:", X_train.shape)

# # Normalization
# scaler = StandardScaler()
# X_train.loc[:, num_cols] = scaler.fit_transform(X_train.loc[:, num_cols])
# X_val.loc[:, num_cols] = scaler.transform(X_val.loc[:, num_cols])
# X_test.loc[:, num_cols] = scaler.transform(X_test.loc[:, num_cols])
# print("Final training set")
# print(X_train.head())




do_optimal_bining = False
if do_optimal_bining:
    ##############
    # Optimal binning/clustering
    from optimal_clustering_binning import solve_optimal_binning_misocp, plot_binning_results, validate_solution, solve_optimal_binning_cutting_plane, plot_binning_results_cutting_planes

    X_sub = X_train.sample(100, random_state=42).iloc[:,:5]#[2, -2, -1]] #6, 7,
    print(X_sub.shape)
    y_sub = y_train.loc[X_sub.index]

    best_labels, best_ell, best_u, best_ss_dists, best_K = None, None, None, np.inf, 0
    tightness_eps = 1
    for K in range(2, 6):
        ####################################
        # Direct Optimization in One Step
        # # X: (n,p), y: (n,)
        labels, ell, u, details, ss_dists = solve_optimal_binning_misocp(
            X_sub.to_numpy(), y_sub.to_numpy(),
            K=K,
            rho=5.0,
            m_min=3,
            lambda_width=0,#0.1,   # optional
            width_max=None,     # optional
            standardize_X=True,
            verbose=False,
            time_limit=120,
            eps=tightness_eps,
        )


        print(f"Squared Distances Sum (K={K}):", ss_dists)
        print("status:", details["status"])
        print("ell:", ell)
        print("u:", u)
        print("outliers:", np.sum(labels == -1))
        bin_sizes = u-ell
        print("Bin sizes: ", bin_sizes)
        if np.any(bin_sizes <= (y_sub.max() - y_sub.min())/(10*K) ):
            print("No more information gain (in terms of y) in K=", K)
            print("Best K=", best_K)
            break 
        if ss_dists <= best_ss_dists:
            best_labels, best_ell, best_u, best_ss_dists, best_K = labels, ell, u, ss_dists, K # Update the best results
            plot_binning_results(X_sub.to_numpy(), y_sub.to_numpy(), labels, ell, u, show_pca=True, title_prefix="MISOCP")

        # #############################
        # # Cutting planes verions
        # labels, ell, u, details, ss_dists = solve_optimal_binning_cutting_plane(
        #     X_sub.to_numpy(), y_sub.to_numpy(),
        #     K=K,
        #     rho=5.0,
        #     m_min=2,
        #     lambda_width=0,
        #     width_max=None,
        #     standardize_X=True,
        #     max_iters=100,
        #     feas_tol=1e-4,
        #     verbose=False,
        # )

        # print(f"Squared Distances Sum (K={K}):", ss_dists)
        # print(details["status"], "obj=", details["objective"], "cuts=", details["cuts_count"])
        # print("outliers:", np.sum(labels == -1))
        # bin_sizes = u-ell
        # print("Bin sizes: ", bin_sizes)
        # if np.any(bin_sizes <= (y_sub.max() - y_sub.min())/(10*K) ):
        #     print("No more information gain (in terms of y) in K=", K)
        #     print("Best K=", best_K)
        #     break 
        # if ss_dists <= best_ss_dists:
        #     best_labels, best_ell, best_u, best_ss_dists, best_K = labels, ell, u, ss_dists, K # Update the best results
        #     plot_binning_results(X_sub.to_numpy(), y_sub.to_numpy(), labels, ell, u, show_pca=True, title_prefix="CP-OA")


    # plot_binning_results_cutting_planes(X_sub.to_numpy(), y_sub.to_numpy(), labels, ell, u, show_pca=True, title_prefix="CP-OA")

    exit()



# import seaborn as sns
# bins=15
# # plt.hist(y_train, bins=bins)
# sns.histplot(x=y_train, hue=d, multiple="layer")   # or "dodge", "stack"
# plt.savefig("./temp/plots/literal_trash/hist_1.pdf", dpi=600)
# # plt.hist(y_val, bins=bins)
# sns.histplot(x=y_val, hue=d, multiple="layer")   # or "dodge", "stack"
# plt.savefig("./temp/plots/literal_trash/hist_2.pdf", dpi=600)
# exit()



# Model settings
import warnings
warnings.filterwarnings('ignore')

# Sensitive feature: list of indices for each group
sensitive_idx = []
df_train = df.loc[df.index.isin(X_train.index), :].reset_index()
for value_ in df[sensitive_feature].unique():
    df_ = df_train.loc[df_train[sensitive_feature] == value_, :]
    sensitive_idx.append(list(df_.index))

sensitive_idx_val = []
df_val = df.loc[df.index.isin(X_val.index), :].reset_index()
for value_ in df[sensitive_feature].unique():
    df_ = df_val.loc[df_val[sensitive_feature] == value_, :]
    sensitive_idx_val.append(list(df_.index))


max_depth = 7
n_estimators = 500
big_M = 1e4
models = [

    # LogResidualRegression(
    #     loss= "mse",
    #     fit_intercept= True,
    #     l2_reg= 1e-3,
    #     eps= 1/big_M,
    #     max_iter= 500,
    #     tol= 1e-5,
    #     weight_cap= big_M,
    #     verbose= True,
    # ),

   
    # BayesianRidge(fit_intercept=True, ), 
    # PoissonRegressor(fit_intercept=True), 
    # # QuantileRegressor(fit_intercept=True, ),
    RandomForestRegressor(
        n_estimators=200,
        max_depth=max_depth,
        min_samples_split=0.1,
        # max_features='auto',  # or 'sqrt' for large feature sets
        random_state=42,
        n_jobs=-1
    ),
    GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=max_depth,
        subsample=1.0,
        min_samples_split=2,
        random_state=42
    ),
    ExtraTreesRegressor(
        n_estimators=200,
        max_depth=max_depth,
        min_samples_split=0.1,
        min_samples_leaf=1,
        # max_features='sqrt',
        bootstrap=True,
        random_state=42
    ),
    # LexicographicFairRegressor(
    #     groups=sensitive_idx,
    #     loss="mae",
    #     tol=1e-6,
    #     fit_intercept=True,
    #     solver="MOSEK",
    # ),

    # ProportionalFairRegressor(
    #     groups=sensitive_idx,
    #     loss = "mae",
    #     fit_intercept = True,
    #     solver = "MOSEK",
    #     baseline_mode="variance",
    # ),
    # KNeighborsRegressor(
    #     n_neighbors=15,
    #     weights='uniform',
    #     metric='euclidean',
    #     p=2,
    #     leaf_size=30,
    #     algorithm='auto'
    # ),
    LeastAbsoluteDeviationRegression(fit_intercept=True, fit_group_intercept=False, solver="MOSEK"),
    LinearRegression(fit_intercept=True), 
    # DecisionTreeRegressor(max_depth=max_depth),

    # ConstantModel(sensitive_idx=sensitive_idx),
    # LinearRegression(fit_intercept=True), 
    # LinearSVR(fit_intercept=True)
    # ElasticNet(fit_intercept=True, alpha=1e-3, l1_ratio=0.1), 

]

d = pd.get_dummies(df.loc[:,sensitive_feature]).iloc[:,1] # BINARY FOCUSED

keep_percentages = [1.0, 0.8, 0.6, 0.4]
for keep_per in keep_percentages:#np.linspace(0.9, 0.55, 3):
    models.append(StableRegressionOld(
        fit_intercept=True, k_percentage=keep_per, lambda_l1=1e-4, lambda_l2=1e-4,
        fit_group_intercept=False, delta_l2=0, group_constraints=False, weight_by_group=False, 
        sensitive_idx=sensitive_idx, sensitive_feature=d.loc[X_train.index].to_numpy(),
        group_percentage_diff=0,
    ))
    
for keep_per in keep_percentages:#np.linspace(0.9, 0.55, 3):
    models.append(StableRegression(
        fit_intercept=True, k_percentage=keep_per, lambda_l1=1e-4, lambda_l2=1e-4,
        fit_group_intercept=False, delta_l2=0, group_constraints=True, weight_by_group=False, 
        sensitive_idx=sensitive_idx, sensitive_feature=d.loc[X_train.index].to_numpy(),
        group_percentage_diff=10,
    ))
    
for keep_per in keep_percentages:#np.linspace(0.9, 0.55, 3):
    # models.append(StableRegression(
    #     fit_intercept=True, k_percentage=keep_per, lambda_l1=1e-4, lambda_l2=1e-4,
    #     fit_group_intercept=False, delta_l2=0, group_constraints=True, weight_by_group=True, 
    #     sensitive_idx=sensitive_idx, sensitive_feature=d.loc[X_train.index].to_numpy(),
    #     group_percentage_diff=10,
    # ))
    models.append(
            StableCovarianceUpperBoundLADRegressor(
            keep=keep_per,      # K = ceil(0.10 * n)
            rho=1.0,        # fairness weight
            l1=1e-4,
            l2=1e-4,
            fit_intercept=True,
            solver="MOSEK",
        )
    )

    #     model.fit(X_train, y_train, d=d_train)  # d only used here
    # yhat = model.predict(X_test)
    # wc_idx = model.worst_case_indices_()
    # print(model.status_, model.objective_value_, wc_idx[:10])

    

# for keep_per in np.linspace(0.9, 0.5, 3):
#     models.append(StableRegression(fit_intercept=True, k_percentage=keep_per, lambda_l1=1e-2, lambda_l2=1e-4,
#                                    fit_group_intercept=False, delta_l2=0, group_constraints=True, weight_by_group=False, 
#                                     sensitive_idx=sensitive_idx, group_percentage_diff=10, sensitive_feature=d.loc[X_train.index].to_numpy()))
    
model_names = [str(model).split("(")[0]+str(i) for i,model in enumerate(models)]
# Fairness notions: 
#   1. Error disparity: E[residuals^2| d] similarity
#   2. No correlation between prediction and sensitive: Corr(d, y)
#   3. No correlation between sensitive and residuals: Corr(d, residuals) 
#   4. Distribution of y has to be similar: E(y|d) similarity | Var(y|d) similarity 

# fairness_metrics = {
#     "Error disparity: max_d E[res^2 | d] - min_d E[res^2 | d]":[],
#     "Prediction-sensitive correlation: Corr(d, f(x))":[],
#     "Residuals-sensitive correlation: Corr(d, y-f(x))":[],
#     "Prediction disparity 1st: max_d E[f(x)|d] - min_d E[f(x)|d]":[],
#     "Prediction disparity 2st: max_d Var[f(x)|d] - min_d Var[f(x)|d]":[],
# }

do_heatmap = False
if do_heatmap:
    fairness_metrics_dict = {
        "r2_score":[],
        "error_disparity":[],
        "pred_sens_corr":[],
        "res_sens_corr":[],
        "pred_mean":[],
        "pred_std":[],
    }

    fairness_metrics_dict_val = {
        "r2_score":[],
        "error_disparity":[],
        "pred_sens_corr":[],
        "res_sens_corr":[],
        "pred_mean":[],
        "pred_std":[],
    }

    # training_metrics = []
    # validation_metrics = []
    for idx, model in enumerate(models):
        try:
            model.fit(X_train, y_train)
        except Exception:
            model.fit(X_train, y_train, d=d.loc[X_train.index])  # d only used here
        y_pred = model.predict(X_train)
        print("MODEL: ", idx, model)
        # print("RMSE: ", root_mean_squared_error(y_train, y_pred))
        rmse_list = []
        for g, g_idx in enumerate(sensitive_idx):
            y_g = y_train.iloc[g_idx]
            try:
                y_pred = y_pred.to_numpy()
            except Exception:
                pass
            rmse_list.append(root_mean_squared_error(y_g, y_pred[g_idx]))
        # print("RMSE list: ", rmse_list)
        # print("RMSE diff: ", np.max(rmse_list) - np.min(rmse_list))


        tuple_ = fairness_metrics(y_train, y_pred, sensitive_idx, d.loc[X_train.index], fairness_type="max_min_error", error_type="rmse")
        metrics_dict = tuple_[0]
        for key in metrics_dict:
            fairness_metrics_dict[key].append(metrics_dict[key])
            print(key, metrics_dict[key])

        # Testing Set
        y_pred = model.predict(X_val)
        print("RMSE: ", root_mean_squared_error(y_val, y_pred))
        rmse_list = []
        for g, g_idx in enumerate(sensitive_idx_val):
            y_g = y_val.iloc[g_idx]
            try:
                y_pred = y_pred.to_numpy()
            except Exception:
                pass
            rmse_list.append(root_mean_squared_error(y_g, y_pred[g_idx]))
            # print("RMSE: ", g, root_mean_squared_error(y_g, y_pred[g_idx]))
        print("RMSE list: ", rmse_list)
        print("RMSE diff: ", np.max(rmse_list) - np.min(rmse_list))


        tuple_ = fairness_metrics(y_val, y_pred, sensitive_idx_val, d.loc[X_val.index], fairness_type="max_min_error", error_type="rmse")
        metrics_dict = tuple_[0]
        for key in metrics_dict:
            fairness_metrics_dict_val[key].append(metrics_dict[key])
            # print(key, metrics_dict[key])

            
        # print(metrics_dict)
    # End of direct models
    # print(fairness_metrics_dict)
    print(model_names)
    for i,name in enumerate(model_names):
        if "StableRegressionOld" in name:
            model_names[i] = "StableRegression"
        elif "StableRegression" in name:
            model_names[i] = "StableConstrained"
        elif "StableCovariance" in name:
            model_names[i] = "StableAdversarial"
        else:
            model_names[i] = name.replace("0","").replace("1","").replace("2","").replace("3","").replace("4","").replace("5","").replace("6","").replace("7","")
            model_names[i] = model_names[i].replace("Regressor", "").replace("LeastAbsoluteDeviation", "LAD")
            

    df_plots = pd.DataFrame(fairness_metrics_dict_val, index=model_names)
    # To Latex
    # Define column-specific rounding (adjust as needed)
    col_names = ["$R^2$ Score", "$R^2$ Disp.", "$|Corr(\hat{y}, d)|$", "res", "$\mathbb{E}$. Disp.", "Std. Disp."]#[name.replace("_", " ") for name in metrics_dict]
    df_plots.columns = col_names
    rounding = {name:3 for name in col_names}
    df_rounded = df_plots.round(rounding).round(3)  # Final 6-decimal polish
    del df_rounded["res"]
    print(df_rounded)
    
    def generate_latex_table(df):
        """
        Generates a LaTeX table with bold headers, bold index, vertical lines,
        and bolds the maximum value in each column.
        """
        # Create a copy to avoid modifying the original dataframe
        df_table = df.copy()
        
        # 0. Bold the maximum value in each column
        for col in df_table.columns:
            # Find the maximum value in the column
            min_val = df_table[col].min()
            max_val = df_table[col].max()

            # Apply formatting: bold string if max, otherwise standard string format
            if col != "$R^2$ Score":
                df_table[col] = df_table[col].apply(
                    lambda x: f"\\textbf{{{x:.3f}}}" if x == min_val else f"{x:.3f}"
                )
            else:
                df_table[col] = df_table[col].apply(
                    lambda x: f"\\textbf{{{x:.3f}}}" if x == max_val else f"{x:.3f}"
                )
        
        # 1. Bold Column Headers (requires escape=False)
        # We wrap the column names in \textbf{}
        df_table.columns = [f"\\textbf{{{c}}}" for c in df_table.columns]
        
        # 2. Define Column Format with vertical lines
        # 'l' for the index, 'r' for the columns. 
        # Add '|' to create vertical separators.
        # Result looks like: |l|r|r|r|r|
        n_cols = len(df_table.columns)
        col_format = '|l|' + 'r|' * n_cols
        
        latex_code = df_table.to_latex(
            index=True,
            column_format=col_format,
            # float_format="{:.3f}".format, # Removed because we formatted values manually
            escape=False,        # Necessary to render \textbf{} correctly
            caption="Model Performance Metrics",
            label="tab:performance",
            bold_rows=True       # Makes the index (Model names) bold
        )
    
        return latex_code



    def generate_diff_table(df, baseline_index="LinearRegression"):
        """
        Generates a second LaTeX table showing percentual difference 
        with respect to the baseline model (LinearRegression).
        """
        if baseline_index not in df.index:
            return f"% Error: Baseline model '{baseline_index}' not found in index."
        
        # Create copy to avoid modifying original
        df_diff = df.copy()
        
        # Get baseline values
        baseline_vals = df_diff.loc[baseline_index]
        
        # Calculate percentage diff: (Model - Baseline) / Baseline * 100
        # Broadcasting works automatically for rows.
        # We calculate numerics FIRST to enable accurate max/min finding.
        for col in df_diff.columns:
            df_diff[col] = ((df_diff[col] - baseline_vals[col]) / baseline_vals[col]) * 100

        # Format string with % sign, sign (+/-), and bold the max value
        for col in df_diff.columns:
            # Determine the value to highlight. 
            # For Accuracy/F1/Parity, 'Max' diff is usually the "best" relative change.
            if col == "$R^2$ Score":
                target_val = df_diff[col].max()
                df_diff[col] = df_diff[col].apply(
                    lambda x: f"\\textbf{{{x:+.2f}\%}}" if x == target_val else f"{x:+.2f}\%"
                )
            else:
                target_val = df_diff[col].min()
                df_diff[col] = df_diff[col].apply(
                    lambda x: f"\\textbf{{{x:+.2f}\%}}" if x == target_val else f"{x:+.2f}\%"
                )         

        # 1. Bold Column Headers
        df_diff.columns = [f"\\textbf{{{c}}}" for c in df_diff.columns]
        
        # 2. Define Column Format with vertical lines
        n_cols = len(df_diff.columns)
        col_format = '|l|' + 'r|' * n_cols
        
        latex_code = df_diff.to_latex(
            index=True,
            column_format=col_format,
            escape=False,
            caption=f"Percentage Difference vs {baseline_index}",
            label="tab:performance_diff",
            bold_rows=True
        )
        
        return latex_code



    latex_table = generate_latex_table(df_rounded)
    print(latex_table)
    diff_latex_table = generate_diff_table(df_rounded, baseline_index="LinearRegression")
    print(diff_latex_table)
    # print(df_plots.round(3).to_latex())
    print("SIZES: ", X_train.shape)
    exit()
    df_plots = (df_plots - df_plots.min()) / (df_plots.max() - df_plots.min())
    del df_plots["res_sens_corr"]
    # print(df_plots.head(20))
    # exit()

    print("ERROR DIFFS (train): ", fairness_metrics_dict["error_disparity"])
    print("ERROR DIFFS (vals): ", fairness_metrics_dict_val["error_disparity"])



    import seaborn as sns
    plt.figure(figsize=(8, 6))

    # Transpose df if you prefer Models on the Y-axis
    sns.heatmap(df_plots, annot=True, cmap='Blues', vmin=0, vmax=1, fmt=".2f", linewidths=.5)

    plt.title('Heatmap of Fairness Metrics', fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig("./temp/plots/plot_heatmap.pdf", dpi=1200)
    exit()
    # You can choose which style you prefer
    # print("Generating Grouped Bar Chart...")
    # plot_grouped_bar(df_plots)

    # print("Generating Radar Chart...")
    # plot_radar_chart(df_plots)

    # print("Generating Heatmap...")
    # plot_heatmap(df_plots)
    # exit()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Assuming X_train, y_train, sensitive_idx, and StableRegression are defined in the context

# 2. Define Experiment Configurations
# We store the specific params that change between experiments here
l1, l2 = 1e-4, 1e-4
experiment_configs = [
    # Exp 0: Baseline LAD (No regularization, No group intercept)
    # {
    #     "lambda_l1": l1, "lambda_l2": l2,
    #     "fit_group_intercept": False, "delta_l2": 0,
    #     "group_constraints": False, "weight_by_group": False
    # },
    {"fit_intercept":True},
    # {"fit_intercept":True, "alpha":2*1e-4, "l1_ratio":0.5}
    # # # Exp 1: Group Intercepts (With regularization)
    # {
    #     "lambda_l1": 1e-3, "lambda_l2": 1e-4,
    #     "fit_group_intercept": True, "delta_l2": 1e-3,
    #     "group_constraints": False, "weight_by_group": False
    # },
    # # Exp 2: Group Constraints (Strict % diff)
    # {
    #     "lambda_l1": 1e-3, "lambda_l2": 1e-4,
    #     "fit_group_intercept": True, "delta_l2": 1e-3,
    #     "group_constraints": True, "group_percentage_diff": 1,
    #     "weight_by_group": False
    # },
    # Exp 3: Weighted by Group
    # {
    #     "lambda_l1": 1e-3, "lambda_l2": 1e-4,
    #     "fit_group_intercept": True, "delta_l2": 1e-3,
    #     "group_constraints": False, "group_percentage_diff": 1e-4,
    #     "weight_by_group": True
    # }
]

# for delta_l2 in [1e-3]:#, 1e-1]:
#     experiment_configs.append({
#         "lambda_l1": l1, "lambda_l2": l2,
#         "fit_group_intercept": True, "delta_l2": delta_l2,
#         "group_constraints": False, "weight_by_group": False
#     })

# # for w_group_0 in np.linspace(0.9, 1.1, 3):
# experiment_configs.append({
#     "lambda_l1": l1, "lambda_l2": l2,
#     "fit_group_intercept": True, "delta_l2": 1e-3,
#     # "fit_group_intercept": False, "delta_l2": 0,
#     "group_constraints": False, "group_percentage_diff": 0,
#     "weight_by_group": True, #, "w_group_0":w_group_0,
# })


d = pd.get_dummies(df.loc[:,sensitive_feature]).iloc[:,1] # BINARY FOCUSED
# for l_UB in [5, 10, 15]:
experiment_configs.append({ # Regular Stable Regression
    "lambda_l1": 1e-4, "lambda_l2": 1e-4,
    "fit_group_intercept": False, "delta_l2": 0,
    "group_constraints": False, "group_percentage_diff": 0,#l_UB,
    "sensitive_feature": d.loc[X_train.index].to_numpy(),
    "weight_by_group": False,
})




# COMPARISON OF TWO STABLES 
epss = [5, 10, 15]
# for eps in epss:
#         experiment_configs.append({
#             "lambda_l1": 1e-4, "lambda_l2": 1e-4,
#             "fit_group_intercept": False, "delta_l2": 0,
#             "group_constraints": True, "group_percentage_diff": eps,#l_UB,
#             "sensitive_feature": d.loc[X_train.index].to_numpy(),
#             "residual_cov_constraint":False, #"residual_cov_thresh": eps,
#             "weight_by_group": False,
#         })

rhos = [(eps)/10 for eps in epss]
# for rho in rhos:
#     experiment_configs.append({"rho":rho})







    # d = pd.get_dummies(df.loc[:,sensitive_feature]).iloc[:,1] # BINARY FOCUSED
    # for l_UB in [1, 1e1, 20]:#[5, 10, 15]:
    #     experiment_configs.append({
    #         "lambda_l1": 1e-3, "lambda_l2": 1e-4,
    #         "fit_group_intercept": False, "delta_l2": 0,
    #         "group_constraints": True, "group_percentage_diff": 10,#l_UB,
    #         "sensitive_feature": d.loc[X_train.index].to_numpy(),
    #         "residual_cov_constraint":True, "residual_cov_thresh": l_UB,
    #         "weight_by_group": True,
    #     })

    # Assuming X_train, y_train, sensitive_idx are defined for Training
    # Assuming X_test, y_test, sensitive_idx_test are defined for Validation
    # Assuming StableRegression class is defined

# 1. Setup Parameters
keep_percentages = np.linspace(0.2, 1, 17)
n_experiments = len(experiment_configs)
n_groups = len(sensitive_idx)

# Structure: results[experiment_id][group_id] -> List of R2 scores per percentage
model_results = [[[] for _ in range(n_groups)] for _ in range(n_experiments)]
model_results_val = [[[] for _ in range(n_groups)] for _ in range(n_experiments)]

variance_results = [[[] for _ in range(n_groups)] for _ in range(n_experiments)]
variance_results_val = [[[] for _ in range(n_groups)] for _ in range(n_experiments)]

# THE REAL METRIC: corr(f(X), d)
corr_results = [[] for _ in range(n_experiments)]
corr_results_val = [[] for _ in range(n_experiments)]

# Other sense of fairness, corr between residuals: corr(y-f(X), d)
res_corr_results = [[] for _ in range(n_experiments)]
res_corr_results_val = [[] for _ in range(n_experiments)]

# Difference of expected value of f(X)
expected_diff_f_X = [[[] for _ in range(n_groups)] for _ in range(n_experiments)]

# Difference of Variance of f(X)
variance_diff_f_X = [[[] for _ in range(n_groups)] for _ in range(n_experiments)]


# 3. Main Experiment Loop
print(f"Starting experiments across {len(keep_percentages)} percentages...")
# exit()
for k_perc in keep_percentages:
    print(f"\n--- Testing Keep Percentage: {k_perc:.2f} ---")
    d_train = d.loc[X_train.index].to_numpy()
    for i, config in enumerate(experiment_configs):
        # Merge common defaults with specific experiment config
        if i == 0:
            model = LinearRegression(fit_intercept=True)
            model.fit(X_train, y_train)
        # elif i == 1:
        #     model = ElasticNet(fit_intercept=True, l1_ratio=2*1e-4, alpha=0.5)
        #     model.fit(X_train, y_train)     
        elif "lambda_l1" in config:
            params = {
                "fit_intercept": True,
                "solver": "MOSEK",
                "objective": "mae",
                "sensitive_idx": sensitive_idx,
                "k_percentage": k_perc  # Dynamic parameter from outer loop
            }
            params.update(config)
            model = StableRegression(**params)
            model.fit(X_train, y_train)

        elif "rho" in config:
            params = {
                "keep": k_perc,
                "fit_intercept":True,
                # "rho":1.0,
                "l1":1e-4,
                "l2":1e-4,
                "solver":"MOSEK",
                "verbose":False

            }
            params.update(config)
            # Initialize and Fit
            model = StableCovarianceUpperBoundLADRegressor(**params)    
            model.fit(X_train, y_train, d_train)
        # print(config)

        # --- Evaluate on TRAINING Set ---
        y_pred = model.predict(X_train)
        # print(y_pred)
        global_r2 = r2_score(y_train, y_pred)
        print(f"[Exp {i}] Train Global R2: {global_r2:.4f}")
        
        for g, g_idx in enumerate(sensitive_idx):
            if hasattr(y_train, 'iloc'):
                y_true_g = y_train.iloc[g_idx]
            else:
                y_true_g = y_train[g_idx]
            if hasattr(y_pred, 'iloc'):
                y_pred_g = y_pred.iloc[g_idx]
            else:
                y_pred_g = y_pred[g_idx]
                
            score = r2_score(y_true_g, y_pred_g)
            model_results[i][g].append(score)
            variance = np.std( np.abs(y_true_g - y_pred_g) )
            variance_results[i][g].append(variance)
        score = np.corrcoef(d.loc[X_train.index], y_pred)[0, 1]
        corr_results[i].append(score)
        print("Current   corr: ", score)
        print("Real data corr: ", np.corrcoef(d_train, y_train)[0, 1])
        # score = np.corrcoef(d.loc[X_train.index], y_train - y_pred)[0, 1]
        score = np.corrcoef(y_train, y_train - y_pred)[0, 1]
        res_corr_results[i].append(score)
        # print("Current corr wrt residuals: ", np.corrcoef(d.loc[X_train.index], y_train-y_pred)[0, 1])

        # --- Evaluate on Testing Set ---
        y_pred_val = model.predict(X_val)#model.predict(X_val)
        
        for g, g_idx in enumerate(sensitive_idx_val):
            # Ensure we use val set indices/data
            if hasattr(y_val, 'iloc'):
                y_true_g_val = y_val.iloc[g_idx]
            else:
                y_true_g_val = y_val[g_idx]
            if hasattr(y_pred_val, 'iloc'):
                y_pred_g_val = y_pred_val.iloc[g_idx]
            else:
                y_pred_g_val = y_pred_val[g_idx]

            score_val = r2_score(y_true_g_val, y_pred_g_val)
            model_results_val[i][g].append(score_val)
            variance_val = np.std( np.abs(y_true_g_val - y_pred_g_val) )
            variance_results_val[i][g].append(variance_val)

            
            # Distribution difference (info of each one here)
            expected_diff_f_X[i][g].append(np.mean(y_pred_g_val))
            variance_diff_f_X[i][g].append(np.std(y_pred_g_val))
        d_val = d.loc[X_val.index] #pd.get_dummies(df_val.loc[:,sensitive_feature]).to_numpy()[:,0] # BINARY FOCUSED
        score_val = np.corrcoef(d_val, y_pred_val)[0, 1]
        corr_results_val[i].append(score_val)
        # score_val = np.corrcoef(d.loc[X_val.index], y_val - y_pred_val)[0, 1]
        print(y_val.shape)
        print(y_pred_val.shape)
        score_val = np.corrcoef(y_val, y_val - y_pred_val)[0, 1]
        res_corr_results_val[i].append(score_val)

        # print("Current corr wrt residuals: ", np.corrcoef(d.loc[X_train.index], y_train-y_pred)[0, 1])

# print(corr_results)
# print(corr_results_val)

# 4. Plotting Results
import os
os.makedirs("./temp/plots", exist_ok=True)

# Define Styles
exp_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
group_markers = ['o', 's', '^', 'D']
group_linestyles = ['-', '--', '-.', ':']
baselines= ["LinearRegression", "StableRegression"]#, "ElasticNet"]
label_names = baselines + [f"StableConstrained(eps={eps})" for eps in epss] + [f"StableAdversarial(rho={rho})" for rho in rhos] #["|corr|<=0.8 (Baseline)","|corr|<=0.225","|corr|<=0.45","|corr|<=0.68",]

# --- Plot 1: Training R2 Scores ---
plt.figure(figsize=(8, 5))

for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    for g in range(n_groups):
        label = label_names[i]  
        linestyle = group_linestyles[g % len(group_linestyles)]
        marker = group_markers[g % len(group_markers)]
        
        plt.plot(
            keep_percentages, 
            model_results[i][g], 
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            alpha=0.8
        )

plt.xlabel("Keep Percentage (k)")
plt.ylabel("R2 Score")
plt.title("Impact of Stability on Group Fairness (Training Set)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig("./temp/plots/delete_vs_1_train.png", dpi=600)
# plt.show()

# --- Plot 0: Training corr Scores ---
plt.figure(figsize=(8, 5))
# for i in [0,3,2,1]:#
for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    label = label_names[i]
    plt.plot(
        keep_percentages, 
        corr_results[i], 
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        alpha=0.8
    )


plt.xlabel("Keep Percentage (k)")
plt.ylabel(r"Correlation Fairness ( $Corr(d, f(x))$ )")
plt.title("Impact of Covariance Fairness Constaint\n in Stable Regression (Training Set)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.7)
plt.tight_layout()
# plt.savefig("./temp/plots/delete_vs_0_correlation.png", dpi=600)
plt.savefig("./temp/plots/delete_vs_0_correlation.pdf", dpi=1200)
# plt.show()

# --- Plot 0: Training corr Scores ---
plt.figure(figsize=(8, 5))
for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    label = label_names[i]
    alpha_ = 0.3 if i <= 4 and i>1 else 0.8
    plt.plot(
        keep_percentages, 
        corr_results_val[i], 
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        alpha=alpha_
    )
plt.xlabel(r"$\mathbf{Rate\;of\;Samples\;to\;Keep:\;}$ $r = K/n$")
plt.ylabel(r"$\mathbf{Correlation\;(Un)fairness:\;}$ $\text{Corr}(\;d,\;f(x)\;)$ ")
plt.title(r"$\mathbf{Impact\;of\;Correlation\;Fairness\;Constaint}$" "\n" r"$\mathbf{in\;Stable\;Regression}\;[Testing\;Set]$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig("./temp/plots/delete_vs_0_correlation_val.pdf", dpi=1200)
# plt.show()

# # --- Plot 01: Training corr Scores ---
# plt.figure(figsize=(10, 5))

# for i in range(n_experiments):
#     color = exp_colors[i % len(exp_colors)]
#     label = label_names[i]
#     plt.plot(
#         keep_percentages, 
#         res_corr_results[i], 
#         label=label,
#         color=color,
#         linestyle=linestyle,
#         marker=marker,
#         alpha=0.8
#     )

# plt.xlabel("Keep Percentage (k)")
# plt.ylabel("Correlation: Sensitive Feat. | Residuals")
# plt.title("Impact of Stability on Group Fairness (Training Set)")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("./temp/plots/delete_vs_1_correlation.png", dpi=600)
# # plt.show()

# # --- Plot 01: Training corr Scores ---
# plt.figure(figsize=(8, 5))
# # plt.axhline(y=0, color="r", linewidth=2, label="Perfect Score")

# for i in range(n_experiments):
#     color = exp_colors[i % len(exp_colors)]
#     label = label_names[i]
#     plt.plot(
#         keep_percentages, 
#         res_corr_results_val[i], 
#         label=label,
#         color=color,
#         linestyle=linestyle,
#         marker=marker,
#         alpha=0.8
#     )

# plt.xlabel("Keep Percentage (k)")
# plt.ylabel("Correlation: Sensitive Feat. | Residuals")
# plt.title("Impact of Stability on Group Fairness (Testing Set)")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("./temp/plots/delete_vs_1_correlation_val.png", dpi=600)
# # plt.show()

# --- Plot 2: Training R2 Disparity ---
plt.figure(figsize=(8, 5))

for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    
    r2_g0 = np.array(model_results[i][0])
    r2_g1 = np.array(model_results[i][1])
    diff = np.abs(r2_g0 - r2_g1)
    label = label_names[i]
    plt.plot(
        keep_percentages, 
        diff, 
        label=label,
        color=color,
        marker='o',
        linewidth=2
    )

plt.xlabel("Keep Percentage (k)")
plt.ylabel("R2 Disparity (|Group0 - Group1|)")
plt.title("Fairness Gap vs Stability (Training Set)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./temp/plots/delete_vs_2_disparity.png", dpi=600)
# plt.show()

# --- Plot 2: Training R2 Disparity ---
plt.figure(figsize=(8, 5))

for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    
    r2_g0 = np.array(model_results[i][0])
    r2_g1 = np.array(model_results[i][1])
    diff = np.abs(r2_g0 - r2_g1)
    label = label_names[i]
    
    plt.plot(
        keep_percentages, 
        variance_results[i][g], 
        label=label,
        color=color,
        marker='o',
        linewidth=2
    )

plt.xlabel("Keep Percentage (k)")
plt.ylabel("Residuals Variance")
plt.title("Fairness Gap vs Stability (Training Set)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./temp/plots/delete_vs_5_variance.png", dpi=600)
# plt.show()


# --- Plot 3: Validation R2 Scores ---
plt.figure(figsize=(8, 5))
for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    for g in range(n_groups):
        label = label_names[i] + f" [ d={g} ]"
        linestyle = group_linestyles[g % len(group_linestyles)]
        marker = group_markers[g % len(group_markers)]
        alpha_ = 0.3 if i <= 4 and i>1 else 0.8

        plt.plot(
            keep_percentages, 
            model_results_val[i][g], 
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            alpha=alpha_
        )
plt.xlabel(r"$\mathbf{Rate\;of\;Samples\;to\;Keep:\;}$ $r = K/n$")
plt.ylabel(r"$\mathbf{Accuracy\;by\;group:}$ $R^2$ Score")
plt.title(r"$\mathbf{Impact\;of\;Correlation\;Fairness\;Constaint}$" "\n" r"$\mathbf{in\;Stable\;Regression}\;[Testing\;Set]$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.7)
plt.ylim([0.86, 0.932])
plt.tight_layout()
plt.savefig("./temp/plots/delete_vs_3_validation.pdf", dpi=1200)

# --- Plot 4: Val R2 Disparity ---
plt.figure(figsize=(8, 5))

for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    r2_g0 = np.array(model_results_val[i][0])
    r2_g1 = np.array(model_results_val[i][1])
    diff = np.abs(r2_g0 - r2_g1)
    label = label_names[i]
    alpha_ = 0.3 if i <= 4 and i>1 else 0.8
    plt.plot(
        keep_percentages, 
        diff, 
        label=label,
        color=color,
        marker='D',
        linewidth=1,
        alpha=alpha_
    )
plt.xlabel(r"$\mathbf{Rate\;of\;Samples\;to\;Keep:\;}$ $r = K/n$")
plt.ylabel(r"$\mathbf{Accuracy Disparity:}$ $|R^2_{d_1} - R^2_{d_2}|$")
plt.title(r"$\mathbf{Impact\;of\;Correlation\;Fairness\;Constaint}$" "\n" r"$\mathbf{in\;Stable\;Regression}\;[Testing\;Set]$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig("./temp/plots/delete_vs_4_disparity.pdf", dpi=1200)


# --- Plot 2: Training R2 Disparity ---
plt.figure(figsize=(8, 5))
for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    
    r2_g0 = np.array(model_results[i][0])
    r2_g1 = np.array(model_results[i][1])
    diff = np.abs(r2_g0 - r2_g1)
    label = label_names[i]
    
    plt.plot(
        keep_percentages, 
        variance_results_val[i][g], 
        # label=f"Exp {i} Variance",
        label=label,
        color=color,
        marker='o',
        linewidth=2
    )

plt.xlabel("Keep Percentage (k)")
plt.ylabel("Residuals Variance")
plt.title("Fairness Gap vs Stability (Testing Set)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./temp/plots/delete_vs_6_variance.png", dpi=600)

# --- Plot 7: Distrib disparity ---
plt.figure(figsize=(8, 5))
for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    
    r2_g0 = np.array(expected_diff_f_X[i][0])
    r2_g1 = np.array(expected_diff_f_X[i][1])
    diff = np.abs(r2_g0 - r2_g1)
    label = label_names[i]
    alpha_ = 0.3 if i <= 4 and i>1 else 0.8
    plt.plot(
        keep_percentages, 
        diff, 
        label=label,
        color=color,
        marker='^',
        linewidth=1,
        ls='dashed',
        alpha=alpha_
    )
plt.xlabel(r"$\mathbf{Rate\;of\;Samples\;to\;Keep:\;}$ $r = K/n$")
plt.ylabel(r"$\mathbf{Distribution\;Disparity\;(Expectation):}$ $| E(\hat{y}|d_1) - E(\hat{y}|d_2) |$")
plt.title(r"$\mathbf{Impact\;of\;Correlation\;Fairness\;Constaint}$" "\n" r"$\mathbf{in\;Stable\;Regression}\;[Testing\;Set]$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig("./temp/plots/delete_vs_7_expected_f_X_val.pdf", dpi=1200)


# --- Plot 7: Distrib disparity ---
plt.figure(figsize=(8, 5))
for i in range(n_experiments):
    color = exp_colors[i % len(exp_colors)]
    
    r2_g0 = np.array(variance_diff_f_X[i][0])
    r2_g1 = np.array(variance_diff_f_X[i][1])
    diff = np.abs(r2_g0 - r2_g1)
    label = label_names[i]
    alpha_ = 0.3 if i <= 4 and i>1 else 0.8
    plt.plot(
        keep_percentages, 
        diff, 
        label=label,
        color=color,
        marker='v',
        linewidth=1,
        ls='dashed',
        alpha=alpha_
    )
plt.xlabel(r"$\mathbf{Rate\;of\;Samples\;to\;Keep:\;}$ $r = K/n$")
plt.ylabel(r"$\mathbf{Distribution\;Disparity\;(std):}$ $| std(\hat{y}|d_1) - std(\hat{y}|d_2) |$")
plt.title(r"$\mathbf{Impact\;of\;Correlation\;Fairness\;Constaint}$" "\n" r"$\mathbf{in\;Stable\;Regression}\;[Testing\;Set]$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig("./temp/plots/delete_vs_8_std_f_X_val.pdf", dpi=1200)



exit()





keep_percentages = np.linspace(0.1, 0.9, 5)
model_results = [[] for i in range(4)]
for i in range(4):
    model_results[i].append([[] for i in range(len(sensitive_idx))])
print(model_results)
# model_results = np.array
for keep_perc in keep_percentages:

    # Testing Stable v2: Fair version
    # model = LeastAbsoluteDeviationRegression(fit_intercept=True, fit_group_intercept=False, solver="MOSEK")
    model = StableRegression(fit_intercept=True, solver="MOSEK", k_percentage=0.7, lambda_l1=0, lambda_l2=0, objective="mae", 
                            sensitive_idx=sensitive_idx, fit_group_intercept=False, delta_l2=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print("R2 of the normal LAD: ", r2_score(y_train, y_pred))
    # print("R2 of the normal LAD (test): ", r2_score(y_test, model.predict(X_test)))
    for g, g_idx in enumerate(sensitive_idx):
        print(f"R2 of the normal LAD (g={g}): ", r2_score(y_train.iloc[g_idx], y_pred.iloc[g_idx]))
        model_results[0][g].append(r2_score(y_train.iloc[g_idx], y_pred.iloc[g_idx]))

    # model = LeastAbsoluteDeviationRegression(
    #     fit_intercept=True, fit_group_intercept=True, solver="MOSEK", sensitive_idx=sensitive_idx,
    #     l2_delta=1e-6
    #     )
    model = StableRegression(fit_intercept=True, solver="MOSEK", k_percentage=0.7, lambda_l1=1e-3, lambda_l2=1e-4, objective="mae", 
                            sensitive_idx=sensitive_idx, fit_group_intercept=True, delta_l2=1e-3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print("R2 of the normal LAD: ", r2_score(y_train, y_pred))
    # print("R2 of the normal LAD (test): ", r2_score(y_test, model.predict(X_test)))
    for g, g_idx in enumerate(sensitive_idx):
        print(f"R2 of the normal LAD (g={g}): ", r2_score(y_train.iloc[g_idx], y_pred.iloc[g_idx]))
        model_results[1][g].append(r2_score(y_train.iloc[g_idx], y_pred.iloc[g_idx]))


    model = StableRegression(fit_intercept=True, solver="MOSEK", k_percentage=0.7, lambda_l1=1e-3, lambda_l2=1e-4, objective="mae", 
                            sensitive_idx=sensitive_idx, fit_group_intercept=True, delta_l2=1e-3,
                            group_constraints=True, group_percentage_diff=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print("R2 of the normal LAD: ", r2_score(y_train, y_pred))
    # print("R2 of the normal LAD (test): ", r2_score(y_test, model.predict(X_test)))
    for g, g_idx in enumerate(sensitive_idx):
        print(f"R2 of the normal LAD (g={g}): ", r2_score(y_train.iloc[g_idx], y_pred.iloc[g_idx]))
        model_results[2][g].append(r2_score(y_train.iloc[g_idx], y_pred.iloc[g_idx]))

    model = StableRegression(fit_intercept=True, solver="MOSEK", k_percentage=0.7, lambda_l1=1e-3, lambda_l2=1e-4, objective="mae", 
                            sensitive_idx=sensitive_idx, fit_group_intercept=True, delta_l2=1e-3,
                            group_constraints=False, group_percentage_diff=1e-4,
                            weight_by_group=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print("R2 of the normal LAD: ", r2_score(y_train, y_pred))
    # print("R2 of the normal LAD (test): ", r2_score(y_test, model.predict(X_test)))
    for g, g_idx in enumerate(sensitive_idx):
        print(f"R2 of the normal LAD (g={g}): ", r2_score(y_train.iloc[g_idx], y_pred.iloc[g_idx]))
        model_results[3][g].append(r2_score(y_train.iloc[g_idx], y_pred.iloc[g_idx]))


plt.figure(figsize=(8,6))
for g in range(len(sensitive_idx)):
    for i in range(4):
        plt.plot(keep_percentages, model_results[i, g])
plt.savefig("./temp/plots/delete_vs_1.ong", dpi=600)
exit()



# Weighted MAE min
A_dim = len(sensitive_idx)
n_trials = 5
# random_weights = np.random.random((A_dim, n_trials))
t_w_ = np.linspace(0, 1, n_trials) # binary groups
random_weights = []
for t_ in range(n_trials):
    random_weights.append([t_w_[t_], 1-t_w_[t_]])
random_weights = np.array(random_weights).T
print(random_weights)
# print(random_weights)
n_train, m = X_train.shape
for t_ in range(n_trials):
    # print(t_)
    w_t = np.zeros(n_train)
    for g, g_idx in enumerate(sensitive_idx):
        # print(random_weights[g, t_])
        w_t[g_idx] += random_weights[g, t_] / len(g_idx)
    # w_t /= np.linalg.norm(random_weights[:,t_], 1)

    # print("sum of w: ", np.sum(w_t))
    # for g, g_idx in enumerate(sensitive_idx): 
    #     print("sum of w_g: ", np.sum(w_t[g_idx]))
    model = LeastAbsoluteDeviationRegression(fit_intercept=True, sensitive_weights=w_t, solver="MOSEK")
    models.append(model)

    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_train)
    # print("r2: ", r2_score(y_train, y_pred))

    

# Mixture of error and fairness
model = ConstantModel(sensitive_idx=sensitive_idx, objective="least_unfair", l1_lambda=l1, l2_lambda=l2)
model.fit(X_train, y_train)
y_fair = model.predict(X_train)
f_min, error_list = fairness_metric(y_train, y_fair, sensitive_idx=sensitive_idx, fairness_type="max_error", error_type="mae")
l_max = mean_absolute_error(y_train, y_fair)
model = ConstantModel(sensitive_idx=sensitive_idx, objective="least_error", l1_lambda=l1, l2_lambda=l2)
model.fit(X_train, y_train)
y_unfair = model.predict(X_train)
f_max, error_list = fairness_metric(y_train, y_unfair, sensitive_idx=sensitive_idx, fairness_type="max_error", error_type="mae")
l_min = mean_absolute_error(y_train, y_unfair)
baseline_values = {"f_max": f_max, "f_min": f_min, "l_max": l_max,"l_min": l_min}



for fair_weight in np.linspace(0, 1, 5): #11
    models.append( ConstantModel(sensitive_idx=sensitive_idx, objective="error_fairness_mixture", fair_weight=fair_weight, baseline_values=baseline_values, l1_lambda=l1, l2_lambda=l2) )




j = 0
for model in models:
    print("Training model=", model)
    model.fit(X_train, y_train)
    if j == 0:
        y_pred = model.predict(X_train)
        res = y_train - y_pred
        plt.figure(figsize=(8,6))
        plt.plot(y_train, res, 'o', label=model)#, markerfill=None)
        plt.savefig("./temp/plots/delete_vs.png")
    # exit()


# for f_w in np.linspace(0, 1, 5):
#     # Stacking of models
#     models.append( 
#         # ContextualFairStacking(
#         #     trained_models=models, 
#         #     sensitive_idx=sensitive_idx, 
#         #     # fair_weight=f_w_, 
#         #     l2_lambda=0,
#         #     sum_to_one=True,
#         #     nonneg = True,
#         #     fit_intercept= True,
#         #     solver= "MOSEK",#"OSQP",
#         #     # verbose: bool = False,
#         #     coupling=1e-3, #closeness to a general W
#         #     gate= "knn", #"rbf",
#         #     # rbf_gamma: Optional[float] = None,
#         #     knn_k = 7,
#         #     # gate_features: Optional[Sequence[int]] = None,
#         # ),
#         ModelStacking(
#             trained_models=models, 
#             sensitive_idx=sensitive_idx, 
#             fair_weight=f_w, 
#             l2_lambda=0,
#             sum_to_one=True,
#             nonneg = True,
#             fit_intercept= True,
#             solver= "MOSEK",#"OSQP",
#         )
#     )
#     models[-1].fit(X_train, y_train)




# Prediction phase
error_type = "mae"
train_loss_list = []
# train_fairness_list = []
colors = plotting_names()
shapes = plotting_shapes()

train_errors = {g:[] for g in range(len(sensitive_idx))}
# train_error_lists = {g:[] for g in range(len(sensitive_idx))}
train_variances = []


#  Compute utopian errors
utopian_errors = []
for g_idx in sensitive_idx:
    X_g, y_g = X_train.iloc[g_idx, :], y_train.iloc[g_idx]
    model = LeastAbsoluteDeviationRegression(fit_intercept=True, solver="MOSEK")
    # model = ConstantModel(sensitive_idx=sensitive_idx, objective="least_error", l1_lambda=l1, l2_lambda=l2)
    model.fit(X_g, y_g)
    y_pred = model.predict(X_train)
    utopian_error, error_list = fairness_metric(y_train, y_pred, sensitive_idx=sensitive_idx, fairness_type="min_error", error_type=error_type)
    utopian_errors.append(utopian_error)




# Compute error and fairness of each model
plt.figure(figsize=(8,6))
for i,model in enumerate(models):
    y_pred = model.predict(X_train)
    print("MSE", root_mean_squared_error(y_train, y_pred)**2)
    print("R2:", r2_score(y_train, y_pred))

    # Save losses
    if error_type == "mse":
        train_loss = root_mean_squared_error(y_train, y_pred)**2
    elif error_type == "mae":
        train_loss = mean_absolute_error(y_train, y_pred)
        print("MAE: ", train_loss)
    train_loss_list.append(train_loss)

    train_fairness, error_list = fairness_metric(y_train, y_pred, sensitive_idx=sensitive_idx, fairness_type="max_min_error", error_type=error_type)
    train_variances.append(np.std(y_pred - y_train))
    for g,error_ in enumerate(error_list):
        train_errors[g].append(error_)
    # train_fairness_list.append(train_fairness)
    print("F: ", train_fairness)
    print("F by group: ", error_list)

    print(shapes[i], model, colors[i])
    plt.plot(train_fairness, train_loss, shapes[i], label=model, color=colors[i], markeredgecolor='black')

plt.plot(max(utopian_errors)-min(utopian_errors), np.mean(utopian_errors), 'o', label="Utopian Solution (?)")
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel("(Un)Fairness")
plt.ylabel("Model Error")
plt.savefig("./temp/plots/delete.png", dpi=600, bbox_inches="tight")


# Variance plots
plt.figure(figsize=(8,6))
for i,model in enumerate(models):
    plt.plot(train_variances[i], train_loss_list[i], shapes[i], label=model, color=colors[i], markeredgecolor='black')
# plt.plot(np.std(utopian_errors), np.mean(utopian_errors), 'o', label="Utopian Solution (?)")
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.savefig("./temp/plots/delete_variances.png", dpi=600, bbox_inches="tight")

# Only Risk plots
plt.figure(figsize=(8,6))
for i,model in enumerate(models):
    plt.plot(train_errors[0][i], train_errors[1][i], shapes[i], label=model, color=colors[i], markeredgecolor='black')
x_ = np.linspace(utopian_errors[0], max(train_errors[0]), 2)
plt.plot(x_, x_, label="Equal Risk")
plt.plot(utopian_errors[0], utopian_errors[1], 'o', label="Utopian Solution")
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel(f"Error 0 ({len(sensitive_idx[0])})")
plt.ylabel(f"Error 1 ({len(sensitive_idx[1])})")
plt.savefig("./temp/plots/delete_errors.png", dpi=600, bbox_inches="tight")
print(train_errors)



plt.figure(figsize=(8,6))
for g, g_idx in enumerate(sensitive_idx):
    plt.plot(range(len(models)), train_errors[g], label=f"Error group {g}")
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel("Parameter")
plt.ylabel("Group Error")
plt.savefig("./temp/plots/delete_errors_path.png", dpi=600, bbox_inches="tight")


# # Testing Set
# plt.figure(figsize=(8,6))
# for i,model in enumerate(models):
#     y_pred = model.predict(X_val)
#     print("MSE", root_mean_squared_error(y_val, y_pred)**2)
#     print("R2:", r2_score(y_val, y_pred))

#     # Save losses
#     if error_type == "mse":
#         val_loss = root_mean_squared_error(y_val, y_pred)**2
#     elif error_type == "mae":
#         val_loss = mean_absolute_error(y_val, y_pred)
#         print("MAE: ", val_loss)
#     # val_loss_list.append(val_loss)

#     val_fairness = fairness_metric(y_val, y_pred, sensitive_idx=sensitive_idx_val, fairness_type="max_error", error_type=error_type)
#     # val_fairness_list.append(val_fairness)
#     print("F: ", val_fairness)

#     plt.plot(val_fairness, val_loss, shapes[i], label=model, color=colors[i], markeredgecolor='black')

# plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
# plt.xlabel("(Un)Fairness")
# plt.ylabel("Model Error")
# plt.savefig("./temp/plots/delete_val.png", dpi=1200, bbox_inches="tight")
