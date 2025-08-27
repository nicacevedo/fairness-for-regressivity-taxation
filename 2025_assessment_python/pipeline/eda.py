import pandas as pd
import numpy as np
import yaml

# Load the data
np.random.seed(123)
X = pd.read_parquet(f"input/training_data.parquet").sample(10000)

# Load YAML params file
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)


desired_columns = params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date'] + ["ind_pin_is_multicard", "sv_is_outlier"]
# print(X.columns)
for i,col in enumerate(X.columns):
    if col in desired_columns:
        print(f"Column info of the {i}-th: ", col)
        print("type: ", X[col].dtype)
        if X[col].unique().size > 10:
            print(np.random.choice(X[col].unique(), 10, replace=False))
        else:
            print(X[col].unique())
