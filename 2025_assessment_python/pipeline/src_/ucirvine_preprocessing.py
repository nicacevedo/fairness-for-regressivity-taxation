import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np



import re

def get_uci_column_names(names_filepath):
    """
    Extracts column names from a standard UCI .names file.

    Args:
        names_filepath (str): The path to the .names file.

    Returns:
        list: A list of extracted column names.
    """
    with open(names_filepath, 'r') as f:
        # 1. Read all lines
        lines = f.readlines()

    # 2. Compile a regex to find a word at the start of the line, 
    #    followed by a colon or a space (the typical format for attribute names)
    #    and exclude lines that start with non-feature characters (like |, %)
    #    and skip blank lines.
    pattern = re.compile(r'^\s*([a-zA-Z0-9_\-]+)\s*[:\s]')

    column_names = []
    for line in lines:
        line = line.strip()
        # Skip comments, blank lines, and special metadata lines
        if not line or line.startswith('|') or line.startswith('%'):
            continue
        
        # Try to match the pattern
        match = pattern.search(line)
        if match:
            # Add the captured group (the name) to the list
            column_names.append(match.group(1))
            
    # 3. Handle the target column, which is often listed first, separated by a period.
    #    Example: 'income: >50K, <=50K.'
    if column_names and '.' in column_names[0]:
        # Split the first element by the period and take the first part 
        # (the name before the period, e.g., "income" from "income.")
        target_names = column_names[0].split('.')
        column_names = target_names + column_names[1:]

    return column_names



# --- 1. CONFIGURATION AND DATA SOURCE ---
# These are the direct URLs for the Adult data from the UCI repository
DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
# The .names file is NOT machine-readable, so we manually define the column names
# based on the file content. There are 14 features + 1 target (15 columns total).
COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'  # This is the target variable
]

# --- 2. DATA LOADING FUNCTION ---

def load_adult_data(data_url=DATA_URL, names=COLUMN_NAMES):
    """
    Loads the UCI Adult dataset directly into a pandas DataFrame.
    
    This function includes the necessary fixes (regex separator, engine='python')
    to handle the non-standard formatting of the UCI .data files, which 
    typically causes the pandas.errors.ParserError.
    
    Args:
        data_url (str): URL or local path to the .data file.
        names (list): List of column names to assign.
        
    Returns:
        pandas.DataFrame: The loaded and partially cleaned dataset.
    """
    print(f"Loading data from: {data_url}")
    try:
        df = pd.read_csv(
            data_url,
            header=None,
            names=names,
            # CRITICAL FIX for ParserError: use a regex to match comma + any space(s)
            sep=',\\s*', 
            engine='python',
            na_values='?' # Treat the common '?' symbol as a missing value (NaN)
        )
        # Drop rows with any missing values identified by '?'
        df = df.dropna()
        print(f"Data loaded successfully. Initial shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error during data loading: {e}")
        return pd.DataFrame()


# --- 3. PREPROCESSING FUNCTION ---

def preprocess_adult_data(df, target_name="income", pass_features=[]):
    """
    Performs the common, robust preprocessing steps for the Adult dataset.

    - Identifies categorical and numerical features.
    - Uses ColumnTransformer for simultaneous scaling (numerical) and 
      one-hot encoding (categorical).
    - Cleans and binarizes the target variable.
    
    Args:
        df (pandas.DataFrame): The raw, loaded Adult dataset.
        
    Returns:
        tuple: (X_processed, y_processed, preprocessor)
               - X_processed (np.ndarray): Processed feature matrix.
               - y_processed (np.ndarray): Binarized target vector.
               - preprocessor (ColumnTransformer): Fitted preprocessor object.
    """

    # 1. Separate features (X) and target (y)
    X = df.drop(target_name, axis=1)
    y = df[target_name].astype(str).str.strip() # Remove any leading/trailing whitespace from labels

    # 2. Binarize the target variable (income)
    # Target: >50K becomes 1, <=50K becomes 0
    
    print(y)
    if target_name in ["Rings", "G3"]:
        # print(y.head(10).to_numpy())
        y = y.astype(int)
        le = LabelEncoder()
        # print(y.head(10).to_numpy())
        y_processed = le.fit_transform(y.to_numpy())+1 #lb.fit_transform(y).ravel()
        # print(y_processed[:10])
    elif target_name == "income":
        lb = LabelBinarizer()
        y_processed = lb.fit_transform(y).ravel()
    # 3. Define feature types for ColumnTransformer
    # These are general types; specific columns may need adjustment
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    # print(pass_features)
    # print(categorical_features)
    for col in pass_features:
        if col in categorical_features:
            categorical_features.remove(col)
    # print(categorical_features)
    # Exclude 'fnlwgt' (Final Weight) from standard scaling/encoding
    # fnlwgt is often dropped or handled separately as it's a population-based weight
    # We will keep it but only scale the other numerical features for simplicity
    numerical_features_to_scale = [col for col in numerical_features if col != 'fnlwgt']
    
    # 4. Create the Preprocessing Pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop="first")

    # 0. Features to passtrough
    if "fnlwgt" in X.columns:
        pass_features += ['fnlwgt']
    for col in pass_features:
        try:
            dict_ = {x:i for i,x in enumerate(X[col].unique())}
            X[col] = X[col].map(dict_) #pd.get_dummies(X[col])#.astype(np.int64)
            X[col] = X[col].astype(np.int64)
            print("converted", col)
        except Exception as e:
            pass
    print("pass features:", pass_features)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_to_scale),
            ('cat', categorical_transformer, categorical_features),
            # Pass through the fnlwgt column without modification, or scale it if desired
            ('passthrough', 'passthrough', pass_features) 
        ],
        remainder='drop' # Drop any columns not specified above
    )

    # 5. Fit and transform the features
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Preprocessing complete.")
    print(f"Original features (raw): {X.shape}")
    print(f"Processed features (scaled & encoded): {X_processed.shape}")
    
    return X_processed, y_processed, preprocessor

# --- 4. MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    # 1. Load the data
    raw_df = load_adult_data()
    
    if not raw_df.empty:
        # 2. Preprocess the data
        X_final, y_final, preprocessor = preprocess_adult_data(raw_df)
        
        # 3. Example of using the processed data with a model (e.g., Logistic Regression)
        # First, split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set size (X, y): {X_train.shape}, {y_train.shape}")
        
        # 4. Train a simple model
        from sklearn.linear_model import LogisticRegression
        
        # Use a higher max_iter due to the scaled feature set size
        model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # 5. Evaluate the model
        accuracy = model.score(X_test, y_test)
        print(f"\nModel training successful!")
        print(f"Logistic Regression Test Accuracy: {accuracy:.4f}")
        
        # Optional: Get the names of the processed columns
        processed_feature_names = preprocessor.get_feature_names_out()
        # print("\nProcessed Feature Names (First 10):", processed_feature_names[:10])