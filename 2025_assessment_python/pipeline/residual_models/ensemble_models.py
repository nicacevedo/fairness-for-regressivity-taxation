import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

class LGBMRegressorWithResiduals:
    """
    A two-stage regressor using LGBMRegressor to first predict the target and
    then to correct the residuals from the first stage.
    """
    def __init__(self, random_state=None, residual_model_split=0.3, lgbm_params=None):
        """
        Initializes the two-stage model.
        Args:
            random_state (int): A seed for random operations to ensure reproducibility.
            residual_model_split (float): The fraction of the training data to hold
                                          out for training the residual model.
            lgbm_params (dict): A dictionary of parameters to pass to the LGBMRegressor.
                                These parameters will be used for both the base and
                                residual models. The residual model's objective will be
                                explicitly set to 'huber'.
        """
        self.random_state = random_state
        self.residual_model_split = residual_model_split
        self.lgbm_params = lgbm_params if lgbm_params is not None else {}
        self.base_model = None
        self.residual_model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits the two-stage model.

        The workflow is as follows:
        1.  Split the training data into a base set and a held-out residual set.
        2.  Train the base model on the base training set.
        3.  Predict on the held-out residual set to compute the residuals.
        4.  Train the residual model on the held-out set, with the residuals as the new target.

        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training target values.
            X_val (pd.DataFrame, optional): The validation features for the base model.
            y_val (pd.Series, optional): The validation target for the base model.
        """
        # --- Stage 1: Train the Base Model ---
        print("Training Stage 1: The Base Model")

        # Split the provided training data for the base and residual models
        X_base_train, X_residual_train, y_base_train, y_residual_train = train_test_split(
            X_train, y_train, test_size=self.residual_model_split, random_state=self.random_state
        )

        # Initialize the base LGBMRegressor with the user-provided parameters
        self.base_model = lgb.LGBMRegressor(**self.lgbm_params)

        # Use the provided validation set if available
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['eval_metric'] = "rmse"
            fit_params['callbacks'] = [lgb.early_stopping(stopping_rounds=50)]
        
        self.base_model.fit(X_base_train, y_base_train, **fit_params)

        # --- Stage 2: Train the Residual Model ---
        print("\nTraining Stage 2: The Residual Model")

        # Get base model predictions on the held-out set
        base_predictions = self.base_model.predict(X_residual_train)

        # Compute the residuals, which will be the new target for the second model
        residuals = y_residual_train.values - base_predictions
        
        # Split the residual data into a training and validation set
        X_res_train_fit, X_res_val, y_res_train_fit, y_res_val = train_test_split(
            X_residual_train, residuals, test_size=0.3, random_state=self.random_state
        )

        # Initialize the residual LGBMRegressor with the same parameters
        # but with the objective explicitly set to 'huber'
        self.lgbm_params['objective'] = 'huber'
        self.residual_model = lgb.LGBMRegressor(
            **self.lgbm_params
        )

        # Train the residual model on the held-out data with the residuals as the target
        self.residual_model.fit(
            X_res_train_fit,
            y_res_train_fit,
            eval_set=[(X_res_val, y_res_val)],
            eval_metric="huber",
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

    def predict(self, X):
        """
        Combines predictions from the base model and the residual model.

        Args:
            X (pd.DataFrame): The feature data to make predictions on.

        Returns:
            np.ndarray: The final combined predictions.
        """
        if self.base_model is None or self.residual_model is None:
            raise RuntimeError("You must call fit() before calling predict().")
        
        # Get predictions from the base model
        base_predictions = self.base_model.predict(X)
        
        # Get predictions from the residual model
        residual_predictions = self.residual_model.predict(X)
        
        # Combine the predictions
        final_predictions = base_predictions + residual_predictions
        
        return final_predictions
