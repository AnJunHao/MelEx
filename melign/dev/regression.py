import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import warnings
from sklearn.linear_model import LinearRegression
from typing import overload, Any
import json

from melign.api.dataset import Dataset
from melign.data.io import PathLike
from melign.data.sequence import song_stats

def concat_dataframe(dataset: Dataset, sheets_dir: PathLike, verbose: bool = True) -> pd.DataFrame:
    """
    Concatenate all sheets in the given directory into a single DataFrame.
    """
    sheets = []
    sheets_dir = Path(sheets_dir)
    for song in tqdm(dataset, desc="Concatenating sheets", disable=not verbose):
        file = sheets_dir / f"{song.name}.xlsx"
        if not file.exists():
            warnings.warn(f"File {file} doesn't exist, skipping...")
            continue
        # https://pandas.pydata.org/docs/dev/whatsnew/v2.2.0.html#calamine-engine-for-read-excel
        # https://pypi.org/project/python-calamine/
        sheet = pd.read_excel(sheets_dir / f"{song.name}.xlsx", engine="calamine")
        sheet["name"] = song.name
        sheets.append(sheet)
    return pd.concat(sheets)

def prepare_features(df: pd.DataFrame, song_df: pd.DataFrame) -> pd.DataFrame:

    df["target"] = df["tp"] - df["fp"]

    # First, we need to merge with song_df to get song-level stats
    song_stats = song_df.set_index('name') if 'name' in song_df.columns else song_df
    df = df.merge(song_stats, left_on='name', right_index=True, how='left')

    # 1. Relative velocity (velocity / mean velocity of original song)
    df['relative_velocity'] = df['velocity'] / df['velocity_mean']

    # 2. Relative note mean (note_mean / mean note of original song)
    df['relative_note_mean'] = df['note_mean'] / df['note_mean_song']

    # 3. Relative duration per event (duration / length / duration_per_event)
    df['relative_duration_per_event'] = df['duration'] / df['length'] / df['duration_per_event']
    # 1 / df_processed['relative_duration_per_event'] for all < 1 values
    df['relative_duration_per_event'] = df['relative_duration_per_event'].where(
        df['relative_duration_per_event'] >= 1, 1 / df['relative_duration_per_event'])

    # 4. Normalized note entropy (note_entropy / log2(note_unique))
    df['normed_note_entropy'] = df['note_entropy'] / np.log2(df['note_unique'])
    # Set NaN to 0
    df['normed_note_entropy'] = df['normed_note_entropy'].fillna(0)

    # 5. Normalized note change (note_change / length)
    df['normed_note_change'] = df['note_change'] / df['length']

    # 6. Normalized misses (misses / length)
    df['normed_misses'] = df['misses'] / df['length']

    return df

def get_song_stats_df(dataset: Dataset) -> pd.DataFrame:
    """
    Get a DataFrame with song-level statistics.
    """
    song_stats_list: list[dict] = []
    for song in dataset:
        stat_dict: dict = song_stats(song.melody, song.performance)
        stat_dict["name"] = song.name
        song_stats_list.append(stat_dict)
    return pd.DataFrame(song_stats_list)

def get_regression_dataframe(
    dataset: Dataset,
    sheets_dir: PathLike
) -> pd.DataFrame:
    df = concat_dataframe(dataset, sheets_dir)
    song_stats_df = get_song_stats_df(dataset)
    df = prepare_features(df, song_stats_df)
    return df

def train_model[T: xgb.XGBRegressor | LinearRegression](
    df: pd.DataFrame,
    output_path: PathLike,
    model: T,
    test_size: float = 0.2,
) -> T:
    """
    Train a model to predict the target variable from the features.
    """
    print(f"Number of samples: {len(df)}")
    
    features = ['length', 'misses', 'error', 'velocity', 'duration',
       'note_mean', 'note_std', 'note_entropy', 'note_unique', 'note_change',
       'relative_velocity', 'relative_note_mean',
       'relative_duration_per_event', 'normed_note_entropy',
       'normed_note_change', 'normed_misses',
       'sum_shadow', 'sum_between', 'sum_above_between',
       'duration_per_event', 'note_mean_song', 'velocity_mean']

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) # type: ignore
    X_train: pd.DataFrame; X_test: pd.DataFrame; y_train: pd.Series; y_test: pd.Series

    # Train the model
    model.fit(X_train, y_train)

    # Save model based on type
    if isinstance(model, xgb.XGBRegressor):
        model.save_model(output_path)
    elif isinstance(model, LinearRegression):
        # Save linear regression model as JSON
        model_data = {
            "model_type": "LinearRegression",
            "intercept": float(model.intercept_),
            "feature_weights": {}
        }
        
        # Add feature weights
        for feature, weight in zip(features, model.coef_):
            model_data["feature_weights"][feature] = float(weight)
        
        # Save to JSON file
        output_path = Path(output_path)
        if output_path.suffix.lower() != '.json':
            output_path = output_path.with_suffix('.json')
        
        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2)

    # Get predictions for evaluation
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate on training data
    print("\n=== Training Set Performance ===")
    evaluate_model(model, X_train, y_train)
    
    # Evaluate on test data
    print("\n=== Test Set Performance ===")
    evaluate_model(model, X_test, y_test)

    return model


@overload
def evaluate_model(
    model: xgb.XGBRegressor | LinearRegression,
    test_df: pd.DataFrame) -> None: ...
@overload
def evaluate_model(
    model: xgb.XGBRegressor | LinearRegression,
    test_X: pd.DataFrame,
    test_y: pd.Series) -> None: ...
@overload
def evaluate_model(
    test_X: pd.DataFrame,
    test_y: pd.Series,
    pred_y: pd.Series) -> None: ...
def evaluate_model(*args: Any) -> None: # type: ignore
    """
    Evaluate model performance and create visualizations.
    """
    model = None
    
    if len(args) == 2:
        # Case 1: model and test_df
        model, test_df = args
        features = ['length', 'misses', 'error', 'velocity', 'duration',
           'note_mean', 'note_std', 'note_entropy', 'note_unique', 'note_change',
           'relative_velocity', 'relative_note_mean',
           'relative_duration_per_event', 'normed_note_entropy',
           'normed_note_change', 'normed_misses',
           'sum_shadow', 'sum_between', 'sum_above_between',
           'duration_per_event', 'note_mean_song', 'velocity_mean']
        test_X = test_df[features]
        test_y = test_df['target']
        pred_y = pd.Series(model.predict(test_X), index=test_y.index)
    elif len(args) == 3:
        if hasattr(args[0], 'predict'):
            # Case 2: model, test_X, test_y
            model, test_X, test_y = args
            pred_y = pd.Series(model.predict(test_X), index=test_y.index)
        else:
            # Case 3: test_X, test_y, pred_y
            test_X, test_y, pred_y = args
    else:
        raise ValueError("Invalid number of arguments")

    # Calculate metrics
    mse = mean_squared_error(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)
    mae = mean_absolute_error(test_y, pred_y)
    rmse = np.sqrt(mse)
    
    # Print metrics
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Create visualizations
    if model is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = [axes[0], axes[1], None]
    
    # 1. Actual vs Predicted
    axes[0].scatter(test_y, pred_y, alpha=0.5, s=1)
    axes[0].plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual (tp - fp)')
    axes[0].set_ylabel('Predicted (tp - fp)')
    axes[0].set_title(f'Actual vs Predicted (R² = {r2:.4f})')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residuals Plot
    residuals = test_y - pred_y
    axes[1].scatter(pred_y, residuals, alpha=0.5, s=1)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted (tp - fp)')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals vs Predicted')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Feature Importance / Coefficients (if model is provided)
    if model is not None and axes[2] is not None:
        if isinstance(model, xgb.XGBRegressor):
            # XGBoost feature importance
            xgb.plot_importance(model, ax=axes[2])
        elif isinstance(model, LinearRegression):
            # Linear Regression coefficients
            coefficients = model.coef_
            feature_names = test_X.columns
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients
            }).sort_values('coefficient', key=abs, ascending=True)
            
            colors = ['red' if c < 0 else 'blue' for c in coef_df['coefficient']]
            axes[2].barh(coef_df['feature'], coef_df['coefficient'], color=colors)
            axes[2].set_xlabel('Coefficient')
            axes[2].set_title('Linear Regression Coefficients')
            axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()