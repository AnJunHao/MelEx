import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import warnings

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
        sheet = pd.read_excel(sheets_dir / f"{song.name}.xlsx")
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

def train_model(
    df: pd.DataFrame,
    output_path: PathLike,
    test_size: float = 0.2,
    n_estimators: int = 1000,
    max_depth: int = 10
) -> None:
    """
    Train a model to predict the target variable from the features.
    """

    print(f"Number of samples: {len(df)}")
    
    features = ['length', 'misses', 'error', 'velocity', 'duration',
       'note_mean', 'note_std', 'note_entropy', 'note_unique', 'note_change',
       'relative_velocity', 'relative_note_mean',
       'relative_duration_per_event', 'normed_note_entropy',
       'normed_note_change', 'normed_misses']

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("=== Model Performance ===")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Testing MSE: {test_mse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Testing MAE: {test_mae:.4f}")
    print(f"Training RMSE: {np.sqrt(train_mse):.4f}")
    print(f"Testing RMSE: {np.sqrt(test_mse):.4f}")

    model.save_model(output_path)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Actual vs Predicted (Training)
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, color='blue', s=1)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2) # type: ignore
    axes[0, 0].set_xlabel('Actual (tp - fp)')
    axes[0, 0].set_ylabel('Predicted (tp - fp)')
    axes[0, 0].set_title(f'Training: Actual vs Predicted (R² = {train_r2:.4f})')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Actual vs Predicted (Testing)
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, color='green', s=1)
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # type: ignore
    axes[0, 1].set_xlabel('Actual (tp - fp)')
    axes[0, 1].set_ylabel('Predicted (tp - fp)')
    axes[0, 1].set_title(f'Testing: Actual vs Predicted (R² = {test_r2:.4f})')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residuals (Training)
    train_residuals = y_train - y_train_pred
    axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.5, color='blue', s=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted (tp - fp)')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Training: Residuals vs Predicted')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Residuals (Testing)
    test_residuals = y_test - y_test_pred
    axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.5, color='green', s=1)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted (tp - fp)')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Testing: Residuals vs Predicted')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()

    xgb.plot_importance(model)
    plt.show()
    plt.close()