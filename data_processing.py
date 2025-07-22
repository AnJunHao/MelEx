import pandas as pd
import numpy as np

def prepare_features(df: pd.DataFrame, song_df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = df["tp"] - df["fp"]
    # Merge with song_df to get song-level statistics
    # Assuming song_df has columns for mean velocity and mean note
    # We'll need to identify the correct column names from the output above

    # 1. Relative velocity: velocity / mean_velocity_of_original_song
    # First, we need to merge with song_df to get song-level stats
    song_stats = song_df.set_index('name') if 'name' in song_df.columns else song_df

    # If song_df doesn't have a 'name' column, we might need to check the first column
    if 'name' not in song_df.columns:
        # Assuming the first column contains song names
        song_stats = song_df.set_index(song_df.columns[0])

    # Merge the dataframes
    df = df.merge(song_stats, left_on='name', right_index=True, how='left')

    # 1. Relative velocity (velocity / mean velocity of original song)
    if 'velocity_mean' in df.columns:
        df['relative_velocity'] = df['velocity'] / df['velocity_mean']
    else:
        print("Warning: Could not find mean velocity column in song_df")

    # 2. Relative note mean (note_mean / mean note of original song)
    if 'note_mean_song' in df.columns:
        df['relative_note_mean'] = df['note_mean'] / df['note_mean_song']
    else:
        print("Warning: Could not find mean note column in song_df")

    # 3. Relative duration per event (duration / length / duration_per_event)
    if 'duration_per_event' in df.columns:
        df['relative_duration_per_event'] = df['duration'] / df['length'] / df['duration_per_event']
        # 1 / df_processed['relative_duration_per_event'] for all < 1 values
        df['relative_duration_per_event'] = df['relative_duration_per_event'].where(
            df['relative_duration_per_event'] >= 1, 1 / df['relative_duration_per_event'])
    else:
        print("Warning: Could not find duration per event column in song_df")

    # 4. Normalized note entropy (note_entropy / log2(note_unique))
    df['normed_note_entropy'] = df['note_entropy'] / np.log2(df['note_unique'])
    # Set NaN to 0
    df['normed_note_entropy'] = df['normed_note_entropy'].fillna(0)

    # 5. Normalized note change (note_change / length)
    df['normed_note_change'] = df['note_change'] / df['length']

    # 6. Normalized misses (misses / length)
    df['normed_misses'] = df['misses'] / df['length']

    return df
