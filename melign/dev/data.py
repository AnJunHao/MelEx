import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import TypedDict
from random import sample
import numpy as np
from collections import Counter
import pandas as pd

from melign.align.alignment import align, AlignConfig, FrozenMatch
from melign.align.score import SimpleModel, tp_fp, MatchLike
from melign.api.dataset import Dataset, Song
from melign.data.io import PathLike

def _prepare_one_song_candidates(args: tuple[Song, int, int, int, Path]) -> int | None:
    """
    Process a single song for parallel candidate generation and save to file.
    Returns number of candidates (0 if skipped).
    """
    song, miss_tolerance, hop_length, min_length, output_dir = args

    if (output_dir / f"{song.name}.pkl").exists():
        return None
    
    score_model = SimpleModel()
    candidates = align(
        song.melody,
        song.performance,
        AlignConfig(
            score_model=score_model,
            miss_tolerance=miss_tolerance,
            hop_length=hop_length,
            candidate_min_length=min_length,
            candidate_min_score=0,
        ),
        skip_wisp=True,
        verbose=False  # No per-song progress bars in parallel mode
    )
    
    # Save immediately in parallel - each song saves to different file
    with open(output_dir / f"{song.name}.pkl", "wb") as f:
        pickle.dump(candidates, f)

    return len(candidates)

def prepare_candidates(
    dataset: Dataset,
    output_dir: PathLike,
    miss_tolerance: int,
    hop_length: int,
    min_length: int,
    verbose: bool = True,
    n_jobs: int = 1
) -> None:
    """
    Save alignment candidates for each song in the dataset.
    
    Args:
        dataset: Dataset object containing songs to process
        output_dir: Directory to save candidate pickle files
        miss_tolerance: Miss tolerance parameter for alignment
        hop_length: Hop length parameter for alignment
        min_length: Minimum length parameter for alignment
        verbose: Whether to show progress bar
        n_jobs: Number of parallel jobs to run (1 for sequential processing)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    candidate_counts: list[int | None] = []
    
    if n_jobs == 1:
        # Sequential processing (original behavior)
        progress_bar = tqdm(dataset, desc="Processing songs", disable=not verbose)
        
        score_model = SimpleModel()
        for song in progress_bar:
            candidates = align(
                song.melody,
                song.performance,
                AlignConfig(
                    score_model=score_model,
                    miss_tolerance=miss_tolerance,
                    hop_length=hop_length,
                    candidate_min_length=min_length,
                    candidate_min_score=0,
                ),
                skip_wisp=True,
                verbose=False  # Don't show per-song progress to avoid clutter
            )
            candidate_counts.append(len(candidates))
            # Save immediately in sequential mode
            with open(output_dir / f"{song.name}.pkl", "wb") as f:
                pickle.dump(candidates, f)
        
        progress_bar.close()
    
    else:
        # Parallel processing
        overall_pbar = None
        if verbose:
            overall_pbar = tqdm(total=len(dataset), desc=f"Processing songs (n_jobs={n_jobs})")
        
        # Prepare arguments for parallel processing
        args_list = [(song, miss_tolerance, hop_length, min_length, output_dir) for song in dataset]
        
        max_workers = n_jobs if n_jobs > 0 else None
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_song = {
                executor.submit(_prepare_one_song_candidates, args): args[0].name 
                for args in args_list
            }
            
            for future in as_completed(future_to_song):
                song_name = future_to_song[future]
                try:
                    num_candidates = future.result()
                    candidate_counts.append(num_candidates)
                    if overall_pbar:
                        overall_pbar.update(1)
                except Exception as exc:
                    print(f"Song <{song_name}> generated an exception: {exc}")
                    candidate_counts.append(0)
                    if overall_pbar:
                        overall_pbar.update(1)
        
        # Close overall progress bar
        if overall_pbar:
            overall_pbar.close()
    # Report statistics
    if candidate_counts:
        # There may be None values in candidate_counts
        num_none = candidate_counts.count(None)
        if num_none > 0:
            print(f"Skipped {num_none} songs because they already exist")
        candidate_counts_filtered: list[int] = [c for c in candidate_counts if c is not None]
        if candidate_counts_filtered:
            avg = np.mean(candidate_counts_filtered)
            min_ = np.min(candidate_counts_filtered)
            max_ = np.max(candidate_counts_filtered)
            print(f"Candidate statistics: avg={avg:.2f}, min={min_}, max={max_}")
    else:
        print("No candidates were generated.")

class Entry(TypedDict):
    name: str
    tp: int
    fp: int
    length: int
    misses: int
    error: float
    velocity: float
    duration: float
    note_mean: float
    note_std: float
    note_entropy: float
    note_unique: int
    note_change: int

class ScoreFunctions:

    @staticmethod
    def length(match: MatchLike) -> int:
        return len(match.events)

    @staticmethod
    def misses(match: MatchLike) -> int:
        return match.sum_miss

    @staticmethod
    def error(match: MatchLike) -> float:
        return match.sum_error

    @staticmethod
    def velocity(match: MatchLike) -> float:
        return float(np.mean([event.velocity for event in match.events]))
    
    @staticmethod
    def duration(match: MatchLike) -> float:
        return match.events[-1].time - match.events[0].time

    @staticmethod
    def note_mean(match: MatchLike) -> float:
        return sum(event.note for event in match.events) / len(match.events)

    @staticmethod
    def note_std(match: MatchLike) -> float:
        return float(np.std([event.note for event in match.events]))

    @staticmethod
    def note_entropy(match: MatchLike) -> float:
        counts = np.array(list(Counter(event.note for event in match.events).values()))
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    @staticmethod
    def note_unique(match: MatchLike) -> int:
        return len(set(event.note for event in match.events))

    @staticmethod
    def note_change(match: MatchLike) -> int:
        return sum(match.events[i].note != match.events[i - 1].note
                   for i in range(1, len(match.events)))

def _score_one_song_candidates(args: tuple[Song, Path, Path, int | None, int | None]) -> None:
    """
    Score a single song's candidates and save to file.
    """
    song, candidates_dir, output_dir, min_length, n_sample = args

    if (output_dir / f"{song.name}.xlsx").exists():
        print(f"Skipping {song.name} because {output_dir / f'{song.name}.xlsx'} already exists")
        return

    candidates: list[FrozenMatch] = pickle.load(open(candidates_dir / f"{song.name}.pkl", "rb"))

    if min_length is not None:
        candidates = [c for c in candidates if len(c.events) >= min_length]

    if n_sample is not None:
        n_samples = min(n_sample, len(candidates))
        candidates = sample(candidates, n_samples)

    data: list[Entry] = []

    for c in candidates:
        assert song.ground_truth is not None
        tp, fp = tp_fp(c, song.ground_truth)
        data.append(Entry(
            name=song.name,
            tp=tp,
            fp=fp,
            length=ScoreFunctions.length(c),
            misses=ScoreFunctions.misses(c),
            error=ScoreFunctions.error(c),
            velocity=ScoreFunctions.velocity(c),
            duration=ScoreFunctions.duration(c),
            note_mean=ScoreFunctions.note_mean(c),
            note_std=ScoreFunctions.note_std(c),
            note_entropy=ScoreFunctions.note_entropy(c),
            note_unique=ScoreFunctions.note_unique(c),
            note_change=ScoreFunctions.note_change(c)))

    pd.DataFrame(data).to_excel(output_dir / f"{song.name}.xlsx")

def score_candidates(
    dataset: Dataset,
    candidates_dir: PathLike,
    output_dir: PathLike,
    min_length: int | None = None,
    n_sample: int | None = None,
    verbose: bool = True,
    n_jobs: int = 1
) -> None:
    """
    Load alignment candidates from a directory and compute scores.
    
    Args:
        dataset: Dataset object containing songs to process
        candidates_dir: Directory containing candidate pickle files
        output_dir: Directory to save score Excel files
        n_sample: Number of candidates to sample per song (None = use all)
        verbose: Whether to show progress bar
        n_jobs: Number of parallel jobs to run (1 for sequential processing)
    """
    candidates_dir = Path(candidates_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if n_jobs == 1:
        # Sequential processing
        progress_bar = tqdm(dataset, desc="Scoring candidates", disable=not verbose)
        
        for song in progress_bar:
            _score_one_song_candidates((song, candidates_dir, output_dir, min_length, n_sample))
        
        progress_bar.close()
    
    else:
        # Parallel processing
        overall_pbar = None
        if verbose:
            overall_pbar = tqdm(total=len(dataset), desc=f"Scoring candidates (n_jobs={n_jobs})")
        
        # Prepare arguments for parallel processing
        args_list = [(song, candidates_dir, output_dir, min_length, n_sample) for song in dataset]
        
        max_workers = n_jobs if n_jobs > 0 else None
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_song = {
                executor.submit(_score_one_song_candidates, args): args[0].name 
                for args in args_list
            }
            
            # Process completed jobs as they finish
            for future in as_completed(future_to_song):
                song_name = future_to_song[future]
                
                try:
                    future.result()  # Wait for completion
                    # Update overall progress bar
                    if overall_pbar:
                        overall_pbar.update(1)
                        
                except Exception as exc:
                    print(f"Song <{song_name}> generated an exception: {exc}")
                    if overall_pbar:
                        overall_pbar.update(1)
        
        # Close overall progress bar
        if overall_pbar:
            overall_pbar.close()