from pathlib import Path
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
from typing import Literal, Iterator, Iterable, overload, TypedDict
from tqdm.auto import tqdm
from dataclasses import dataclass

from melign.align.alignment import AlignConfig, align, Alignment
from melign.data.io import PathLike, extract_original_events
from melign.data.sequence import Melody, Performance
from melign.align.eval_and_vis import evaluate_melody, plot_alignment

@dataclass(frozen=True, slots=True)
class Song:
    name: str
    melody: Melody
    performance: Performance
    performance_path: Path
    ground_truth: Melody | None
    baseline: Melody | None
    audio_path: Path | None

class DatasetConfig(TypedDict):
    melody_ext: str | Iterable[str]
    performance_ext: str | Iterable[str]
    ground_truth_ext: str | Iterable[str]
    ground_truth_track_idx: int | None
    baseline_ext: str | Iterable[str]
    baseline_track_idx: int | None
    audio_ext: str | Iterable[str]

default_extensions: DatasetConfig = {
    "melody_ext": (".m.mid", ".note", ".est.note"),
    "performance_ext": ".t.mid",
    "ground_truth_ext": ".gt.mid",
    "ground_truth_track_idx": 3,
    "baseline_ext": ".bl.mid",
    "baseline_track_idx": 3,
    "audio_ext": (".mp3", ".opus"),
}

class Dataset:
    """
    A dataset of songs, organized in this structure:
    
    dataset/
    ├── song one/
    │   ├── song one.m.mid (required: reference melody)
    │   ├── song one.t.mid (required: piano transcription)
    │   ├── song one.gt.mid (optional: ground truth)
    │   └── song one.bl.mid (optional: baseline)
    ├── song two/
    │   └── ... (files for <song two>)
    └── ... (other song directories)
    """
    def __init__(
        self,
        source: 'DatasetLike',
        extensions: DatasetConfig = default_extensions,
    ):
        self.extensions = extensions
        if isinstance(source, Dataset):
            self.song_dirs = [d for d in source.song_dirs]
        elif isinstance(source, (Path, str)):
            self.song_dirs = [d for d in Path(source).iterdir() if d.is_dir()]
        elif isinstance(source, Iterable):
            self.song_dirs = [Path(d) for d in source]
        if len(self.song_dirs) == 0:
            raise FileNotFoundError(f"No directories found in {source}")

    def _get_file_path(self, song_dir: Path, song_name: str, ext: str | Iterable[str]) -> Path | None:
        if isinstance(ext, str):
            ext = [ext]
        for e in ext:
            if (song_dir / f"{song_name}{e}").exists():
                return song_dir / f"{song_name}{e}"
        return None

    def __len__(self):
        return len(self.song_dirs)

    @overload
    def __getitem__(self, query: int | str) -> Song: ...
    @overload
    def __getitem__(self, query: slice | Iterable[str]) -> 'Dataset': ...
    def __getitem__(self, query: int | str | slice | Iterable[str]) -> 'Song | Dataset':
        # Handle string queries by finding the song with that name
        if isinstance(query, str):
            for i, song_dir in enumerate(self.song_dirs):
                if song_dir.name == query:
                    return self[i]
            else:
                raise KeyError(f"Song '{query}' not found in dataset")
        if isinstance(query, slice):
            return Dataset(self.song_dirs[query])
        if isinstance(query, Iterable):
            return Dataset([dir_ for dir_ in self.song_dirs if dir_.name in query])

        # Handle integer queries (existing behavior)
        song_dir = self.song_dirs[query]
        song_name = song_dir.name
        
        # Required files
        melody_path = self._get_file_path(song_dir, song_name, self.extensions["melody_ext"])
        performance_path = self._get_file_path(song_dir, song_name, self.extensions["performance_ext"])
        
        if melody_path is None:
            raise FileNotFoundError(f"Required melody file not found: {melody_path}")
        if performance_path is None:
            raise FileNotFoundError(f"Required performance file not found: {performance_path}")
        
        melody = Melody(melody_path)
        performance = Performance(performance_path)
        
        # Optional files - create objects only if files exist
        ground_truth_path = self._get_file_path(song_dir, song_name, self.extensions["ground_truth_ext"])
        if ground_truth_path is not None:
            ground_truth = Melody(ground_truth_path, track_idx=self.extensions["ground_truth_track_idx"])
        else:
            ground_truth = None
        
        baseline_path = self._get_file_path(song_dir, song_name, self.extensions["baseline_ext"])
        if baseline_path is not None:
            baseline = Melody(baseline_path, track_idx=self.extensions["baseline_track_idx"])
        else:
            baseline = None
        
        audio_path = self._get_file_path(song_dir, song_name, self.extensions["audio_ext"])
        if audio_path is not None:
            audio_path = Path(audio_path)
        else:
            audio_path = None
        
        return Song(song_name, melody, performance, performance_path, ground_truth, baseline, audio_path)

    def __iter__(self) -> Iterator[Song]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"Dataset(source={self.song_dirs.__repr__()})"

    def is_full(self) -> bool:
        if not self.is_valid():
            return False
        for song_dir in self.song_dirs:
            song_name = song_dir.name
            if (
                self._get_file_path(song_dir, song_name, self.extensions["ground_truth_ext"]) is None or
                self._get_file_path(song_dir, song_name, self.extensions["baseline_ext"]) is None or
                self._get_file_path(song_dir, song_name, self.extensions["audio_ext"]) is None
            ):
                return False
        return True

    def is_valid(self) -> bool:
        for song_dir in self.song_dirs:
            song_name = song_dir.name
            if (
                self._get_file_path(song_dir, song_name, self.extensions["melody_ext"]) is None or
                self._get_file_path(song_dir, song_name, self.extensions["performance_ext"]) is None
            ):
                return False
        return True

type DatasetLike = Dataset | PathLike | Iterable[PathLike]

def inference_pipeline(
    config: AlignConfig,
    dataset: DatasetLike,
    save_dir: PathLike | None = None,
    verbose: Literal[0, 1, 2] = 1
) -> dict[str, Alignment]:
    """
    Inference pipeline for a dataset.
    """
    dataset = Dataset(dataset)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    result: dict[str, Alignment] = {}
    for song in tqdm(dataset, desc="Processing songs", disable=verbose!=1):
        result[song.name] = align(song.melody, song.performance, config, verbose=verbose==2)
        if save_dir is not None:
            extract_original_events(
                result[song.name].events,
                song.performance_path,
                save_dir / f"{song.name}.mid")
    
    return result

def eval_pipeline(
    config: AlignConfig,
    dataset: DatasetLike,
    result_dir: PathLike | None = None,
    save_plot: bool = False,
    save_params: bool = False,
    save_csv: bool = False,
    save_mid: bool = False,
    baseline: bool = True,
    verbose: Literal[0, 1, 2, 3] = 1
) -> tuple[pd.DataFrame, dict[str, Alignment]]:
    """
    Evaluate alignment pipeline on a dataset.
    
    Args:
        config: AlignConfig object with alignment parameters
        dataset: Dataset object or path to the dataset directory
        result_dir: Directory to save results (if None, creates timestamped directory)
        save_plot: Whether to create alignment plots
        save_params: Whether to save parameters
        save_csv: Whether to save results to CSV
        save_mid: Whether to save alignment as MIDI
        baseline: Whether to calculate baseline metrics
        verbose: Verbosity level
            0: no output,
            1: single progress bar for all songs, nothing else,
            2: per-song progress bar (from align function), no overall progress,
            3: full detail including per-song progress bar
        
    Returns:
        DataFrame with evaluation results for each sample
    """
    # Create dataset object
    dataset = Dataset(dataset)
    
    # Only create result directory if any save operation is needed
    needs_result_dir = save_plot or save_csv or save_mid or save_params
    
    if needs_result_dir:
        # Create result directory if not provided
        if result_dir is None:
            result_dir = Path("results") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            result_dir = Path(result_dir)
        
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the parameters
        if save_params:
            config_dict = config.__dict__.copy()
            # Convert function to string representation for JSON serialization
            config_dict['score_func'] = config_dict['score_func'].__name__
            with open(result_dir / 'params.json', 'w') as f:
                json.dump(config_dict, f, indent=4)
    
    # Store results for statistical analysis
    baseline_results = []
    alignment_results = []
    sample_names = []
    alignment_runtimes = []

    # Dictionary to store alignment results
    alignment_dict: dict[str, Alignment] = {}
    
    # Initialize progress bar for verbose=1 (single progress bar for all songs)
    overall_pbar = None
    if verbose == 1:
        overall_pbar = tqdm(total=len(dataset), desc="Processing songs")
    
    for i, song in enumerate(dataset):
        if verbose == 3:
            print(f"Aligning <{song.name}> ({i+1}/{len(dataset)})...")
        
        # Time the alignment operation
        start_time = time.time()
        # Convert verbose level to boolean for align function
        # verbose=2,3 show per-song progress bars, verbose=0,1 don't
        alignment = align(song.melody, song.performance, config, defer_score=True, verbose=(verbose>=2))
        end_time = time.time()
        runtime = end_time - start_time

        alignment_dict[song.name] = alignment
        
        # Evaluate alignment
        assert song.ground_truth is not None, f"Ground truth file for <{song.name}> not found"
        result_alignment = evaluate_melody(
            song.ground_truth, alignment.events, plot=False)
        
        # Save alignment result as MIDI
        if save_mid:
            assert isinstance(result_dir, Path)
            extract_original_events(
                alignment.events,
                song.performance_path,
                result_dir / f"{song.name}.mid")
        
        # Create alignment plot if requested
        if save_plot:
            assert isinstance(result_dir, Path)
            plot_alignment(alignment, song.melody, song.performance, 
                               song.ground_truth, result_dir / f"{song.name}_align.png")
        
        # Evaluate baseline if requested
        result_baseline = None
        if baseline:
            assert song.baseline is not None, f"Baseline file for <{song.name}> not found"
            result_baseline = evaluate_melody(song.ground_truth, song.baseline, plot=False)
            baseline_results.append(result_baseline)
        
        # Print results if verbose=3 (full detail)
        if verbose == 3:
            if baseline and result_baseline is not None:
                print(f"Baseline: {result_baseline.precision*100:.1f}% {result_baseline.recall*100:.1f}% {result_baseline.f1_score*100:.1f}%")
            print(f"Alignment: {result_alignment.precision*100:.1f}% {result_alignment.recall*100:.1f}% {result_alignment.f1_score*100:.1f}%")
            print(f"Runtime: {runtime:.3f}s")
        
        # Store results
        alignment_results.append(result_alignment)
        sample_names.append(song.name)
        alignment_runtimes.append(runtime)
        
        # Update overall progress bar for verbose=1
        if overall_pbar:
            overall_pbar.update(1)
    
    # Close overall progress bar
    if overall_pbar:
        overall_pbar.close()
    
    # Calculate aggregate statistics
    def calculate_stats(results):
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1_scores = [r.f1_score for r in results]
        
        return {
            'precision': {
                'mean': np.mean(precisions),
                'std': np.std(precisions),
                'min': np.min(precisions),
                'max': np.max(precisions)
            },
            'recall': {
                'mean': np.mean(recalls),
                'std': np.std(recalls),
                'min': np.min(recalls),
                'max': np.max(recalls)
            },
            'f1_score': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'min': np.min(f1_scores),
                'max': np.max(f1_scores)
            },
            'tp_total': sum(r.tp for r in results),
            'fp_total': sum(r.fp for r in results),
            'fn_total': sum(r.fn for r in results)
        }
    
    alignment_stats = calculate_stats(alignment_results)
    
    baseline_stats = None
    if baseline:
        baseline_stats = calculate_stats(baseline_results)
    
    # Print detailed statistics if verbose=3 (full detail)
    if verbose == 3:
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        print(f"\nTested on {len(sample_names)} samples")
        
        if baseline:
            assert baseline_stats is not None
            print("\nBASELINE RESULTS:")
            print("-" * 40)
            print(f"Precision: {baseline_stats['precision']['mean']:.3f} ± {baseline_stats['precision']['std']:.3f}")
            print(f"Recall:    {baseline_stats['recall']['mean']:.3f} ± {baseline_stats['recall']['std']:.3f}")
            print(f"F1 Score:  {baseline_stats['f1_score']['mean']:.3f} ± {baseline_stats['f1_score']['std']:.3f}")
            print(f"Total TP: {baseline_stats['tp_total']}, FP: {baseline_stats['fp_total']}, FN: {baseline_stats['fn_total']}")
        
        print("\nALIGNMENT RESULTS:")
        print("-" * 40)
        print(f"Precision: {alignment_stats['precision']['mean']:.3f} ± {alignment_stats['precision']['std']:.3f}")
        print(f"Recall:    {alignment_stats['recall']['mean']:.3f} ± {alignment_stats['recall']['std']:.3f}")
        print(f"F1 Score:  {alignment_stats['f1_score']['mean']:.3f} ± {alignment_stats['f1_score']['std']:.3f}")
        print(f"Total TP: {alignment_stats['tp_total']}, FP: {alignment_stats['fp_total']}, FN: {alignment_stats['fn_total']}")
        
        if baseline:
            assert baseline_stats is not None
            print("\nIMPROVEMENT:")
            print("-" * 40)
            precision_improvement = alignment_stats['precision']['mean'] - baseline_stats['precision']['mean']
            recall_improvement = alignment_stats['recall']['mean'] - baseline_stats['recall']['mean']
            f1_improvement = alignment_stats['f1_score']['mean'] - baseline_stats['f1_score']['mean']
            
            print(f"Precision: {precision_improvement:+.3f} ({precision_improvement/baseline_stats['precision']['mean']*100:+.1f}%)")
            print(f"Recall:    {recall_improvement:+.3f} ({recall_improvement/baseline_stats['recall']['mean']*100:+.1f}%)")
            print(f"F1 Score:  {f1_improvement:+.3f} ({f1_improvement/baseline_stats['f1_score']['mean']*100:+.1f}%)")
        
        print("\nALIGNMENT RUNTIME:")
        print("-" * 40)
        print(f"Average: {np.mean(alignment_runtimes):.3f}s")
        print(f"Std Dev: {np.std(alignment_runtimes):.3f}s")
        print(f"Min:     {np.min(alignment_runtimes):.3f}s")
        print(f"Max:     {np.max(alignment_runtimes):.3f}s")
        print(f"Total:   {np.sum(alignment_runtimes):.3f}s")
    
    # Create detailed per-sample results DataFrame
    results_data = []
    for i, (name, alignment, runtime) in enumerate(zip(sample_names, alignment_results, alignment_runtimes)):
        row_data = {
            'Sample': name,
            'Alignment_Precision': alignment.precision,
            'Alignment_Recall': alignment.recall,
            'Alignment_F1': alignment.f1_score,
            'Runtime_seconds': runtime
        }
        
        if baseline:
            baseline_result = baseline_results[i]
            row_data.update({
                'Baseline_Precision': baseline_result.precision,
                'Baseline_Recall': baseline_result.recall,
                'Baseline_F1': baseline_result.f1_score,
                'Precision_Improvement': alignment.precision - baseline_result.precision,
                'Recall_Improvement': alignment.recall - baseline_result.recall,
                'F1_Improvement': alignment.f1_score - baseline_result.f1_score,
            })
        
        results_data.append(row_data)
    
    # Add a final row with the aggregate statistics
    aggregate_row = {
        'Sample': 'Aggregate',
        'Alignment_Precision': alignment_stats['precision']['mean'],
        'Alignment_Recall': alignment_stats['recall']['mean'],
        'Alignment_F1': alignment_stats['f1_score']['mean'],
        'Runtime_seconds': np.mean(alignment_runtimes)
    }
    
    if baseline:
        assert baseline_stats is not None
        aggregate_row.update({
            'Baseline_Precision': baseline_stats['precision']['mean'],
            'Baseline_Recall': baseline_stats['recall']['mean'],
            'Baseline_F1': baseline_stats['f1_score']['mean'],
            'Precision_Improvement': alignment_stats['precision']['mean'] - baseline_stats['precision']['mean'],
            'Recall_Improvement': alignment_stats['recall']['mean'] - baseline_stats['recall']['mean'],
            'F1_Improvement': alignment_stats['f1_score']['mean'] - baseline_stats['f1_score']['mean'],
        })
    
    results_data.append(aggregate_row)
    
    df = pd.DataFrame(results_data)
    
    # Save results to CSV
    if save_csv:
        assert isinstance(result_dir, Path)
        df.to_csv(result_dir / 'results.csv', index=False)
        if verbose == 3:
            print(f"\nDetailed results saved to '{result_dir / 'results.csv'}'")
    
    return df, alignment_dict

