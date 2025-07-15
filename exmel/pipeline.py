from pathlib import Path
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
from typing import Literal, NamedTuple, Iterator
from tqdm import tqdm

from exmel.alignment import AlignConfig, align, Alignment
from exmel.score import duration_adjusted_weighted_sum_velocity
from exmel.io import PathLike, extract_original_events
from exmel.sequence import Melody, Performance
from exmel.eval import evaluate_melody, plot_alignment

class Song(NamedTuple):
    name: str
    melody: Melody
    performance: Performance
    performance_path: Path
    ground_truth: Melody | None
    baseline: Melody | None

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
    def __init__(self, path: PathLike):
        self.dataset_path = Path(path)
        self.song_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        if len(self.song_dirs) == 0:
            raise FileNotFoundError(f"No directories found in {self.dataset_path}")

    def __len__(self):
        return len(self.song_dirs)

    def __getitem__(self, index: int) -> Song:
        song_dir = self.song_dirs[index]
        song_name = song_dir.name
        
        # Required files
        melody_path = song_dir / f"{song_name}.m.mid"
        performance_path = song_dir / f"{song_name}.t.mid"
        
        if not melody_path.exists():
            raise FileNotFoundError(f"Required melody file not found: {melody_path}")
        if not performance_path.exists():
            raise FileNotFoundError(f"Required performance file not found: {performance_path}")
        
        melody = Melody(melody_path)
        performance = Performance(performance_path)
        
        # Optional files - create objects only if files exist
        ground_truth_path = song_dir / f"{song_name}.gt.mid"
        ground_truth = Melody(ground_truth_path, track_idx=3) if ground_truth_path.exists() else None
        
        baseline_path = song_dir / f"{song_name}.bl.mid"
        baseline = Melody(baseline_path, track_idx=3) if baseline_path.exists() else None
        
        return Song(song_name, melody, performance, performance_path, ground_truth, baseline)

    def __iter__(self) -> Iterator[Song]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"Dataset(path={self.dataset_path.__repr__()})"

type DatasetLike = Dataset | PathLike

def inference_pipeline(
    config: AlignConfig,
    dataset: DatasetLike,
    save_dir: PathLike | None = None,
    verbose: Literal[0, 1, 2] = 1
) -> dict[str, Alignment]:
    """
    Inference pipeline for a dataset.
    """
    if isinstance(dataset, (Path, str)):
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
) -> pd.DataFrame:
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
    if isinstance(dataset, (Path, str)):
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
        alignment = align(song.melody, song.performance, config, verbose=(verbose>=2))
        end_time = time.time()
        runtime = end_time - start_time
        
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
    
    if baseline:
        baseline_stats = calculate_stats(baseline_results)
    
    # Print detailed statistics if verbose=3 (full detail)
    if verbose == 3:
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        print(f"\nTested on {len(sample_names)} samples")
        
        if baseline:
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
    
    return df


# Example usage (keeping the original test case)
if __name__ == "__main__":
    config = AlignConfig(
        score_func=duration_adjusted_weighted_sum_velocity,
        same_key=False,
        same_speed=False,
        speed_prior=1.0,
        variable_tail=True,
        local_tolerance=0.5,
        miss_tolerance=2,
        candidate_min_score=8,
        candidate_min_length=10,
        hop_length=2,
        split_melody=True
    )
    
    # Run the evaluation pipeline
    results_df = eval_pipeline(
        config=config,
        dataset="dataset",
        save_plot=True,
        save_csv=True,
        result_dir=None,  # Will create timestamped directory
        baseline=True,
        verbose=3
    )