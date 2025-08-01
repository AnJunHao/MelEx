from pathlib import Path
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
from typing import Literal
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from os import process_cpu_count

from melex.align.alignment import AlignConfig, align, Alignment, self_eval
from melex.data.sequence import Melody
from melex.data.io import PathLike, extract_original_events
from melex.align.eval_and_vis import evaluate_melody, plot_alignment
from melex.api.dataset import Dataset, DatasetLike, Song
from melex.api.preset import get_default_config
from melex.api.pretty_xlsx import format_dataframe_to_excel

def _evaluate_single_song(args: tuple[Song, AlignConfig, bool, bool, bool, bool, PathLike | None]):
    """
    Process a single song for parallel execution.
    Returns all the data needed for the main thread to handle I/O and aggregation.
    """
    # Unpack arguments
    (
        song, config, baseline, verbose_per_song,
        save_mid, save_plot, result_dir
    ) = args
    
    # Time the alignment operation
    start_time = time.time()
    alignment = align(song.melody, song.performance, config, defer_score=True, verbose=verbose_per_song)
    end_time = time.time()
    runtime = end_time - start_time
    
    # Evaluate alignment
    assert song.ground_truth is not None, f"Ground truth file for <{song.name}> not found"
    result_alignment = evaluate_melody(song.ground_truth, alignment.events, plot=False)
    
    # Evaluate baseline if requested
    result_baseline = None
    if baseline:
        assert song.baseline is not None, f"Baseline file for <{song.name}> not found"
        result_baseline = evaluate_melody(song.ground_truth, song.baseline, plot=False)
    
    # Handle file I/O operations (parallel-safe, unique filenames)
    if result_dir is not None:
        if save_mid:
            extract_original_events(
                alignment.events,
                song.performance_path,
                Path(result_dir) / f"{song.name}.mid")
        if save_plot:
            plot_alignment(alignment, song.melody, song.performance, 
                           song.ground_truth, Path(result_dir) / f"{song.name}_align.png")
    
    return {
        'song': song,
        'alignment': alignment,
        'result_alignment': result_alignment,
        'result_baseline': result_baseline,
        'runtime': runtime
    }


def _inference_single_song(args: tuple[Song, AlignConfig, bool, bool, bool, PathLike | None]):
    """
    Process a single song for parallel inference execution.
    Returns alignment data needed for the main thread to handle I/O and aggregation.
    """
    # Unpack arguments
    (
        song, config, verbose_per_song, save_midi, save_plot, save_dir
    ) = args
    
    # Perform alignment
    alignment = align(song.melody, song.performance, config, verbose=verbose_per_song)
    
    # Handle file I/O operations (parallel-safe, unique filenames)  
    if save_dir is not None:
        if save_midi:
            extract_original_events(
                alignment.events,
                song.performance_path,
                Path(save_dir) / f"{song.name}.mid")
        if save_plot:
            plot_alignment(alignment, song.melody, song.performance, 
                           save_path=Path(save_dir) / f"{song.name}_align.png")
    
    return {
        'song': song,
        'alignment': alignment
    }

def inference_pipeline(
    dataset: DatasetLike,
    config: AlignConfig = get_default_config(),
    save_midi: bool = True,
    save_excel: bool = True,
    save_plot: bool = True,
    save_dir: PathLike | None = None,
    verbose: Literal[0, 1, 2] = 1,
    n_jobs: int = 0
) -> tuple[dict[str, Melody], pd.DataFrame]:
    """
    Inference pipeline for a dataset.
    
    Args:
        dataset: Dataset object or path to the dataset directory
        config: AlignConfig object with alignment parameters
        save_midi: Whether to save alignment as MIDI files
        save_excel: Whether to save evaluation results to Excel
        save_plot: Whether to create alignment plots
        save_dir: Directory to save results (if None, creates timestamped directory)
        verbose: Verbosity level
            0: no output,
            1: single progress bar for all songs,
            2: per-song progress bar (from align function)
        n_jobs: Number of parallel jobs to run (1 for sequential processing, 0 for auto-detect)
        
    Returns:
        Tuple of (melody dictionary, evaluation DataFrame)
    """
    if n_jobs > 1 and verbose >= 2:
        warnings.warn("Parallel processing is not supported with verbose >= 2 (per task progress bar), setting verbose to 1")
        verbose = 1

    if not isinstance(dataset, Dataset):
        dataset = Dataset(dataset)
    
    assert dataset.is_valid(), "Dataset is not valid"

    if save_midi or save_plot or save_excel:
        if save_dir is None:
            save_dir = Path("outputs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if verbose >= 1:
                print(f"No save directory provided, using default: {save_dir}")
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    result: dict[str, Alignment] = {}
    
    # Convert verbose level to boolean for per-song progress (verbose=2 shows per-song progress bars)
    verbose_per_song = (verbose >= 2)
    
    if n_jobs == 1:
        # Sequential processing (original behavior)
        for song in tqdm(dataset, desc="Processing songs", disable=verbose!=1):
            result[song.name] = align(song.melody, song.performance, config, verbose=verbose_per_song)
            if save_midi:
                assert isinstance(save_dir, Path)
                extract_original_events(
                    result[song.name].events,
                    song.performance_path,
                    save_dir / f"{song.name}.mid",
                    no_overlap=True)
            if save_plot:
                assert isinstance(save_dir, Path)
                plot_alignment(result[song.name], song.melody, song.performance, 
                               save_path=save_dir / f"{song.name}_align.png")
    else:
        # Parallel processing
        # Initialize progress bar for verbose=1 (single progress bar for all songs)
        overall_pbar = None
        if verbose == 1:
            overall_pbar = tqdm(total=len(dataset), desc=f"Processing songs (n_jobs={n_jobs if n_jobs > 0 else process_cpu_count()})")
        
        # Prepare arguments for parallel processing
        args_list = [
            (song, config, verbose_per_song, save_midi, save_plot, save_dir)
            for song in dataset
        ]
        
        max_workers = n_jobs if n_jobs > 0 else None
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_idx = {
                executor.submit(_inference_single_song, args): i 
                for i, args in enumerate(args_list)
            }
            
            # Process completed jobs as they finish
            results_dict = {}  # To store results by original index
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                song = args_list[idx][0]  # Get the song from args
                
                try:
                    process_result = future.result()
                    results_dict[idx] = process_result
                    
                    # Update overall progress bar for verbose=1
                    if overall_pbar:
                        overall_pbar.update(1)
                        
                except Exception as exc:
                    print(f"Song <{song.name}> generated an exception: {exc}")
                    if overall_pbar:
                        overall_pbar.update(1)
        
        # Close overall progress bar
        if overall_pbar:
            overall_pbar.close()
        
        # Process results in original order
        for i in range(len(dataset)):
            if i not in results_dict:
                continue  # Skip failed jobs
                
            process_result = results_dict[i]
            song = process_result['song']
            
            # Store results in order
            result[song.name] = process_result['alignment']

    evals = []
    for name, alignment in result.items():
        entry: dict = {'name': name}
        entry.update(alignment.metadata)
        del entry["discarded_matches"]
        entry.update(self_eval(alignment))
        evals.append(entry)

    df = pd.DataFrame(evals)

    if save_excel:
        assert isinstance(save_dir, Path)
        format_dataframe_to_excel(df, save_dir / "evaluation.xlsx")

    return {name: Melody(alignment) for name, alignment in result.items()}, df

def eval_pipeline(
    dataset: DatasetLike,
    config: AlignConfig = get_default_config(),
    result_dir: PathLike | None = None,
    save_plot: bool = False,
    save_params: bool = False,
    save_csv: bool = False,
    save_midi: bool = False,
    baseline: bool = True,
    verbose: Literal[0, 1, 2, 3] = 1,
    n_jobs: int = 0
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
        n_jobs: Number of parallel jobs to run (1 for sequential processing, 0 for auto-detect)
        
    Returns:
        DataFrame with evaluation results for each sample
    """
    if n_jobs > 1 and verbose >= 2:
        warnings.warn("Parallel processing is not supported with verbose >= 2 (per task progress bar), setting verbose to 1")
        verbose = 1

    dataset = Dataset(dataset)
    
    # Only create result directory if any save operation is needed
    needs_result_dir = save_plot or save_csv or save_midi or save_params
    
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
    
    # Convert verbose level to boolean for per-song progress (verbose=2,3 show per-song progress bars)
    verbose_per_song = (verbose >= 2)
    
    if n_jobs == 1:
        # Sequential processing (original behavior)
        # Initialize progress bar for verbose=1 (single progress bar for all songs)
        overall_pbar = None
        if verbose == 1:
            overall_pbar = tqdm(total=len(dataset), desc=f"Processing songs (n_jobs={n_jobs})")
        
        for i, song in enumerate(dataset):
            if verbose == 3:
                print(f"Aligning <{song.name}> ({i+1}/{len(dataset)})...")
            
            # Process song
            result = _evaluate_single_song((song, config, baseline, verbose_per_song, save_midi, save_plot, result_dir))
            
            # Store results
            alignment_dict[song.name] = result['alignment']
            alignment_results.append(result['result_alignment'])
            if baseline and result['result_baseline'] is not None:
                baseline_results.append(result['result_baseline'])
            sample_names.append(song.name)
            alignment_runtimes.append(result['runtime'])
            
            # Print results if verbose=3 (full detail)
            if verbose == 3:
                if baseline and result['result_baseline'] is not None:
                    baseline_res = result['result_baseline']
                    print(f"Baseline: {baseline_res.precision*100:.1f}% {baseline_res.recall*100:.1f}% {baseline_res.f1_score*100:.1f}%")
                alignment_res = result['result_alignment']
                print(f"Alignment: {alignment_res.precision*100:.1f}% {alignment_res.recall*100:.1f}% {alignment_res.f1_score*100:.1f}%")
                print(f"Runtime: {result['runtime']:.3f}s")
            
            # Update overall progress bar for verbose=1
            if overall_pbar:
                overall_pbar.update(1)
        
        # Close overall progress bar
        if overall_pbar:
            overall_pbar.close()
    
    else:
        # Parallel processing
        # Initialize progress bar for verbose=1 (single progress bar for all songs)
        overall_pbar = None
        if verbose == 1:
            overall_pbar = tqdm(total=len(dataset), desc=f"Processing songs (n_jobs={n_jobs})")
        
        # Prepare arguments for parallel processing
        args_list = [
            (song, config, baseline, verbose_per_song, save_midi, save_plot, result_dir)
            for song in dataset
        ]
        
        max_workers = n_jobs if n_jobs > 0 else None
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_idx = {
                executor.submit(_evaluate_single_song, args): i 
                for i, args in enumerate(args_list)
            }
            
            # Process completed jobs as they finish
            results_dict = {}  # To store results by original index
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                song = args_list[idx][0]  # Get the song from args
                
                try:
                    result = future.result()
                    results_dict[idx] = result
                    
                    if verbose == 3:
                        print(f"Completed <{song.name}> ({idx+1}/{len(dataset)})...")
                        if baseline and result['result_baseline'] is not None:
                            baseline_res = result['result_baseline']
                            print(f"Baseline: {baseline_res.precision*100:.1f}% {baseline_res.recall*100:.1f}% {baseline_res.f1_score*100:.1f}%")
                        alignment_res = result['result_alignment']
                        print(f"Alignment: {alignment_res.precision*100:.1f}% {alignment_res.recall*100:.1f}% {alignment_res.f1_score*100:.1f}%")
                        print(f"Runtime: {result['runtime']:.3f}s")
                    
                    # Update overall progress bar for verbose=1
                    if overall_pbar:
                        overall_pbar.update(1)
                        
                except Exception as exc:
                    print(f"Song <{song.name}> generated an exception: {exc}")
                    if overall_pbar:
                        overall_pbar.update(1)
        
        # Close overall progress bar
        if overall_pbar:
            overall_pbar.close()
        
        # Process results in original order and handle I/O operations sequentially
        for i in range(len(dataset)):
            if i not in results_dict:
                continue  # Skip failed jobs
                
            result = results_dict[i]
            song = args_list[i][0]
            
            # Store results in order
            alignment_dict[song.name] = result['alignment']
            alignment_results.append(result['result_alignment'])
            if baseline and result['result_baseline'] is not None:
                baseline_results.append(result['result_baseline'])
            sample_names.append(song.name)
            alignment_runtimes.append(result['runtime'])

    # Calculate aggregate statistics
    def calculate_stats(results):
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1_scores = [r.f1_score for r in results]
        
        return {
            'precision': {
                'mean': np.mean(precisions),
                'median': np.median(precisions),
                'std': np.std(precisions),
                'min': np.min(precisions),
                'max': np.max(precisions)
            },
            'recall': {
                'mean': np.mean(recalls),
                'median': np.median(recalls),
                'std': np.std(recalls),
                'min': np.min(recalls),
                'max': np.max(recalls)
            },
            'f1_score': {
                'mean': np.mean(f1_scores),
                'median': np.median(f1_scores),
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
    
    # Add aggregate rows with mean and median statistics
    mean_aggregate_row = {
        'Sample': 'Mean',
        'Alignment_Precision': alignment_stats['precision']['mean'],
        'Alignment_Recall': alignment_stats['recall']['mean'],
        'Alignment_F1': alignment_stats['f1_score']['mean'],
        'Runtime_seconds': np.mean(alignment_runtimes)
    }
    
    if baseline:
        assert baseline_stats is not None
        mean_aggregate_row.update({
            'Baseline_Precision': baseline_stats['precision']['mean'],
            'Baseline_Recall': baseline_stats['recall']['mean'],
            'Baseline_F1': baseline_stats['f1_score']['mean'],
            'Precision_Improvement': alignment_stats['precision']['mean'] - baseline_stats['precision']['mean'],
            'Recall_Improvement': alignment_stats['recall']['mean'] - baseline_stats['recall']['mean'],
            'F1_Improvement': alignment_stats['f1_score']['mean'] - baseline_stats['f1_score']['mean'],
        })
    
    median_aggregate_row = {
        'Sample': 'Median',
        'Alignment_Precision': alignment_stats['precision']['median'],
        'Alignment_Recall': alignment_stats['recall']['median'],
        'Alignment_F1': alignment_stats['f1_score']['median'],
        'Runtime_seconds': np.median(alignment_runtimes)
    }
    
    if baseline:
        assert baseline_stats is not None
        median_aggregate_row.update({
            'Baseline_Precision': baseline_stats['precision']['median'],
            'Baseline_Recall': baseline_stats['recall']['median'],
            'Baseline_F1': baseline_stats['f1_score']['median'],
            'Precision_Improvement': alignment_stats['precision']['median'] - baseline_stats['precision']['median'],
            'Recall_Improvement': alignment_stats['recall']['median'] - baseline_stats['recall']['median'],
            'F1_Improvement': alignment_stats['f1_score']['median'] - baseline_stats['f1_score']['median'],
        })
    
    results_data.append(mean_aggregate_row)
    results_data.append(median_aggregate_row)
    
    df = pd.DataFrame(results_data)
    
    # Save results to CSV
    if save_csv:
        assert isinstance(result_dir, Path)
        df.to_csv(result_dir / 'results.csv', index=False)
        if verbose == 3:
            print(f"\nDetailed results saved to '{result_dir / 'results.csv'}'")
    
    return df, alignment_dict

