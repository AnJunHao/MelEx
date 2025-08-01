import pickle
from typing import Iterable, TypedDict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from collections import defaultdict

from melex.align.alignment import FrozenMatch, concat_matches, filter_within
from melex.align.score import ScoreModel, StructuralMapping
from melex.align.dp import weighted_interval_scheduling, non_crossing_weighted_bi_interval_scheduling
from melex.align.eval_and_vis import evaluate_melody
from melex.api.dataset import Dataset, Song
from melex.data.io import PathLike
from melex.data.sequence import song_stats

DURATION_TOLERANCE = 0.5
MELODY_MIN_RECURRENCE = 0.97

class Result(TypedDict):
    min_score: float
    f1: float
    precision: float
    recall: float

def _min_score_one_song(
    args: tuple[Song, Path, int | None, Iterable[float], Iterable[ScoreModel]]
    ) -> dict[int, list[Result]]:
    song, candidates_dir, min_length, search_range, score_models = args

    candidates: list[FrozenMatch] = pickle.load(open(candidates_dir / f"{song.name}.pkl", "rb"))

    if min_length is not None:
        candidates = [c for c in candidates if len(c.events) >= min_length]

    if abs(song.melody.duration - song.performance.duration) / song.melody.duration < DURATION_TOLERANCE:
        max_match = max(candidates, key=lambda x: len(x))
        structural_mapping = StructuralMapping(max_match, 1)
        structural_candidates = [
            candidate.update_score(structural_mapping(candidate, song.melody.duration))
            for candidate in candidates
        ]
        structural_score, structural_subset = non_crossing_weighted_bi_interval_scheduling(
            structural_candidates, return_subset=True, verbose=False)
        recurrence = sum(m.sum_miss + len(m) for m in structural_subset) / len(song.melody)
        if recurrence < MELODY_MIN_RECURRENCE:
            structural_subset = None
    else:
        structural_subset = None

    output: dict[int, list[Result]] = dict()

    for i, score_model in enumerate(score_models):

        score_model.load_song_stats(song_stats(song.melody, song.performance))
        scores = score_model(candidates)

        results: list[Result] = []

        for m in search_range:
            updated_candidates: list[FrozenMatch] = []
            for candidate, score in zip(candidates, scores):
                if score >= m:
                    updated_candidates.append(candidate.update_score(score))
            opt_score, opt_subset = weighted_interval_scheduling(
                updated_candidates, return_subset=True, verbose=False)
            concat_events = concat_matches(opt_subset)
            if structural_subset is not None:
                concat_events = filter_within(concat_events, structural_subset)
            assert song.ground_truth is not None
            result = evaluate_melody(song.ground_truth, concat_events, plot=False)
            results.append(Result(
                min_score=m, f1=result.f1_score, precision=result.precision, recall=result.recall))
        
        output[i] = results
    
    return output


def search_min_score(
    dataset: Dataset,
    candidates_dir: PathLike,
    min_length: int | None,
    search_range: Iterable[float],
    score_models: Iterable[ScoreModel],
    verbose: bool = True,
    n_jobs: int = 1
    ) -> list[Result]:
    """
    Compute min_score results for each song in the dataset, possibly in parallel.
    Returns a flat list of Result for all songs and all min_score values.
    """
    candidates_dir = Path(candidates_dir)
    results: list[dict[int, list[Result]]] = []
    if n_jobs == 1:
        progress_bar = tqdm(dataset, desc="Min score", disable=not verbose)
        for song in progress_bar:
            song_results = _min_score_one_song((song, candidates_dir, min_length, search_range, score_models))
            results.append(song_results)
        progress_bar.close()
    else:
        overall_pbar = None
        if verbose:
            overall_pbar = tqdm(total=len(dataset), desc=f"Min score (n_jobs={n_jobs})")
        args_list = [(song, candidates_dir, min_length, search_range, score_models) for song in dataset]
        max_workers = n_jobs if n_jobs > 0 else None
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_song = {
                executor.submit(_min_score_one_song, args): args[0].name
                for args in args_list
            }
            for future in as_completed(future_to_song):
                song_name = future_to_song[future]
                try:
                    song_results = future.result()
                    results.append(song_results)
                    if overall_pbar:
                        overall_pbar.update(1)
                except Exception as exc:
                    print(f"Song <{song_name}> generated an exception: {exc}")
                    if overall_pbar:
                        overall_pbar.update(1)
        if overall_pbar:
            overall_pbar.close()
    
    # {model_idx: {min_score: {f1: ...}}}
    agg_results: dict[int, dict[float, dict[str, float]]] = {
        i: defaultdict(lambda: {"f1": 0.0, "precision": 0.0, "recall": 0.0})
        for i in range(len(list(score_models)))}

    for r in results:
        for i, d in r.items():
            for entry in d:
                agg_results[i][entry["min_score"]]["f1"] += entry["f1"] / len(dataset)
                agg_results[i][entry["min_score"]]["precision"] += entry["precision"] / len(dataset)
                agg_results[i][entry["min_score"]]["recall"] += entry["recall"] / len(dataset)

    max_f1_min_scores = [max(agg, key=lambda x: agg[x]["f1"]) for agg in agg_results.values()]

    # print(f"Max F1: Min score {max_f1_min_score}, F1 {agg_results[max_f1_min_score]['f1']}, Precision {agg_results[max_f1_min_score]['precision']}, Recall {agg_results[max_f1_min_score]['recall']}")
    output: list[Result] = []
    for i, ms in enumerate(max_f1_min_scores):
        output.append(Result(
            min_score=ms,
            f1=agg_results[i][ms]["f1"],
            precision=agg_results[i][ms]["precision"],
            recall=agg_results[i][ms]["recall"],
        ))
    return output
    
    