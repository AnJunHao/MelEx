import pickle
from typing import Iterable, TypedDict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from collections import defaultdict

from melign.align.alignment import Alignment, FrozenMatch, concat_matches
from melign.align.score import ScoreModel
from melign.align.wisp import weighted_interval_scheduling
from melign.align.eval_and_vis import evaluate_melody
from melign.api.dataset import Dataset, Song
from melign.data.io import PathLike
from melign.data.sequence import song_stats

class Result(TypedDict):
    min_score: float
    f1: float
    precision: float
    recall: float

def _min_score_one_song(args: tuple[Song, Path, int | None, Iterable[float], ScoreModel]) -> list[Result]:
    song, candidates_dir, min_length, search_range, score_model = args

    candidates: list[FrozenMatch] = pickle.load(open(candidates_dir / f"{song.name}.pkl", "rb"))

    if min_length is not None:
        candidates = [c for c in candidates if len(c.events) >= min_length]

    score_model.load_song_stats(song_stats(song.melody, song.performance))
    scores = score_model(candidates)

    print(scores[0])

    results: list[Result] = []

    for m in search_range:
        updated_candidates: list[FrozenMatch] = []
        for candidate, score in zip(candidates, scores):
            if score >= m:
                updated_candidates.append(candidate.update_score(score))
        opt_score, opt_subset = weighted_interval_scheduling(
            updated_candidates, return_subset=True, verbose=False)
        concat_events = concat_matches(opt_subset)
        assert song.ground_truth is not None
        result = evaluate_melody(song.ground_truth, concat_events, plot=False)
        results.append(Result(
            min_score=m, f1=result.f1_score, precision=result.precision, recall=result.recall))
    
    return results


def search_min_score(
    dataset: Dataset,
    candidates_dir: PathLike,
    min_length: int | None,
    search_range: Iterable[float],
    score_model: ScoreModel,
    verbose: bool = True,
    n_jobs: int = 1
    ) -> tuple[dict[float, dict[str, float]], Result]:
    """
    Compute min_score results for each song in the dataset, possibly in parallel.
    Returns a flat list of Result for all songs and all min_score values.
    """
    candidates_dir = Path(candidates_dir)
    results: list[Result] = []
    if n_jobs == 1:
        progress_bar = tqdm(dataset, desc="Min score", disable=not verbose)
        for song in progress_bar:
            song_results = _min_score_one_song((song, candidates_dir, min_length, search_range, score_model))
            results.extend(song_results)
        progress_bar.close()
    else:
        overall_pbar = None
        if verbose:
            overall_pbar = tqdm(total=len(dataset), desc=f"Min score (n_jobs={n_jobs})")
        args_list = [(song, candidates_dir, min_length, search_range, score_model) for song in dataset]
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
                    results.extend(song_results)
                    if overall_pbar:
                        overall_pbar.update(1)
                except Exception as exc:
                    print(f"Song <{song_name}> generated an exception: {exc}")
                    if overall_pbar:
                        overall_pbar.update(1)
        if overall_pbar:
            overall_pbar.close()

    agg_results: defaultdict[float, dict[str, float]] = defaultdict(lambda: {"f1": 0.0, "precision": 0.0, "recall": 0.0})

    for r in results:
        agg_results[r["min_score"]]["f1"] += r["f1"] / len(dataset)
        agg_results[r["min_score"]]["precision"] += r["precision"] / len(dataset)
        agg_results[r["min_score"]]["recall"] += r["recall"] / len(dataset)

    max_f1_min_score = max(agg_results, key=lambda x: agg_results[x]["f1"])

    # print(f"Max F1: Min score {max_f1_min_score}, F1 {agg_results[max_f1_min_score]['f1']}, Precision {agg_results[max_f1_min_score]['precision']}, Recall {agg_results[max_f1_min_score]['recall']}")
    
    return agg_results, Result(
        min_score=max_f1_min_score,
        f1=agg_results[max_f1_min_score]["f1"],
        precision=agg_results[max_f1_min_score]["precision"],
        recall=agg_results[max_f1_min_score]["recall"],
    )
    
    