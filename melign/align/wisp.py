# WISP: Weighted Interval Scheduling Problem

import bisect
from typing import Protocol, overload, Literal
from collections.abc import Sequence
from dataclasses import dataclass
from tqdm.auto import tqdm
from math import log2, ceil

class Event(Protocol):
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def score(self) -> float: ...

def find_last_non_overlapping(ends: Sequence[float], start: float) -> int:
    # Find the last event that ends before start
    return bisect.bisect_right(ends, start) - 1

@overload
def weighted_interval_scheduling[T: Event](
    events: Sequence[T],
    return_subset: Literal[False],
    verbose: bool = False,
) -> float: ...
@overload
def weighted_interval_scheduling[T: Event](
    events: Sequence[T],
    return_subset: Literal[True],
    verbose: bool = False,
) -> tuple[float, list[T]]: ...
def weighted_interval_scheduling[T: Event](
    events: Sequence[T],
    return_subset: bool,
    verbose: bool = False
    ) -> tuple[float, list[T]] | float:

    if not events:
        if return_subset:
            return 0.0, []
        else:
            return 0.0
    
    # Sort events by their end time
    sorted_events = sorted(events, key=lambda x: x.end)
    ends = [event.end for event in sorted_events]

    # DP array to store the maximum score
    n = len(sorted_events)
    dp: list[float] = [0] * n

    # Initialize the first event's score
    dp[0] = sorted_events[0].score

    # Fill the dp array using the recurrence relation
    for i in tqdm(range(1, n), disable=not verbose):
        # Option 1: Don't select current event
        exclude_score = dp[i-1]
        
        # Option 2: Select current event
        include_score = sorted_events[i].score
        last_non_overlap = find_last_non_overlapping(ends, sorted_events[i].start)
        if last_non_overlap != -1:
            include_score += dp[last_non_overlap]
        
        # Maximum of including or excluding the event
        dp[i] = max(exclude_score, include_score)

    if not return_subset:
        return dp[-1]
    
    # Backtrack to find the optimal subset
    optimal_events: list[T] = []
    i = n - 1
    
    while i >= 0:
        if i == 0:
            # First event - include it if it contributes to the optimal solution
            if dp[i] > 0:
                optimal_events.append(sorted_events[i])
            break
        
        # Check if current event was included in the optimal solution
        exclude_score = dp[i-1]
        include_score = sorted_events[i].score
        last_non_overlap = find_last_non_overlapping(ends, sorted_events[i].start)
        if last_non_overlap != -1:
            include_score += dp[last_non_overlap]
        
        if include_score > exclude_score:
            # Current event was included
            optimal_events.append(sorted_events[i])
            # Move to the last non-overlapping event
            i = last_non_overlap
        else:
            # Current event was not included
            i -= 1
    
    # Reverse to get events in chronological order
    optimal_events.reverse()
    
    # The answer is the maximum score and the optimal subset
    return dp[-1], optimal_events

class Detection(Protocol):
    @property
    def tp(self) -> int: ...
    @property
    def fp(self) -> int: ...
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...

@overload
def f1_optimal_interval_scheduling[T: Detection](
    detections: Sequence[T],
    num_gt: int,
    return_subset: Literal[False],
    epsilon: float = 1e-4,
    verbose: bool = False,
) -> float: ...
@overload
def f1_optimal_interval_scheduling[T: Detection](
    detections: Sequence[T],
    num_gt: int,
    return_subset: Literal[True],
    epsilon: float = 1e-4,
    verbose: bool = False,
) -> tuple[float, list[T]]: ...
def f1_optimal_interval_scheduling[T: Detection](
    detections: Sequence[T],
    num_gt: int,
    return_subset: bool,
    epsilon: float = 1e-4,
    verbose: bool = False,
) -> tuple[float, list[T]] | float:

    if not detections or num_gt <= 0:
        return (0.0, []) if return_subset else 0.0

    @dataclass(slots=True, frozen=True)
    class _Event:
        start: float
        end: float
        score: float
        original: T

    def _build_events(R: float) -> list[_Event]:
        coef_tp = 2.0 - R     # (2-R)
        coef_fp = -R          # (-R)
        return [
            _Event(
                start=d.start,
                end=d.end,
                score=coef_tp * d.tp + coef_fp * d.fp,
                original=d,
            )
            for d in detections]

    low, high = 0.0, 1.0                     # F1 is always in [0,1]
    num_iter = ceil(log2(1 / epsilon))
    with tqdm(
        total=num_iter,
        desc="Binary search",
        disable=not verbose,
    ) as pbar:
        while high - low > epsilon:
            mid = (low + high) / 2.0
            best_weight = weighted_interval_scheduling(
                _build_events(mid), return_subset=False, verbose=False
            )
            if best_weight >= mid * num_gt:      # feasibility test
                low = mid                        # F1 >= mid attainable
            else:
                high = mid                       # not attainable
            _ = pbar.update(1)

    max_f1 = low                             # within `epsilon` of optimum

    if not return_subset:
        return max_f1

    _, chosen_events = weighted_interval_scheduling(
        _build_events(max_f1), return_subset=True, verbose=False
    )
    chosen_detections = [ev.original for ev in chosen_events]
    return max_f1, chosen_detections
    
if __name__ == "__main__":

    @dataclass(slots=True, frozen=True)
    class _Event:
        start: float
        end: float
        score: float

    # Example usage
    events = [
        _Event(0, 3, 3),
        _Event(2, 4, 2),
        _Event(0, 5, 4),
        _Event(3, 6, 1),
        _Event(4, 7, 2),
        _Event(3, 9, 5),
        _Event(3, 9, 5),
        _Event(5, 10, 2),
        _Event(8, 10, 1),
    ]

    # Solve the WISP problem
    max_score, optimal_subset = weighted_interval_scheduling(events, return_subset=True)
    assert max_score == 8
    assert optimal_subset == [
        _Event(0, 3, 3),
        _Event(3, 9, 5),
    ]
    print(f"The maximum score is: {max_score}")
    print(f"The optimal subset contains {len(optimal_subset)} events:")
    for event in optimal_subset:
        print(f"  Event: start={event.start}, end={event.end}, score={event.score}")