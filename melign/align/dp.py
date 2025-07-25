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

class Pair(Protocol):
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def melody_start(self) -> float: ...
    @property
    def melody_end(self) -> float: ...
    @property
    def score(self) -> float: ...

class _Fenwick:
    __slots__ = ("n", "_tree")

    def __init__(self, n: int) -> None:
        self.n = n
        self._tree: list[tuple[float, int | None]] = [(0.0, None)] * (n + 1)

    # keep the *highest* score for every prefix
    def update(self, i: int, value: float, idx: int) -> None:
        while i <= self.n:
            if value > self._tree[i][0]:
                self._tree[i] = (value, idx)
            i += i & -i

    # best on range [1 .. i]
    def query(self, i: int) -> tuple[float, int | None]:
        best: tuple[float, int | None] = (0.0, None)
        while i:
            if self._tree[i][0] > best[0]:
                best = self._tree[i]
            i -= i & -i
        return best

@overload
def non_crossing_weighted_bi_interval_scheduling[T: Pair](
    pairs: Sequence[T],
    return_subset: Literal[False],
    verbose: bool = False,
) -> float: ...
@overload
def non_crossing_weighted_bi_interval_scheduling[T: Pair](
    pairs: Sequence[T],
    return_subset: Literal[True],
    verbose: bool = False,
) -> tuple[float, list[T]]: ...
def non_crossing_weighted_bi_interval_scheduling[T: Pair](
    pairs: Sequence[T],
    return_subset: bool,
    verbose: bool = False,
) -> float | tuple[float, list[T]]:
    """
    Maximum-weight subset of non-overlapping, non-crossing bi-interval pairs.

    Time complexity  :  O(n log n)
    Space complexity :  O(n)
    """
    n = len(pairs)
    if n == 0:
        return (0.0, []) if return_subset else 0.0

    # ----------------------  pre-processing  ----------------------
    # internal record: (idx, a_start, a_end, b_start, b_end, score)
    recs: list[tuple[int, float, float, float, float, float]] = [
        (i, p.start, p.end, p.melody_start, p.melody_end, p.score) for i, p in enumerate(pairs)
    ]

    # coordinate compression of all b_end values
    b_end_values = sorted({r[4] for r in recs})
    b_end_rank = {v: i + 1 for i, v in enumerate(b_end_values)}  # 1-based

    fenwick = _Fenwick(len(b_end_values))

    # sort helpers
    by_a_start = sorted(recs, key=lambda r: r[1])
    by_a_end = sorted(recs, key=lambda r: r[2])

    # DP tables
    best_score: list[float] = [0.0] * n
    predecessor: list[int | None] = [None] * n

    # --------------------------  sweep  ---------------------------
    ptr = 0  # walks through by_a_end
    for r in by_a_start:
        idx, a_s, _, b_s, _, _ = r

        # 1) insert into Fenwick every pair that already *ended* on A
        while ptr < n and by_a_end[ptr][2] < a_s:
            q = by_a_end[ptr]
            fenwick.update(b_end_rank[q[4]], best_score[q[0]], q[0])
            ptr += 1

        # 2) query: best chain whose b_end < current b_start
        pos = bisect.bisect_left(b_end_values, b_s)
        prev_best, prev_idx = fenwick.query(pos) if pos else (0.0, None)

        # 3) extend the chain with the current pair
        best_score[idx] = prev_best + r[5]
        predecessor[idx] = prev_idx

        if verbose:
            print(
                f"processing #{idx}:  score={r[5]},  extend={prev_best} "
                f"-> total={best_score[idx]}"
            )

    # -----------------------  reconstruction  ---------------------
    max_idx = max(range(n), key=best_score.__getitem__)
    max_weight = best_score[max_idx]

    if not return_subset:
        return max_weight

    chain_indices: list[int] = []
    cur: int | None = max_idx
    while cur is not None:
        chain_indices.append(cur)
        cur = predecessor[cur]
    chain_indices.reverse()

    return max_weight, [pairs[i] for i in chain_indices]


# --------------------- Tests ---------------------
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

    @dataclass(slots=True, frozen=True)
    class _Pair:
        start: float
        end: float
        melody_start: float
        melody_end: float
        score: float

    # 1) empty input
    assert non_crossing_weighted_bi_interval_scheduling([], False) == 0.0

    # 2) three pairs touching at endpoints
    p1 = _Pair(0, 10, 0, 10, 5)
    p2 = _Pair(10, 20, 10, 20, 7)
    p3 = _Pair(20, 30, 20, 30, 4)
    w, chosen = non_crossing_weighted_bi_interval_scheduling([p1, p2, p3], True)
    assert w == 9 and chosen == [p1, p3], f"Got: {w=} and {chosen=}"

    p1 = _Pair(0, 10, 0, 10, 5)
    p2 = _Pair(10, 20, 10, 20, 10)
    p3 = _Pair(20, 30, 20, 30, 4)
    w, chosen = non_crossing_weighted_bi_interval_scheduling([p1, p2, p3], True)
    assert w == 10 and chosen == [p2], f"Got: {w=} and {chosen=}"

    # 3) overlapping on A → only one can be taken
    q1 = _Pair(0, 15, 0, 10, 8)
    q2 = _Pair(10, 20, 10, 30, 9)  # overlaps on A with q1
    assert non_crossing_weighted_bi_interval_scheduling([q1, q2], False) == 9

    # 4) crossing pairs
    c1 = _Pair(0, 10, 100, 120, 6)
    c2 = _Pair(5, 15, 0, 50, 7)  # crosses c1
    c3 = _Pair(20, 30, 130, 160, 9)
    # best is {c2, c3}
    w, subset = non_crossing_weighted_bi_interval_scheduling([c1, c2, c3], True)
    assert w == 16 and subset == [c2, c3], f"Got: {w=} and {subset=}"

    # 5) Complex example
    p1 = _Pair(0, 1, 0, 2, 1)
    p2 = _Pair(2, 3, 5, 6, 2)
    p3 = _Pair(4, 5, 1, 4, 4)
    p4 = _Pair(6, 8, 3, 7, 5)
    p5 = _Pair(7, 9, 8, 9, 3)
    p6 = _Pair(-2, -1, 10, 11, 8)
    p7 = _Pair(10, 11, 2.1, 2.2, 7.1)
    p8 = _Pair(1.1, 1.2, 4.1, 4.2, 2.2)

    w, subset = non_crossing_weighted_bi_interval_scheduling([p1, p2, p3, p4, p5], True)
    assert w == 7 and subset == [p3, p5], f"Got: {w=} and {subset=}"

    w, subset = non_crossing_weighted_bi_interval_scheduling([p1, p2, p3, p4, p5, p6], True)
    assert w == 8 and subset == [p6], f"Got: {w=} and {subset=}"

    w, subset = non_crossing_weighted_bi_interval_scheduling([p1, p2, p3, p4, p5, p6, p7], True)
    assert w == 8.1 and subset == [p1, p7], f"Got: {w=} and {subset=}"

    w, subset = non_crossing_weighted_bi_interval_scheduling([p1, p2, p3, p4, p5, p6, p7, p8], True)
    assert w == 8.2 and subset == [p1, p8, p2, p5], f"Got: {w=} and {subset=}"

    print("All tests passed.")