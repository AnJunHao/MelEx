# WISP: Weighted Interval Scheduling Problem

import bisect
from typing import Protocol, Sequence
from dataclasses import dataclass
from tqdm import tqdm

class EventProtocol(Protocol):
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def score(self) -> float: ...

def find_last_non_overlapping(ends: Sequence[float], start: float) -> int:
    # Find the last event that ends before start
    return bisect.bisect_right(ends, start) - 1

def weighted_interval_scheduling[T: EventProtocol](
    events: Sequence[T],
    verbose: bool = False
    ) -> tuple[float, list[T]]:
    if not events:
        return 0.0, []
    
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
    max_score, optimal_subset = weighted_interval_scheduling(events)
    assert max_score == 8
    assert optimal_subset == [
        _Event(0, 3, 3),
        _Event(3, 9, 5),
    ]
    print(f"The maximum score is: {max_score}")
    print(f"The optimal subset contains {len(optimal_subset)} events:")
    for event in optimal_subset:
        print(f"  Event: start={event.start}, end={event.end}, score={event.score}")