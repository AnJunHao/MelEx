from typing import Protocol, Sequence
from math import log10

from exmel.event import MidiEvent
from exmel.sequence import MelodyLike, Melody

class MatchLike(Protocol):
    @property
    def events(self) -> Sequence[MidiEvent]: ...
    @property
    def sum_miss(self) -> int: ...
    @property
    def sum_error(self) -> float: ...

def duration(match: MatchLike) -> float:
    return match.events[-1].time - match.events[0].time

def sum_velocity(match: MatchLike) -> float:
    return sum(event.velocity for event in match.events) / 128

def weighted_sum_velocity(match: MatchLike) -> float:
    return sum_velocity(match) * (1 - match.sum_miss / len(match.events))

def duration_adjusted_weighted_sum_velocity(match: MatchLike) -> float:
    if duration(match) > 1:
        return weighted_sum_velocity(match) * log10(duration(match))
    else:
        return 0

def contribution(match: MatchLike, gt: MelodyLike, tolerance: float = 0.1) -> int:
    """
    Contribution of a match to the F1 score is measured by:
    tp - fp (https://poe.com/s/wvWUr2WYG87FLQBsfIEH)
    """
    if not isinstance(gt, Melody):
        gt = Melody(gt)
    gt %= 12
    tp = 0
    fp = 0
    for event in match.events:
        nearest = gt.nearest(event)
        if nearest is not None and abs(nearest.time - event.time) < tolerance:
            tp += 1
        else:
            fp += 1
    return tp - fp