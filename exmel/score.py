from typing import Protocol, Sequence
from exmel.event import MidiEvent
from math import log10

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