from typing import Protocol
from exmel.event import MidiEvent
from dataclasses import dataclass
from math import log10

class MatchLike(Protocol):
    events: list[MidiEvent]
    sum_miss: int
    sum_error: float

class ScoreWrapper(Protocol):
    
    candidate_min_score: float
    candidate_min_length: int

    def score(self, match: MatchLike) -> float: ...
    def __call__(self, match: MatchLike) -> float: ...

@dataclass
class SumVelocity(ScoreWrapper):
    candidate_min_score: float = 10
    candidate_min_length: int = 15
    def score(self, match: MatchLike) -> float:
        return sum(event.velocity for event in match.events) / 128
    def __call__(self, match: MatchLike) -> float:
        return self.score(match)

sum_velocity = SumVelocity()

@dataclass
class WeightedSumVelocity(ScoreWrapper):
    candidate_min_score: float = 10
    candidate_min_length: int = 15
    def score(self, match: MatchLike) -> float:
        return sum_velocity(match) * (1 - match.sum_miss / len(match.events))
    def __call__(self, match: MatchLike) -> float:
        return self.score(match)

weighted_sum_velocity = WeightedSumVelocity()

@dataclass
class DurationAdjustedWeightedSumVelocity(ScoreWrapper):
    candidate_min_score: float = 10
    candidate_min_length: int = 15
    def score(self, match: MatchLike) -> float:
        if duration(match) > 1:
            return weighted_sum_velocity(match) * log10(duration(match))
        else:
            return 0
    def __call__(self, match: MatchLike) -> float:
        return self.score(match)

duration_adjusted_weighted_sum_velocity = DurationAdjustedWeightedSumVelocity()

@dataclass
class Duration(ScoreWrapper):
    candidate_min_score: float = 10
    candidate_min_length: int = 15
    def score(self, match: MatchLike) -> float:
        return match.events[-1].time - match.events[0].time
    def __call__(self, match: MatchLike) -> float:
        return self.score(match)

duration = Duration()

@dataclass
class WeightedDuration(ScoreWrapper):
    candidate_min_score: float = 10
    candidate_min_length: int = 15
    def score(self, match: MatchLike) -> float:
        return duration(match) * (1 - match.sum_miss / len(match.events))
    def __call__(self, match: MatchLike) -> float:
        return self.score(match)

weighted_duration = WeightedDuration()