from dataclasses import dataclass
from functools import cached_property

from exmel.sequence import Melody, MelodyLike, Performance, PerformanceLike
from exmel.event import MidiEvent

@dataclass(slots=True)
class Match:
    events: list[MidiEvent]
    head_miss: int
    tail_miss: int
    sum_miss: int
    sum_error: float

    @property
    def start(self) -> float:
        return self.events[0].time
    
    @property
    def end(self) -> float:
        return self.events[-1].time
    
    @property
    def score(self) -> float:
        return score(self)

    def __len__(self) -> int:
        return len(self.events)

    def freeze(self) -> 'FrozenMatch':
        return FrozenMatch(
            events=tuple(self.events),
            head_miss=self.head_miss,
            tail_miss=self.tail_miss,
            sum_miss=self.sum_miss, 
            sum_error=self.sum_error)
    
    def unfreeze(self) -> 'Match':
        return self

@dataclass(slots=True, frozen=True)
class FrozenMatch:
    events: tuple[MidiEvent, ...]
    head_miss: int
    tail_miss: int
    sum_miss: int
    sum_error: float
    
    @cached_property
    def start(self) -> float:
        return self.events[0].time
    
    @cached_property
    def end(self) -> float:
        return self.events[-1].time

    @cached_property
    def score(self) -> float:
        return score(self)

    def __len__(self) -> int:
        return len(self.events)

    def freeze(self) -> 'FrozenMatch':
        return self

    def unfreeze(self) -> Match:
        return Match(
            events=list(self.events),
            head_miss=self.head_miss,
            tail_miss=self.tail_miss,
            sum_miss=self.sum_miss, 
            sum_error=self.sum_error)

type MatchLike = Match | FrozenMatch

def score(match: MatchLike) -> float:
    return sum(event.velocity for event in match.events)

def scan(
    melody: MelodyLike,
    performance: PerformanceLike,
    local_tolerance: float = 0.1,
    miss_tolerance: int = 2,
    candidate_min_length: int = 10,
) -> list[Match]:

    if not isinstance(melody, Melody):
        melody = Melody(melody)
    if not isinstance(performance, Performance):
        performance = Performance(performance)
    
    if len(melody) == 0:
        return []
    if len(melody) == 1:
        tv_list = performance.events_by_note[melody[0].note]
        return [Match([MidiEvent(tv[0], melody[0].note, tv[1])], 0, 0, 0, 0)
                for tv in tv_list]

    candidates = scan(
        melody[:-1], performance,
        local_tolerance, miss_tolerance, candidate_min_length)

    if len(candidates) == 0:
        return []

    output = []
    for candidate in candidates:

        # Add stopped candidates to output
        if candidate.tail_miss > miss_tolerance:
            if len(candidate) >= candidate_min_length:
                output.append(candidate)
            continue

        # Scan live candidates
        melody_shift = melody[-1] - melody[-2-candidate.tail_miss]
        predicted_event = candidate.events[-1] + melody_shift

        nearest = performance.nearest(predicted_event)
        if nearest is not None:
            diff = nearest - predicted_event
            if abs(diff.time) / abs(melody_shift.time) <= local_tolerance:
                candidate.events.append(nearest)
                candidate.sum_miss += candidate.tail_miss
                candidate.tail_miss = 0
                candidate.sum_error += abs(diff.time)
            else:
                candidate.tail_miss += 1
        else:
            # No such note in piano. Maybe the melody is wrong.
            candidate.tail_miss += 1
            # Maybe we can tolerate this note?

        if candidate.tail_miss <= miss_tolerance or len(candidate) >= candidate_min_length:
            # Add live candidates & long enough stopped candidates
            output.append(candidate)

    return output