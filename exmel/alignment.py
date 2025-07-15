from dataclasses import dataclass, field
from typing import Any, Callable, overload, Literal, Iterable
import warnings
from tqdm import tqdm

# from icecream import ic

from exmel.sequence import Melody, MelodyLike, Performance, PerformanceLike
from exmel.event import MidiEvent
from exmel.wisp import weighted_interval_scheduling
from exmel.score import duration_adjusted_weighted_sum_velocity

@dataclass
class Match:
    events: list[MidiEvent]
    speed: float
    tail_miss: int
    sum_miss: int
    sum_error: float
    score_func: Callable[['Match'], float]
    melody_start: float
    melody_end: float

    @property
    def start(self) -> float:
        return self.events[0].time
    
    @property
    def end(self) -> float:
        return self.events[-1].time
    
    @property
    def score(self) -> float:
        return self.score_func(self)

    def freeze(self) -> 'FrozenMatch':
        return FrozenMatch(
            events=tuple(self.events),
            speed=self.speed,
            sum_miss=self.sum_miss,
            sum_error=self.sum_error,
            melody_start=self.melody_start,
            melody_end=self.melody_end,
            start=self.events[0].time,
            end=self.events[-1].time,
            score=self.score_func(self)
        )

    def update_speed_before_append(self, speed: float) -> None:
        current_sum = self.speed * len(self.events)
        self.speed = (current_sum + speed) / (len(self.events) + 1)

    def copy(self) -> 'Match':
        return Match(
            events=self.events.copy(),
            speed=self.speed,
            tail_miss=self.tail_miss,
            sum_miss=self.sum_miss,
            sum_error=self.sum_error,
            score_func=self.score_func,
            melody_start=self.melody_start,
            melody_end=self.melody_end
        )

    def __len__(self) -> int:
        return len(self.events)

@dataclass(frozen=True, slots=True)
class FrozenMatch:
    events: tuple[MidiEvent, ...]
    speed: float
    sum_miss: int
    sum_error: float
    melody_start: float
    melody_end: float
    start: float = field(hash=True)
    end: float = field(hash=True)
    score: float = field(hash=True)

type MatchLike = Match | FrozenMatch

def concat_matches(matches: Iterable[MatchLike]) -> list[MidiEvent]:
    return [event
            for match in matches
            for event in match.events]

@overload
def scan(
    melody: MelodyLike,
    performance: PerformanceLike,
    score_func: Callable[[MatchLike], float],
    same_key: bool,
    same_speed: bool,
    speed_prior: float,
    variable_tail: bool,
    local_rel_tolerance: float,
    local_abs_tolerance: float,
    miss_tolerance: int,
    candidate_min_score: float,
    recursive_call: Literal[False]
) -> list[FrozenMatch]: ...
@overload
def scan(
    melody: MelodyLike,
    performance: PerformanceLike,
    score_func: Callable[[MatchLike], float],
    same_key: bool,
    same_speed: bool,
    speed_prior: float,
    variable_tail: bool,
    local_rel_tolerance: float,
    local_abs_tolerance: float,
    miss_tolerance: int,
    candidate_min_score: float,
    recursive_call: Literal[True]
) -> tuple[list[Match], list[FrozenMatch]]: ...
def scan(
    melody: MelodyLike,
    performance: PerformanceLike,
    score_func: Callable[[MatchLike], float],
    same_key: bool,
    same_speed: bool,
    speed_prior: float,
    variable_tail: bool,
    local_rel_tolerance: float,
    local_abs_tolerance: float,
    miss_tolerance: int,
    candidate_min_score: float,
    recursive_call: bool = False
) -> list[FrozenMatch] | tuple[list[Match], list[FrozenMatch]]:
    """
    Scan the melody from left to right, and find longest possible matches in the performance.

    If candidates failed to match melody for `miss_tolerance` times in a row,
    we will not continue to scan this candidate further into the melody.
    However, if the candidate has good enough score, we will freeze it and add it to the output.

    Matches with score less than `candidate_min_score` may be contained in the output,
    as long as it does not exceed `miss_tolerance`.
    Setting `enforce_min_score` to True will force all matches to have score
    greater than `candidate_min_score` before returning.
    """

    #################### Type Enforcement ####################
    if not isinstance(melody, Melody):
        melody = Melody(melody)
    if not isinstance(performance, Performance):
        performance = Performance(performance)
    
    #################### Base Cases ####################
    if len(melody) == 0:
        raise ValueError("Melody is empty")
    elif len(melody) == 1:
        if not recursive_call:
            raise ValueError("Melody has only one event")
        if same_key:
            tv_list = performance.events_by_note.get(melody[0].note, [])
            lives = [Match([MidiEvent(tv[0], melody[0].note, tv[1])],
                            speed_prior, 0, 0, 0.0,
                            score_func,
                            melody[0].time, melody[0].time)
                        for tv in tv_list]
        else:
            lives =  [Match([event],
                            speed_prior, 0, 0, 0.0,
                            score_func,
                            melody[0].time, melody[0].time)
                        for event in performance.global_events]
        return lives, []

    #################### Recursive Call ####################
    lives, frozens = scan(
        melody[:-1], performance, 
        score_func, same_key, same_speed, speed_prior, variable_tail,
        local_rel_tolerance, local_abs_tolerance,
        miss_tolerance, candidate_min_score,
        recursive_call=True)

    if len(lives) == 0:
        if recursive_call:
            return [], frozens
        else:
            return frozens

    #################### Scan Step ####################
    new_lives: list[Match] = []

    for candidate in lives:

        # Scan live candidates
        melody_shift = melody[-1] - melody[-2-candidate.tail_miss]

        #################### Speed Not Determined ####################
        # if not same_speed and len(candidate) == 1:
        #     from_ = candidate.events[-1] + melody_shift*0.5
        #     to_ = candidate.events[-1] + melody_shift*2
        #     events = performance.between(from_, to_)
        #     if len(events) == 0:
        #         candidate.tail_miss += 1
        #         if candidate.tail_miss <= miss_tolerance:
        #             new_lives.append(candidate)
        #         continue
        #     for event in events:
        #         copy = candidate.copy()
        #         actual_shift = event - copy.events[-1]
        #         speed_ratio = actual_shift / melody_shift
        #         copy.update_speed_before_append(speed_ratio)
        #         copy.events.append(event)
        #         copy.sum_miss += copy.tail_miss
        #         copy.tail_miss = 0
        #         copy.melody_end = melody[-1].time
        #         new_lives.append(copy)
        #     continue

        #################### Speed Determined ####################
        predicted_event = candidate.events[-1] + melody_shift*candidate.speed
        nearest = performance.nearest(predicted_event, start=candidate.events[-1].time)
        if nearest is not None:
            diff = nearest - predicted_event
            tolerance = max(melody_shift.time * candidate.speed * local_rel_tolerance,
                            local_abs_tolerance)
            if abs(diff.time) <= tolerance:

                if not same_speed:
                    #################### Update Speed ####################
                    actual_shift = nearest - candidate.events[-1]
                    speed_ratio = actual_shift / melody_shift
                    candidate.update_speed_before_append(speed_ratio)
                
                #################### Append Event ####################
                candidate.events.append(nearest)
                candidate.sum_miss += candidate.tail_miss
                candidate.tail_miss = 0
                candidate.sum_error += abs(diff.time)
                candidate.melody_end = melody[-1].time
            
        #################### Update Misses ####################
            else:
                candidate.tail_miss += 1
        else:
            # No such note in piano. Maybe the melody is wrong.
            candidate.tail_miss += 1
            # Maybe we can tolerate this note?

        #################### Prepare Outputs ####################
        if candidate.tail_miss <= miss_tolerance:
            # Add live candidates 
            new_lives.append(candidate)

        #################### Variable Tail ####################
            if (variable_tail and
                candidate.tail_miss == 0 and
                candidate.score >= candidate_min_score):
                copy = candidate.copy()
                frozens.append(copy.freeze())
        elif not variable_tail and candidate.score >= candidate_min_score:
            # good enough candidates are already added to frozen if variable_tail
            # so we only need to freeze when variable_tail is False
            frozens.append(candidate.freeze())

    if not recursive_call:
        if variable_tail:
            return frozens
        else:
            frozens.extend(c.freeze() for c in new_lives if c.score >= candidate_min_score)
            return frozens

    return new_lives, frozens

@dataclass
class AlignConfig:
    score_func: Callable[[MatchLike], float] = duration_adjusted_weighted_sum_velocity
    same_key: bool = False
    same_speed: bool = False
    speed_prior: float = 1.0
    variable_tail: bool = True
    local_tolerance: float = 0.5
    miss_tolerance: int = 2
    candidate_min_score: float = 10
    candidate_min_length: int = 15
    hop_length: int = 2
    split_melody: bool = True

default_config = AlignConfig()

@dataclass(frozen=True, slots=True)
class Alignment:
    events: list[MidiEvent]
    matches: list[FrozenMatch]
    discarded_matches: list[FrozenMatch]
    score: float
    sum_miss: int
    sum_error: float

def align(
    melody: MelodyLike,
    performance: PerformanceLike,
    config: AlignConfig = default_config,
    verbose: bool = True,
) -> Alignment:

    if config.hop_length < 1:
        raise ValueError("hop_length must be at least 1")
    if not isinstance(melody, Melody):
        melody = Melody(melody)
    if not isinstance(performance, Performance):
        performance = Performance(performance)

    candidates: list[FrozenMatch] = []

    if config.split_melody:
        melodies = melody.split()
    else:
        melodies = [melody]

    local_abs_tolerance = melody.mean_diff() * config.local_tolerance

    total_scans = sum(len(range(0,
                                len(melody) - config.candidate_min_length,
                                config.hop_length)) 
                     for melody in melodies if len(melody) >= config.candidate_min_length)
    
    with tqdm(total=total_scans, desc="Scanning alignments", disable=not verbose) as pbar:
        for melody in melodies:
            for scan_start in range(0,
                                    len(melody) - config.candidate_min_length,
                                    config.hop_length):
                candidates.extend(scan(
                    melody[scan_start:],
                    performance,
                    config.score_func,
                    config.same_key,
                    config.same_speed,
                    config.speed_prior,
                    config.variable_tail,
                    config.local_tolerance,
                    local_abs_tolerance,
                    config.miss_tolerance,
                    config.candidate_min_score,
                    recursive_call=False))
                pbar.update(1)
    
    if len(candidates) == 0:
        return Alignment([], [], [], 0, 0, 0)

    opt_score, opt_subset = weighted_interval_scheduling(candidates, verbose=False)
    discarded_matches = [match for match in candidates if match not in opt_subset]
    concat_events = concat_matches(opt_subset)
    return Alignment(concat_events, opt_subset, discarded_matches, opt_score,
                    sum(match.sum_miss for match in opt_subset),
                    sum(match.sum_error for match in opt_subset))