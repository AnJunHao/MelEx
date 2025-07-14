from dataclasses import dataclass
from typing import Any, Callable, overload, cast
import warnings
from tqdm import tqdm

from exmel.sequence import Melody, MelodyLike, Performance, PerformanceLike
from exmel.event import MidiEvent
from exmel.wisp import weighted_interval_scheduling
from exmel.score import ScoreWrapper

@dataclass
class Match:
    events: list[MidiEvent]
    head_miss: int
    tail_miss: int
    sum_miss: int
    sum_error: float
    score_func: Callable[['Match'], float]
    melody_start: float
    melody_end: float

    def __setattr__(self, name: str, value) -> None:
        if hasattr(self, '_frozen') and self._frozen and name != "selected":
            raise AttributeError(f"Cannot modify frozen object: {name}")
        super().__setattr__(name, value)

    # def __getattribute__(self, name: str) -> Any:
    #     if name == 'events' and hasattr(self, '_frozen') and self._frozen:
    #         warnings.warn("Accessing Match.events after freezing, returning a tuple")
    #         return tuple(self.__dict__['events'])
    #     return super().__getattribute__(name)

    @property
    def start(self) -> float:
        return self.events[0].time
    
    @property
    def end(self) -> float:
        return self.events[-1].time
    
    @property
    def score(self) -> float:
        if not hasattr(self, '_score_cache'):
            self._score_cache = self.score_func(self)
        return self._score_cache

    def freeze(self) -> None:
        self._frozen = True

    def __len__(self) -> int:
        return len(self.events)

def concat_matches(matches: list[Match]) -> list[MidiEvent]:
    return [event
            for match in matches
            for event in match.__dict__['events']]

@overload
def scan(
    melody: MelodyLike,
    performance: PerformanceLike,
    score_func: ScoreWrapper, # Score wrapper contains candidate_min_score and candidate_min_length
    same_key: bool,
    local_tolerance: float,
    miss_tolerance: int,
    candidate_min_score: None,
    enforce_min_score: bool = False
) -> list[Match]: ...
@overload
def scan(
    melody: MelodyLike,
    performance: PerformanceLike,
    score_func: Callable[['Match'], float],
    same_key: bool,
    local_tolerance: float,
    miss_tolerance: int,
    candidate_min_score: float, # Supply this if you use plain function instead of score wrapper
    enforce_min_score: bool = False
) -> list[Match]: ...
def scan(
    melody: MelodyLike,
    performance: PerformanceLike,
    score_func: Callable[['Match'], float] | ScoreWrapper,
    same_key: bool,
    local_tolerance: float,
    miss_tolerance: int,
    candidate_min_score: float | None = None,
    enforce_min_score: bool = False
) -> list[Match]:
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

    if not isinstance(melody, Melody):
        melody = Melody(melody)
    if not isinstance(performance, Performance):
        performance = Performance(performance)
    if candidate_min_score is None:
        score_func = cast(ScoreWrapper, score_func) # For type narrowing
        candidate_min_score = score_func.candidate_min_score
    
    if len(melody) == 0:
        return []
    if len(melody) == 1:
        if same_key:
            tv_list = performance.events_by_note.get(melody[0].note, [])
            return [Match([MidiEvent(tv[0], melody[0].note, tv[1])],
                        0, 0, 0, 0, score_func,
                        melody[0].time, melody[0].time)
                    for tv in tv_list]
        else:
            return [Match([event],
                        0, 0, 0, 0, score_func,
                        melody[0].time, melody[0].time)
                    for event in performance.global_events]

    candidates = scan( 
        melody[:-1], performance, score_func,
        same_key, local_tolerance, miss_tolerance, candidate_min_score)

    if len(candidates) == 0:
        return []

    output: list[Match] = []
    for candidate in candidates:

        # Add frozen candidates to output
        if candidate.tail_miss > miss_tolerance:
            if candidate.score >= candidate_min_score:
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
                candidate.melody_end = melody[-1].time
            else:
                candidate.tail_miss += 1
        else:
            # No such note in piano. Maybe the melody is wrong.
            candidate.tail_miss += 1
            # Maybe we can tolerate this note?

        if candidate.tail_miss <= miss_tolerance:
            # Add live candidates 
            output.append(candidate)
        elif candidate.score >= candidate_min_score:
            # good enough candidates -> Freeze
            candidate.freeze()
            output.append(candidate)

    if enforce_min_score:
        output = [match for match in output if match.score >= candidate_min_score]

    return output

@dataclass(frozen=True, slots=True)
class Alignment:
    events: list[MidiEvent]
    matches: list[Match]
    discarded_matches: list[Match]
    score: float
    sum_miss: int
    sum_error: float

@overload
def align(
    melody: MelodyLike,
    performance: PerformanceLike,
    score_func: ScoreWrapper,  # Score wrapper contains candidate_min_score and candidate_min_length
    candidate_min_score: None = None,
    candidate_min_length: None = None,
    same_key: bool = False,
    split_melody: bool = True,
    hop_length: int = 2,
    local_tolerance: float = 0.2,
    miss_tolerance: int = 2,
    verbose: bool = True,
) -> Alignment: ...
@overload
def align(
    melody: MelodyLike,
    performance: PerformanceLike,
    score_func: Callable[['Match'], float],
    candidate_min_score: float, # Supply these two if you use plain function instead of score wrapper
    candidate_min_length: int,
    same_key: bool = False,
    split_melody: bool = True,
    hop_length: int = 2,
    local_tolerance: float = 0.2,
    miss_tolerance: int = 2,
    verbose: bool = True,
) -> Alignment: ...
def align(
    melody: MelodyLike,
    performance: PerformanceLike,
    score_func: Callable[['Match'], float] | ScoreWrapper,
    candidate_min_score: float | None = None,
    candidate_min_length: int | None = None,
    same_key: bool = False,
    split_melody: bool = True,
    hop_length: int = 2,
    local_tolerance: float = 0.2,
    miss_tolerance: int = 2,
    verbose: bool = True,
) -> Alignment:

    if hop_length < 1:
        raise ValueError("hop_length must be at least 1")
    if not isinstance(melody, Melody):
        melody = Melody(melody)
    if not isinstance(performance, Performance):
        performance = Performance(performance)

    if candidate_min_score is None or candidate_min_length is None:
        score_func = cast(ScoreWrapper, score_func) # For type narrowing
        candidate_min_score = score_func.candidate_min_score
        candidate_min_length = score_func.candidate_min_length

    candidates: list[Match] = []

    if split_melody:
        melodies = melody.split()
    else:
        melodies = [melody]
    
    # Calculate total number of scan operations
    total_scans = sum(len(range(0, len(melody) - candidate_min_length, hop_length)) 
                     for melody in melodies if len(melody) >= candidate_min_length)
    
    with tqdm(total=total_scans, desc="Scanning alignments", disable=not verbose) as pbar:
        for melody in melodies:
            for scan_start in range(0, len(melody) - candidate_min_length, hop_length):
                candidates.extend(scan(
                    melody[scan_start:],
                    performance,
                    score_func,
                    same_key,
                    local_tolerance,
                    miss_tolerance,
                    candidate_min_score,
                    enforce_min_score=True))
                pbar.update(1)
    
    if len(candidates) == 0:
        return Alignment([], [], [], 0, 0, 0)

    opt_score, opt_subset = weighted_interval_scheduling(candidates)
    discarded_matches = [match for match in candidates if match not in opt_subset]
    concat_events = concat_matches(opt_subset)
    return Alignment(concat_events, opt_subset, discarded_matches, opt_score,
                    sum(match.sum_miss for match in opt_subset),
                    sum(match.sum_error for match in opt_subset))