from dataclasses import dataclass, field
from typing import Callable, overload, Literal, Iterable, Iterator, TypedDict
from tqdm.auto import tqdm
import warnings
from icecream import ic

from melign.data.sequence import Melody, MelodyLike, Performance, PerformanceLike, song_stats
from melign.data.event import MidiEvent
from melign.align.dp import weighted_interval_scheduling, non_crossing_weighted_bi_interval_scheduling
from melign.align.score import ScoreModel, MelodicsModel, StructuralMapping

@dataclass
class Match:
    events: list[MidiEvent]
    speed: float
    tail_miss: int
    sum_miss: int
    sum_error: float
    sum_shadow: int
    sum_between: int
    sum_above_between: int
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

    def freeze(self, defer_score=False) -> 'FrozenMatch':
        if defer_score:
            score = float("-inf")
        else:
            score = self.score

        frozen_match = FrozenMatch(
            events=tuple(self.events),
            speed=self.speed,
            sum_miss=self.sum_miss,
            sum_error=self.sum_error,
            sum_shadow=self.sum_shadow,
            sum_between=self.sum_between,
            sum_above_between=self.sum_above_between,
            melody_start=self.melody_start,
            melody_end=self.melody_end,
            start=self.events[0].time,
            end=self.events[-1].time,
            score=score)
        
        return frozen_match

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
            sum_shadow=self.sum_shadow,
            sum_between=self.sum_between,
            sum_above_between=self.sum_above_between,
            score_func=self.score_func,
            melody_start=self.melody_start,
            melody_end=self.melody_end
        )

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[MidiEvent]:
        return iter(self.events)

    @overload
    def __getitem__(self, key: int) -> MidiEvent: ...
    @overload
    def __getitem__(self, key: slice) -> list[MidiEvent]: ...
    def __getitem__(self, key: int | slice) -> MidiEvent | list[MidiEvent]:
        return self.events[key]

@dataclass(frozen=True, slots=True)
class FrozenMatch:
    events: tuple[MidiEvent, ...]
    speed: float
    sum_miss: int
    sum_error: float
    sum_shadow: int
    sum_between: int
    sum_above_between: int
    melody_start: float
    melody_end: float
    start: float = field(hash=True)
    end: float = field(hash=True)
    score: float = field(hash=True)

    def update_score(self, score: float) -> 'FrozenMatch':
        return FrozenMatch(
            events=self.events,
            speed=self.speed,
            sum_miss=self.sum_miss,
            sum_error=self.sum_error,
            sum_shadow=self.sum_shadow,
            sum_between=self.sum_between,
            sum_above_between=self.sum_above_between,
            melody_start=self.melody_start,
            melody_end=self.melody_end,
            start=self.start,
            end=self.end,
            score=score
        )

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[MidiEvent]:
        return iter(self.events)

    @overload
    def __getitem__(self, key: int) -> MidiEvent: ...
    @overload
    def __getitem__(self, key: slice) -> tuple[MidiEvent, ...]: ...
    def __getitem__(self, key: int | slice) -> MidiEvent | tuple[MidiEvent, ...]:
        return self.events[key]

type MatchLike = Match | FrozenMatch

def concat_matches(matches: Iterable[MatchLike]) -> list[MidiEvent]:
    return [event
            for match in matches
            for event in match.events]

def filter_within[T: MidiEvent | MatchLike](
    queries: Iterable[T],
    within: Iterable[MatchLike]
    ) -> list[T]:
    output = []
    for q in queries:
        for m in within:
            if isinstance(q, MidiEvent) and m.start <= q.time <= m.end:
                output.append(q)
                break
            elif isinstance(q, (FrozenMatch, Match)) and m.start <= q.start and q.end <= m.end:
                output.append(q)
                break
    return output

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
    candidate_min_score: float | None,
    candidate_min_length: int,
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
    candidate_min_score: float | None   ,
    candidate_min_length: int,
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
    candidate_min_score: float | None,
    candidate_min_length: int,
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
                            speed_prior, 0, 0, 0.0, 0, 0, 0,
                            score_func,
                            melody[0].time, melody[0].time)
                        for tv in tv_list]
        else:
            lives =  [Match([event],
                            speed_prior, 0, 0, 0.0, 0, 0, 0,
                            score_func,
                            melody[0].time, melody[0].time)
                        for event in performance.global_events]
        return lives, []

    #################### Recursive Call ####################
    lives, frozens = scan(
        melody[:-1], performance, 
        score_func, same_key, same_speed, speed_prior, variable_tail,
        local_rel_tolerance, local_abs_tolerance,
        miss_tolerance, candidate_min_score, candidate_min_length,
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

                #################### Update Shadow / Between ####################
                s, b, ab = s_b_ab(candidate.events, performance)
                candidate.sum_shadow += s
                candidate.sum_between += b
                candidate.sum_above_between += ab
            
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
                (candidate_min_score is None or candidate.score >= candidate_min_score) and
                len(candidate) >= candidate_min_length):
                copy = candidate.copy()
                frozens.append(copy.freeze(defer_score=candidate_min_score is None))
        elif (not variable_tail and
              (candidate_min_score is None or candidate.score >= candidate_min_score) and
              len(candidate) >= candidate_min_length):
            # good enough candidates are already added to frozen if variable_tail
            # so we only need to freeze when variable_tail is False
            frozens.append(candidate.freeze(defer_score=candidate_min_score is None))

    if not recursive_call:
        if variable_tail:
            return frozens
        else:
            frozens.extend(
                c.freeze(defer_score=candidate_min_score is None)
                for c in new_lives
                if len(c) >= candidate_min_length and
                   (candidate_min_score is None or c.score >= candidate_min_score))
            return frozens

    return new_lives, frozens

@dataclass(slots=True)
class AlignConfig:
    score_model: ScoreModel = field(default_factory=MelodicsModel)
    same_key: bool = False
    same_speed: bool = True
    speed_prior: float = 1.0
    variable_tail: bool = True
    local_tolerance: float = 0.5
    miss_tolerance: int = 2
    candidate_min_score: float = 4
    candidate_min_length: int = 5
    hop_length: int = 1
    split_melody: bool = True
    structural_align: bool = True
    structural_max_difference: float = 1
    melody_min_recurrence: float = 0.975
    duration_tolerance: float = 0.5
    structural_only: bool = False

@dataclass(frozen=True, slots=True)
class Alignment:
    events: list[MidiEvent]
    matches: list[FrozenMatch]
    score: float
    discarded_matches: list[FrozenMatch] | None = None

    def __repr__(self) -> str:
        if self.discarded_matches is None:  
            return f"Alignment(events=[...({len(self.events)} events)...], " \
                    f"matches=[...({len(self.matches)} matches)...], " \
                    f"score={self.score}, "
        else:
            return f"Alignment(events=[...({len(self.events)} events)...], " \
                    f"matches=[...({len(self.matches)} matches)...], " \
                    f"score={self.score}, " \
                    f"discarded_matches=[...({len(self.discarded_matches)} discarded matches)...])"

    def __len__(self) -> int:
        return len(self.events)

    @overload
    def __getitem__(self, key: int) -> MidiEvent: ...
    @overload
    def __getitem__(self, key: slice) -> list[MidiEvent]: ...
    def __getitem__(self, key: int | slice) -> MidiEvent | list[MidiEvent]:
        return self.events[key]

    def __iter__(self) -> Iterator[MidiEvent]:
        return iter(self.events)

@overload
def align(  # type: ignore[reportOverlappingOverload]
    melody: MelodyLike,
    performance: PerformanceLike,
    config: AlignConfig,
    defer_score: bool = True,
    verbose: bool = True,
    skip_wisp: Literal[False] = False,
    return_discarded_matches: bool = False
) -> Alignment: ...
@overload
def align(
    melody: MelodyLike,
    performance: PerformanceLike,
    config: AlignConfig,
    defer_score: bool = True,
    verbose: bool = True,
    skip_wisp: Literal[True] = True,
    return_discarded_matches: bool = False
) -> list[FrozenMatch]: ...
def align(
    melody: MelodyLike,
    performance: PerformanceLike,
    config: AlignConfig,
    defer_score: bool = True,
    verbose: bool = True,
    skip_wisp: Literal[False] | Literal[True] = False,
    return_discarded_matches: bool = False
) -> Alignment | list[FrozenMatch]:

    if config.hop_length < 1:
        raise ValueError("hop_length must be at least 1")
    if not isinstance(melody, Melody):
        melody = Melody(melody)
    if not isinstance(performance, Performance):
        performance = Performance(performance)

    candidates: list[FrozenMatch] = []

    config.score_model.load_song_stats(song_stats(melody, performance))

    if config.split_melody:
        melodies = melody.split()
    else:
        melodies = [melody]

    local_abs_tolerance = melody.mean_diff() * config.local_tolerance

    total_scans = sum(len(range(0,
                                len(m) - config.candidate_min_length,
                                config.hop_length)) 
                     for m in melodies if len(m) >= config.candidate_min_length)
    
    with tqdm(total=total_scans, desc="Scanning alignments", disable=not verbose) as pbar:
        for m in melodies:
            for scan_start in range(0,
                                    len(m) - config.candidate_min_length,
                                    config.hop_length):
                candidates.extend(scan(
                    m[scan_start:],
                    performance,
                    config.score_model,
                    config.same_key,
                    config.same_speed,
                    config.speed_prior,
                    config.variable_tail,
                    config.local_tolerance,
                    local_abs_tolerance,
                    config.miss_tolerance,
                    None if defer_score else config.candidate_min_score,
                    config.candidate_min_length,
                    recursive_call=False))
                pbar.update(1)
    
    if len(candidates) == 0:
        if return_discarded_matches:
            return Alignment(
                events=[], matches=[],
                score=0,
                discarded_matches=[])
        else:
            return Alignment(
                events=[], matches=[],
                score=0,
                discarded_matches=None)


    if (config.structural_align and
        abs(melody.duration - performance.duration) / melody.duration < config.duration_tolerance):
        if not defer_score:
            warnings.warn("defer_score should be True when structural_align is True", stacklevel=2)
        max_match = max(candidates, key=lambda x: len(x))
        structural_mapping = StructuralMapping(max_match, config.structural_max_difference)
        structural_candidates = [
            candidate.update_score(structural_mapping(candidate, melody.duration))
            for candidate in candidates
        ]
        structural_score, structural_subset = non_crossing_weighted_bi_interval_scheduling(
            structural_candidates, return_subset=True, verbose=False)
        recurrence = sum(m.sum_miss + len(m) for m in structural_subset) / len(melody)
        if recurrence < config.melody_min_recurrence:
            structural_subset = None
        elif skip_wisp:
            output = filter_within(candidates, structural_subset) # Assign to output just for type inspection
            return output
        elif config.structural_only:
            concat_events = concat_matches(structural_subset)
            return Alignment(concat_events, structural_subset, structural_score, None)
    else:
        structural_subset = None

    if skip_wisp:
        return candidates
        
    if defer_score:
        scores = config.score_model(candidates)
        updated_candidates: list[FrozenMatch] = []
        for candidate, score in zip(candidates, scores):
            if score >= config.candidate_min_score:
                updated_candidates.append(candidate.update_score(score))
        candidates = updated_candidates

    opt_score, opt_subset = weighted_interval_scheduling(
        candidates, return_subset=True, verbose=False)

    if return_discarded_matches:
        discarded_matches = [match for match in candidates if match not in opt_subset]
    else:
        discarded_matches = None

    concat_events = concat_matches(opt_subset)

    if structural_subset is not None:
        concat_events = filter_within(concat_events, structural_subset)
        
    return Alignment(concat_events, opt_subset, opt_score,
                    discarded_matches)

def s_b_ab(events: list[MidiEvent], performance: Performance) -> tuple[int, int, int]:
    """
    Calculate the number of shadow, between, and above-between events,
    for the second last event in the list.
    """
    if len(events) <= 1:
        warnings.warn("Should never happen: s_b_ab must be called with at least 2 events", stacklevel=2)
        return 0, 0, 0

    #################### Shadow: Nearby and above event ####################
    potential_shadows = performance.global_between(events[-2].time-0.1, events[-2].time+0.1)
    s_ = 0
    for s in potential_shadows:
        if s in events[-3:]:
            continue
        if s.note > events[-2].note:
            s_ = 1
            break

    #################### Between: Between the last two events ####################
    #################### Between Above: Between and above the last two events ####################
    potential_between = performance.global_between(events[-2].time+0.1, events[-1].time-0.1)
    b_ = 0
    ab_ = 0
    for b in potential_between:
        if (events[-2].note <= b.note and b.note <= events[-1].note) or \
            (events[-1].note <= b.note and b.note <= events[-2].note):
            b_ = 1
        elif (b.note > events[-1].note and b.note > events[-2].note):
            ab_ = 1
        if b_ == 1 and ab_ == 1:
            break
    return s_, b_, ab_

class SelfEvalResult(TypedDict):
    pred_f1: float
    pred_precision: float
    pred_recall: float
    above_between: float
    between: float
    error: float
    miss: float
    shadow: float

def self_eval(alignment: Alignment) -> SelfEvalResult:
    above_between = sum(match.sum_above_between for match in alignment.matches) / len(alignment.events)
    between = sum(match.sum_between for match in alignment.matches) / len(alignment.events)
    error = sum(match.sum_error for match in alignment.matches) / len(alignment.events)
    miss = sum(match.sum_miss for match in alignment.matches) / len(alignment.events)
    shadow = sum(match.sum_shadow for match in alignment.matches) / len(alignment.events)
    pred_f1 = 1.0052516471187187 \
         - 0.5237980085343735 * between \
         - 0.5730591616288301 * miss \
         - 0.5464726773608072 * error \
         - 0.752308438803511 * shadow
    pred_precision = 0.9986701254937118 \
         - 0.12164280750603358 * miss \
         - 0.3349362405481795 * error \
         - 0.5340878865537385 * shadow \
         - 0.766615119773141 * above_between
    pred_recall = 1.00715405271537 \
         - 0.6772013197828821 * between \
         - 1.0475273327909769 * miss \
         - 0.7285961336990973 * error \
         - 0.8895154044302019 * shadow

    return SelfEvalResult(
        pred_f1=pred_f1,
        pred_precision=pred_precision,
        pred_recall=pred_recall,
        above_between=above_between,
        between=between,
        error=error,
        miss=miss,
        shadow=shadow
    )