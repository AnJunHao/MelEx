from typing import Protocol, Sequence, Literal, Callable, Any, overload, Iterable
import numpy as np
from collections import Counter
import xgboost as xgb
import pandas as pd

from exmel.event import MidiEvent
from exmel.sequence import MelodyLike, Melody
from exmel.io import PathLike

class MatchLike(Protocol):
    @property
    def events(self) -> Sequence[MidiEvent]: ...
    @property
    def sum_miss(self) -> int: ...
    @property
    def sum_error(self) -> float: ...

type SongStatKeys = Literal['duration_per_event', 'note_mean_song', 'velocity_mean']

class ScoreModel(Protocol):

    song_stats: dict[SongStatKeys, float] | None
    penalty: float

    def __init__(self, weights: Any, penalty: float = 0) -> None: ...

    @overload
    def __call__(self, match: Iterable[MatchLike]) -> list[float]: ...
    @overload
    def __call__(self, match: MatchLike) -> float: ...
    def __call__(self, match: MatchLike | Iterable[MatchLike]) -> float | list[float]: ...

    def load_song_stats(self, song_stats: dict[SongStatKeys, float]) -> None:
        self.song_stats = song_stats

class MelodicsModel(ScoreModel):

    def __init__(
        self,
        weights: dict[Literal["error"], float] = {"error": 0.5},
        penalty: float = 0
    ) -> None:
        self.weights = weights
        self.penalty = penalty
        self.song_stats = None

    @overload
    def __call__(self, match: Iterable[MatchLike]) -> list[float]: ...
    @overload
    def __call__(self, match: MatchLike) -> float: ...
    def __call__(self, match: MatchLike | Iterable[MatchLike]) -> float | list[float]:
        if isinstance(match, Iterable):
            return [self(m) for m in match]
        else:
            assert self.song_stats is not None, "Song stats required, use `load_song_stats` first"
            v = sum(event.velocity for event in match.events) / 128
            w1 = (1 - match.sum_miss / len(match.events))
            w2 = sum(event.note for event in match.events) / len(match.events) / self.song_stats['note_mean_song']
            return v*w1*w2 - self.weights['error']*match.sum_error/self.song_stats['duration_per_event']

class RegressionModel(ScoreModel):

    def relative_velocity(self, match: MatchLike) -> float:
        assert self.song_stats is not None, "Song stats required, use `load_song_stats` first"
        return self.velocity(match) / self.song_stats['velocity_mean']

    def relative_note_mean(self, match: MatchLike) -> float:
        assert self.song_stats is not None, "Song stats required, use `load_song_stats` first"
        return self.note_mean(match) / self.song_stats['note_mean_song']

    def relative_duration_per_event(self, match: MatchLike) -> float:
        assert self.song_stats is not None, "Song stats required, use `load_song_stats` first"
        rdpe = self.duration(match) / self.length(match) / self.song_stats['duration_per_event']
        return 1 / rdpe if rdpe < 1 else rdpe

    def normed_note_entropy(self, match: MatchLike) -> float:
        note_unique = self.note_unique(match)
        return self.note_entropy(match) / np.log2(note_unique) if note_unique >= 2 else 0

    def normed_note_change(self, match: MatchLike) -> float:
        return self.note_change(match) / self.length(match)

    def normed_misses(self, match: MatchLike) -> float:
        return self.misses(match) / self.length(match)

    @staticmethod
    def length(match: MatchLike) -> int:
        return len(match.events)

    @staticmethod
    def misses(match: MatchLike) -> int:
        return match.sum_miss

    @staticmethod
    def error(match: MatchLike) -> float:
        return match.sum_error

    @staticmethod
    def velocity(match: MatchLike) -> float:
        return sum(event.velocity for event in match.events) / len(match.events)
    
    @staticmethod
    def duration(match: MatchLike) -> float:
        return match.events[-1].time - match.events[0].time

    @staticmethod
    def note_mean(match: MatchLike) -> float:
        return sum(event.note for event in match.events) / len(match.events)

    @staticmethod
    def note_std(match: MatchLike) -> float:
        return float(np.std([event.note for event in match.events]))

    @staticmethod
    def note_entropy(match: MatchLike) -> float:
        counts = np.array(list(Counter(event.note for event in match.events).values()))
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    @staticmethod
    def note_unique(match: MatchLike) -> int:
        return len(set(event.note for event in match.events))

    @staticmethod
    def note_change(match: MatchLike) -> int:
        return sum(match.events[i].note != match.events[i - 1].note
                   for i in range(1, len(match.events)))

type WeightKeys = Literal[
    'intercept',
    'length', 'misses', 'error', 'velocity', 'duration',
    'note_mean', 'note_std', 'note_entropy', 'note_unique', 'note_change',
    'relative_velocity', 'relative_note_mean', 'relative_duration_per_event',
    'normed_note_entropy', 'normed_note_change', 'normed_misses']

class LinearModel(RegressionModel):

    def __init__(self, weights: dict[WeightKeys, float], penalty: float = 0):
        assert 'intercept' in weights, "Intercept is required in weights"
        self.weights = weights
        self.penalty = penalty
        self.song_stats = None
        self.func_map: dict[WeightKeys, Callable[[MatchLike], float]] = {
            'intercept': lambda _: 1,
            'length': self.length,
            'misses': self.misses,
            'error': self.error,
            'velocity': self.velocity,
            'duration': self.duration,
            'note_mean': self.note_mean,
            'note_std': self.note_std,
            'note_entropy': self.note_entropy,
            'note_unique': self.note_unique,
            'note_change': self.note_change,
            'relative_velocity': self.relative_velocity,
            'relative_note_mean': self.relative_note_mean,
            'relative_duration_per_event': self.relative_duration_per_event,
            'normed_note_entropy': self.normed_note_entropy,
            'normed_note_change': self.normed_note_change,
            'normed_misses': self.normed_misses,
        }

    @overload
    def __call__(self, match: MatchLike) -> float: ...
    @overload
    def __call__(self, match: Iterable[MatchLike]) -> list[float]: ...
    def __call__(self, match: MatchLike | Iterable[MatchLike]) -> float | list[float]:
        # Intercept is added to the sum (1 * intercept)
        if isinstance(match, Iterable):
            # Penalty included in self(m)
            return [self(m) for m in match]
        else:
            return sum(self.func_map[key](match) * self.weights[key]
                    for key in self.weights) - self.penalty

class XGBoostModel(RegressionModel):
    
    def __init__(self, weights: PathLike, penalty: float = 0) -> None:
        self.model = xgb.XGBRegressor()
        self.model.load_model(weights)
        self.penalty = penalty

    def get_data(self, match: MatchLike) -> dict[WeightKeys, float]:
        return {
            'length': self.length(match),
            'misses': self.misses(match),
            'error': self.error(match),
            'velocity': self.velocity(match),
            'duration': self.duration(match),
            'note_mean': self.note_mean(match),
            'note_std': self.note_std(match),
            'note_entropy': self.note_entropy(match),
            'note_unique': self.note_unique(match),
            'note_change': self.note_change(match),
            'relative_velocity': self.relative_velocity(match),
            'relative_note_mean': self.relative_note_mean(match),
            'relative_duration_per_event': self.relative_duration_per_event(match),
            'normed_note_entropy': self.normed_note_entropy(match),
            'normed_note_change': self.normed_note_change(match),
            'normed_misses': self.normed_misses(match),
        }

    @overload
    def __call__(self, match: MatchLike) -> float: ...
    @overload
    def __call__(self, match: Iterable[MatchLike]) -> list[float]: ...
    def __call__(self, match: MatchLike | Iterable[MatchLike]) -> float | list[float]:
        if isinstance(match, Iterable):
            data = [self.get_data(m) for m in match]
            df = pd.DataFrame(data)
            return [float(score) - self.penalty for score in self.model.predict(df)]
        else:
            data = self.get_data(match)
            return float(self.model.predict(pd.DataFrame([data]))[0]) - self.penalty

def get_linear_model_default_weights() -> dict[WeightKeys, float]:
    return {
        'intercept': -23.120287,
        'relative_note_mean': 11.182220,
        'misses': -0.261118,
        'error': -1.348220,
        'velocity': 0.117107,
        'length': 1.006953,
    }

def tp_fp(
    match: MatchLike,
    gt: MelodyLike,
    tolerance: float = 0.1,
) -> tuple[int, int]:
    if not isinstance(gt, Melody):
        gt = Melody(gt)
    gt %= 12
    match_melody = Melody(match.events)
    match_melody %= 12
    tp = 0
    fp = 0
    for event in match_melody:
        nearest = gt.nearest(event)
        if nearest is not None and abs(nearest.time - event.time) < tolerance:
            tp += 1
        else:
            fp += 1
    return tp, fp