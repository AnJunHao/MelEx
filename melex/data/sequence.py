from pathlib import Path
import bisect
from typing import Literal, Iterator, overload, Iterable
from collections import defaultdict
import warnings
import statistics

from melex.data.event import MelEvent, MidiEvent, EventLike
from melex.data.io import (
    load_midi, load_note, PathLike,
    melody_to_midi, extract_original_events, melody_to_note
)

class Melody:

    events: list[MelEvent]

    def __init__(self,
                source: "PathLike | Melody | Iterable[EventLike]",
                track_idx: int | None = None) -> None:
        """
        Initialize the melody with a MIDI file.
        """
        if isinstance(source, (Path, str)):
            source = Path(source)
            if source.suffix in (".mid", ".midi"):
                self.events = load_midi(source, track_idx, include_velocity=False)
            elif source.suffix in (".note", ".txt"):
                self.events = load_note(source)
            else:
                raise ValueError(f"Unsupported file type: {source.suffix}")
        elif isinstance(source, Melody):
            self.events = source.events
        elif isinstance(source, Iterable):
            self.events = [MelEvent(event.time, event.note) for event in source]
        else:
            raise TypeError(f"Invalid source type: {type(source)}")

    def _build_indices(self) -> None:
        self.events_by_note: dict[int, list[MelEvent]] = {}
        for event in self.events:
            if event.note not in self.events_by_note:
                self.events_by_note[event.note] = []
            self.events_by_note[event.note].append(event)
        for note in self.events_by_note:
            self.events_by_note[note].sort(key=lambda x: x.time)

    def __xor__(self, note_shift: int) -> "Melody":
        """
        Shift the melody by a number of semitones.
        """
        if isinstance(note_shift, int):
            return Melody([event ^ note_shift for event in self.events])
        return NotImplemented
    
    def __lshift__(self, time_shift: float) -> "Melody":
        """
        Shift the melody by a number of seconds.
        """
        if isinstance(time_shift, (float, int)):
            return Melody([event << time_shift for event in self.events])
        return NotImplemented
    
    def __rshift__(self, time_shift: float) -> "Melody":
        """
        Shift the melody by a number of seconds.
        """
        if isinstance(time_shift, (float, int)):
            return Melody([event >> time_shift for event in self.events])
        return NotImplemented

    def __floordiv__(self, other: float) -> "Melody":
        """
        Quantize the melody to the nearest multiple of other.
        """
        if isinstance(other, (float, int)):
            return Melody([event // other for event in self.events])
        return NotImplemented

    def __mod__(self, other: Literal[12]) -> "Melody":
        """
        Apply note modulo operation to all events (typically used for octave reduction).
        """
        if other == 12:
            return Melody([event % 12 for event in self.events])
        elif isinstance(other, int):
            warnings.warn(f"Modulo by {other} does not make sense. You may want to use 12 instead.")
            return Melody([event % other for event in self.events])
        else:
            return NotImplemented

    def __matmul__(self, other: EventLike | float) -> MelEvent | None:
        """
        Find the nearest event to the given event.
        """
        if isinstance(other, (float, int)):
            return self.nearest_global(other)
        return self.nearest(other)
    
    def __rmatmul__(self, other: EventLike | float) -> MelEvent | None:
        """
        Find the nearest event to the given event.
        """
        if isinstance(other, (float, int)):
            return self.nearest_global(other)
        return self.nearest(other)

    def __repr__(self) -> str:
        if len(self.events) <= 50:
            # New line for each event
            return f"Melody([{',\n'.join([event.__repr__() for event in self.events])}])"
        else:
            return f"Melody([{self.events[0]}, ...({len(self.events) - 2} events)..., {self.events[-1]}])"

    def __len__(self) -> int:
        return len(self.events)

    @overload
    def __getitem__(self, key: int) -> MelEvent: ...
    @overload
    def __getitem__(self, key: slice) -> "Melody": ...
    @overload
    def __getitem__(self, key: tuple[float | None, float | None]) -> "Melody": ...
    def __getitem__(self, key: int | slice | tuple[float | None, float | None]) -> "MelEvent | Melody":
        if isinstance(key, slice):
            return Melody(self.events[key])
        elif isinstance(key, tuple):
            assert len(key) == 2, "Tuple for time slicing must contain exactly two elements"
            return self.time_slice(key[0], key[1])
        else:
            return self.events[key]

    def __iter__(self) -> Iterator[MelEvent]:
        return iter(self.events)

    def time_slice(self, start: float | None = None, stop: float | None = None) -> "Melody":
        """
        Slice the melody by time.
        """
        if start is None:
            start_idx = None
        else:
            event = self @ start
            assert event is not None, "Cannot perform float slicing on empty melody"
            start_idx = self.events.index(event)
            if event.time < start:
                start_idx += 1
        
        if stop is None:
            stop_idx = None
        else:
            event = self @ stop
            assert event is not None, "Cannot perform float slicing on empty melody"
            stop_idx = self.events.index(event)
            if event.time > stop:
                pass
            else:
                stop_idx += 1
            if stop_idx >= len(self.events):
                stop_idx = None
        
        return Melody(self.events[start_idx:stop_idx])

    def nearest(self, reference: EventLike) -> MelEvent | None:
        """
        Find the nearest event to the given time.
        When given an EventLike, finds the nearest event with the same note.
        """
        # Build indices if not already built
        if not hasattr(self, 'events_by_note'):
            self._build_indices()
        
        # Find the nearest event with the same note using indices
        note = reference.note
        ref_time = reference.time
        
        if note not in self.events_by_note:
            # If no events with this note, fall back to nearest by time only
            return None
        
        # Use binary search on the sorted events for this note
        note_events = self.events_by_note[note]
        idx = bisect.bisect_left(note_events, ref_time, key=lambda x: x.time)
        
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(note_events):
            candidates.append(idx)
        
        # Find the nearest by absolute time difference
        return note_events[min(candidates, key=lambda i: abs(note_events[i].time - ref_time))]

    def nearest_global(self, time: float) -> MelEvent | None:
        if len(self.events) == 0:
            return None
        idx = bisect.bisect_left(self.events, time, key=lambda x: x.time)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(self.events):
            candidates.append(idx)
        return self.events[min(candidates, key=lambda x: abs(self.events[x].time - time))]

    @property
    def duration(self) -> float:
        return self.events[-1].time - self.events[0].time

    def to_midi(self, path: PathLike, original_midi: PathLike | None = None) -> None:
        if original_midi is not None:
            extract_original_events(self.events, original_midi, path)
        else:
            melody_to_midi(self.events, path)

    def to_note(self, path: PathLike) -> None:
        melody_to_note(self.events, path)

    def plot(self, save_path: PathLike | None = None, show_plot: bool = True, show_splits: bool = False, split_threshold: float = 16) -> None:
        from melex.align.eval_and_vis import plot_melody
        plot_melody(self, save_path, show_plot, show_splits, split_threshold)

    def diff(self) -> list[float]:
        return [self[i+1].time - self[i].time for i in range(len(self)-1)]

    def mean_diff(self) -> float:
        diffs = self.diff()
        diffs = filter(lambda x: x > 0, diffs)
        return statistics.geometric_mean(diffs)

    def split(self, threshold: float = 16) -> list["Melody"]:
        """
        Separate the melody into multiple melodies based on the threshold.
        """
        threshold = self.mean_diff() * threshold
        indices = [i+1 for i, d in enumerate(self.diff()) if d > threshold]
        indices = [0] + indices + [len(self)]
        return [Melody(self.events[indices[i]:indices[i+1]]) for i in range(len(indices)-1)]

type time_velocity_tuple = tuple[float, int]
type MelodyLike = Melody | Iterable[EventLike] | PathLike

class Performance:
    """
    Optimized MIDI event tracker that uses precomputed indices and lookup tables
    to support efficient nearest neighbor queries on note events.
    """
    
    def __init__(self, source: 'PathLike | Performance | Iterable[MidiEvent]') -> None:
        """
        Initialize the tracker by loading MIDI data and building search indices.
        
        Time Complexity: O(n log n) where n is the number of note events
        """
        
        self.events_by_note: dict[int, list[time_velocity_tuple]] = {}
        if isinstance(source, (Path, str)):
            self._load_midi(source)
            self._build_indices()
        elif isinstance(source, Performance):
            self.events_by_note = source.events_by_note
            self.global_events = source.global_events
            self.global_times = source.global_times
            self.note_times = source.note_times
            self.note_velocities = source.note_velocities
        elif isinstance(source, Iterable):
            self.events_by_note = defaultdict(list)
            for event in source:
                self.events_by_note[event.note].append((event.time, event.velocity))
            self.events_by_note = dict(self.events_by_note)
            self._build_indices()
        else:
            raise TypeError(f"Invalid source type: {type(source)}")
    
    def _load_midi(self, midi_file: PathLike) -> None:
        """
        Load MIDI file, convert ticks to seconds, and organize events by note.
        
        Time Complexity: O(n log n) where n is the number of note events
        """

        all_events = load_midi(midi_file)
        
        # Organize events by note
        for event in all_events:
            if event.note not in self.events_by_note:
                self.events_by_note[event.note] = []
            self.events_by_note[event.note].append((event.time, event.velocity))
        
        # Sort each note's events by time
        for note in self.events_by_note:
            self.events_by_note[note].sort(key=lambda x: x[0])
    
    def _build_indices(self) -> None:
        """
        Build precomputed data structures to accelerate query performance.
        
        Time Complexity: O(n log n) where n is the number of note events
        """
        # 1. Convert to arrays
        self.note_times: dict[int, list[float]] = {}
        self.note_velocities: dict[int, list[int]] = {}
        
        for note, events in self.events_by_note.items():
            times = [event[0] for event in events]
            velocities = [event[1] for event in events]
            self.note_times[note] = times
            self.note_velocities[note] = velocities
        
        # 2. Build global time-sorted index for cross-note queries
        self.global_events: list[MidiEvent] = []  # (time, note, velocity)
        for note, events in self.events_by_note.items():
            for time, velocity in events:
                self.global_events.append(MidiEvent(time, note, velocity))
        
        self.global_events.sort(key=lambda x: x.time)  # Sort by time
        self.global_times = [event.time for event in self.global_events]

        # Build gloabl time-to-index mapping for fast exact queries
        self.global_time_to_index = {time: i for i, time in enumerate(self.global_times)}

    @property
    def duration(self) -> float:
        """
        Calculate the duration of the MIDI file.
        Returns the time of the last event in seconds.
        
        Time Complexity: O(1) - cached property
        """
        if not self.global_events:
            return 0.0
        return self.global_events[-1].time

    def nearest(self, mel_event: EventLike, start: float | None = None) -> MidiEvent | None:
        """
        Find the nearest occurrence of the note to the given event.
        
        Args:
            mel_event: The reference event to find nearest neighbor for
            start: If provided, only consider events strictly after this time
        
        Time Complexity:
            O(log k) where k is the number of events for the note
        """
        note = mel_event.note
        ref_time = mel_event.time
        
        if note not in self.note_times:
            return None
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        if not times:
            return None
        
        # Find the minimum valid index based on start constraint
        if start is not None:
            min_idx = bisect.bisect_right(times, start)
            if min_idx >= len(times):
                return None  # No events strictly after start time
        else:
            min_idx = 0
        
        # Binary search for the reference time
        idx = bisect.bisect_left(times, ref_time)
        
        # Check neighbors to find the nearest, but only at or after min_idx
        candidates = []
        if idx > min_idx:
            candidates.append(idx - 1)
        if idx < len(times) and idx >= min_idx:
            candidates.append(idx)
        
        if not candidates:
            return None
        
        # Find the nearest by absolute time difference
        nearest_idx = min(candidates, key=lambda i: abs(times[i] - ref_time))
        return MidiEvent(times[nearest_idx], note, velocities[nearest_idx])
    
    def nearest_multi(self, mel_event: EventLike, n: int) -> list[MidiEvent]:
        """
        Find n nearest occurrences of the note to the given event.
        
        Args:
            mel_event: The reference event to find nearest neighbors for
            n: Number of nearest events to return
        
        Time Complexity: 
            - O(log k + n) where k is the number of events for the note
        """
        note = mel_event.note
        ref_time = mel_event.time
        
        if note not in self.note_times or n <= 0:
            return []
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        if not times:
            return []
        
        idx = bisect.bisect_left(times, ref_time)
        
        # Use two-pointer expansion starting from the determined index
        left = idx - 1
        right = idx
        results = []
        
        # Special case: if we started from a bucket, check if the bucket event itself is closest
        if idx < len(times) and (left < 0 or abs(times[idx] - ref_time) <= abs(times[left] - ref_time)):
            results.append(MidiEvent(times[idx], note, velocities[idx]))
            right = idx + 1
        elif left >= 0:
            results.append(MidiEvent(times[left], note, velocities[left]))
            left -= 1
        
        # Collect remaining n-1 nearest events by expanding left and right
        while len(results) < n and (left >= 0 or right < len(times)):
            left_dist = float('inf')
            right_dist = float('inf')
            
            # Calculate distances
            if left >= 0:
                left_dist = abs(times[left] - ref_time)
            if right < len(times):
                right_dist = abs(times[right] - ref_time)
            
            # Choose the closer event
            if left_dist <= right_dist and left >= 0:
                results.append(MidiEvent(times[left], note, velocities[left]))
                left -= 1
            elif right < len(times):
                results.append(MidiEvent(times[right], note, velocities[right]))
                right += 1
            else:
                break
        
        # Sort by distance for consistent ordering
        results.sort(key=lambda event: abs(event.time - ref_time))
        
        return results

    def nearest_global(self, ref_time: float) -> MidiEvent | None:
        """
        Find the nearest event across all notes to the given time.
        
        Time Complexity: O(log N) where N is the total number of events
        """
        if len(self.global_events) == 0:
            return None
        
        # Binary search on global time array
        idx = bisect.bisect_left(self.global_times, ref_time)
        
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(self.global_events):
            candidates.append(idx)
        
        # Find the nearest by absolute time difference
        nearest_idx = min(candidates, key=lambda i: abs(self.global_events[i].time - ref_time))
        return self.global_events[nearest_idx]

    def plot(self, save_path: PathLike | None = None, show_plot: bool = True) -> None:
        from melex.align.eval_and_vis import plot_performance
        plot_performance(self, save_path, show_plot)

    def between(self, from_: EventLike, to_: EventLike) -> list[MidiEvent]:
        """
        Extract all events for a specific note within a given time range.
        
        Args:
            note: The MIDI note number to search for
            start_time: The start time of the range (inclusive)
            end_time: The end time of the range (inclusive)
        
        Returns:
            List of MidiEvent objects for the specified note within the time range,
            sorted by time.
        
        Time Complexity: O(log k + m) where k is the number of events for the note
                         and m is the number of events in the range
        """
        note = from_.note
        assert note == to_.note
        start_time, end_time = from_.time, to_.time
        
        if note not in self.note_times:
            return []
        
        if start_time > end_time:
            return []
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        if not times:
            return []
        
        # Use binary search to find the start and end indices
        start_idx = bisect.bisect_left(times, start_time)
        end_idx = bisect.bisect_right(times, end_time)
        
        # Extract events in the range
        results = []
        for i in range(start_idx, end_idx):
            results.append(MidiEvent(times[i], note, velocities[i]))
        
        return results

    def global_between(self, from_: float, to_: float) -> list[MidiEvent]:
        """
        Extract all events within a given time range (exclusive of boundaries).
        """
        # Time-based search on global events
        start_time, end_time = from_, to_
        
        if start_time > end_time:
            return []
        
        if not self.global_times:
            return []
        
        # Use binary search to find the start and end indices in global events
        start_idx = bisect.bisect_left(self.global_times, start_time)
        end_idx = bisect.bisect_right(self.global_times, end_time)
        
        # Return all events in the time range
        return self.global_events[start_idx:end_idx]

    def exact_global_between(self, from_: float, to_: float) -> list[MidiEvent]:
        """
        Extract all events for a specific note within a given time range (exclusive of boundaries).
        The "from_" and "to_" must exist in the global_times array (e.g. in the performance)
        """
        start_idx = self.global_time_to_index[from_]
        end_idx = self.global_time_to_index[to_]
        if start_idx >= end_idx - 1:
            return []
        return self.global_events[start_idx+1:end_idx]

    def __xor__(self, note_shift: int) -> "Performance":
        """
        Shift all notes in the performance by a given number of semitones.
        """
        if isinstance(note_shift, int):
            shifted_events = [e ^ note_shift for e in self.global_events]
            return Performance(shifted_events)
        return NotImplemented
    
    def __rshift__(self, time_shift: float) -> "Performance":
        """
        Shift the performance by a number of seconds (forward in time).
        """
        if isinstance(time_shift, (float, int)):
            shifted_events = [event >> time_shift for event in self.global_events]
            return Performance(shifted_events)
        return NotImplemented
    
    def __lshift__(self, time_shift: float) -> "Performance":
        """
        Shift the performance by a number of seconds (backward in time).
        """
        if isinstance(time_shift, (float, int)):
            shifted_events = [event << time_shift for event in self.global_events]
            return Performance(shifted_events)
        return NotImplemented
    
    def __mul__(self, time_scale: float) -> "Performance":
        """
        Scale the performance by a time factor.
        """
        if isinstance(time_scale, (float, int)):
            scaled_events = [event * time_scale for event in self.global_events]
            return Performance(scaled_events)
        return NotImplemented
    
    def __truediv__(self, time_scale: float) -> "Performance":
        """
        Scale the performance by dividing time by a factor.
        """
        if isinstance(time_scale, (float, int)):
            scaled_events = [event / time_scale for event in self.global_events]
            return Performance(scaled_events)
        return NotImplemented
    
    def __floordiv__(self, quantize_step: float) -> "Performance":
        """
        Quantize the performance to the nearest multiple of quantize_step.
        """
        if isinstance(quantize_step, (float, int)):
            quantized_events = [event // quantize_step for event in self.global_events]
            return Performance(quantized_events)
        return NotImplemented
    
    def __mod__(self, other: Literal[12]) -> "Performance":
        """
        Apply note modulo operation to all events (typically used for octave reduction).
        """
        if other == 12:
            modulo_events = [event % 12 for event in self.global_events]
            return Performance(modulo_events)
        elif isinstance(other, int):
            warnings.warn(f"Modulo by {other} does not make sense. You may want to use 12 instead.")
            modulo_events = [event % other for event in self.global_events]
            return Performance(modulo_events)
        else:
            return NotImplemented

    def __matmul__(self, other: 'EventLike | float') -> MidiEvent | None:
        """
        Find the nearest event to the given event or time.
        If other is a float or int, find the nearest global event to that time.
        If other is EventLike, find the nearest event with the same note.
        """
        if isinstance(other, (float, int)):
            return self.nearest_global(other)
        return self.nearest(other)

    def __rmatmul__(self, other: EventLike | float) -> MidiEvent | None:
        """
        Find the nearest event to the given event or time.
        If other is a float or int, find the nearest global event to that time.
        If other is EventLike, find the nearest event with the same note.
        """
        if isinstance(other, (float, int)):
            return self.nearest_global(other)
        return self.nearest(other)

    def __len__(self) -> int:
        return len(self.global_events)

    @overload
    def __getitem__(self, key: int) -> MidiEvent: ...
    @overload
    def __getitem__(self, key: slice) -> "Performance": ...
    @overload
    def __getitem__(self, key: tuple[float | None, float | None]) -> "Performance": ...
    def __getitem__(self, key: int | slice | tuple[float | None, float | None]) -> "MidiEvent | Performance":
        if isinstance(key, slice):
            return Performance(self.global_events[key])
        elif isinstance(key, tuple):
            assert len(key) == 2, "Tuple for time slicing must contain exactly two elements"
            return self.time_slice(key[0], key[1])
        else:
            return self.global_events[key]

    def __iter__(self) -> Iterator[MidiEvent]:
        return iter(self.global_events)

    def time_slice(self, start: float | None = None, stop: float | None = None) -> "Performance":
        """
        Slice the performance by time.
        """
        if start is None:
            start_idx = None
        else:
            event = self @ start
            assert event is not None, "Cannot perform float slicing on empty performance"
            start_idx = self.global_events.index(event)
            if event.time < start:
                start_idx += 1
        
        if stop is None:
            stop_idx = None
        else:
            event = self @ stop
            assert event is not None, "Cannot perform float slicing on empty performance"
            stop_idx = self.global_events.index(event)
            if event.time > stop:
                pass
            else:
                stop_idx += 1
            if stop_idx >= len(self.global_events):
                stop_idx = None
        
        return Performance(self.global_events[start_idx:stop_idx])

    def __repr__(self) -> str:
        if len(self.global_events) <= 50:
            return f"Performance([{',\n'.join([event.__repr__() for event in self.global_events])}])"
        else:
            return f"Performance([{self.global_events[0]}, ...({len(self.global_events) - 2} events)..., {self.global_events[-1]}])"

type PerformanceLike = Performance | PathLike | Iterable[MidiEvent]

type SongStatKeys = Literal['duration_per_event', 'note_mean_song', 'velocity_mean']

def song_stats(melody: MelodyLike, performance: PerformanceLike) -> dict[SongStatKeys, float]:
    if not isinstance(melody, Melody):
        melody = Melody(melody)
    if not isinstance(performance, Performance):
        performance = Performance(performance)
    melodies = melody.split()
    sum_duration, sum_events = 0, 0
    for m in melodies:
        sum_duration += m.duration
        sum_events += len(m)
    duration_per_event = sum_duration / sum_events
    note_mean = statistics.mean(event.note for event in performance.global_events)
    velocity_mean = statistics.mean(event.velocity for event in performance.global_events)
    return {
        'duration_per_event': duration_per_event,
        'note_mean_song': note_mean,
        'velocity_mean': velocity_mean}