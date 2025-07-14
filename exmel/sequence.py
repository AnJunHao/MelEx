from pathlib import Path
import bisect
import mido
from typing import Literal, Iterator, overload, Iterable
import warnings
import statistics

from exmel.event import MelEvent, MidiEvent, EventLike
from exmel.io import load_midi, load_note, PathLike, save_melody

class Melody:

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
            elif source.suffix in (".note"):
                self.events = load_note(source)
            else:
                raise ValueError(f"Invalid file type: {source.suffix}")
        elif isinstance(source, Melody):
            self.events = source.events
        elif isinstance(source, Iterable):
            self.events = [MelEvent(event.time, event.note) for event in source]
        else:
            raise TypeError(f"Invalid source type: {type(source)}")
    
    def _load_midi(self, midi_file: Path, track_idx: int | None = None) -> None:
        """
        Load the MIDI file and convert ticks to seconds, then sort events by time.
        """
        mid = mido.MidiFile(midi_file)
        
        # Track tempo changes and convert ticks to seconds
        tempo = 500000  # Default tempo (120 BPM)
        ticks_per_beat = mid.ticks_per_beat
        events = []
        
        # Convert ticks to seconds for all tracks
        if track_idx is None:
            iter_tracks = mid.tracks
        else:
            iter_tracks = [mid.tracks[track_idx]]
        for track in iter_tracks:
            track_time_ticks = 0
            for msg in track:
                track_time_ticks += msg.time
                
                # Handle tempo changes
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                
                if msg.type == 'note_on' and msg.velocity > 0:  # Note onset
                    # Convert ticks to seconds using current tempo
                    time_seconds = track_time_ticks * tempo / (ticks_per_beat * 1000000)
                    events.append(MelEvent(time_seconds, msg.note))
        
        events.sort(key=lambda x: x.time)
        self.events = events

    def __xor__(self, shift: int) -> "Melody":
        """
        Shift the melody by a number of semitones.
        """
        if isinstance(shift, int):
            return Melody([event ^ shift for event in self.events])
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
        if other == 12:
            return Melody([event % 12 for event in self.events])
        elif isinstance(other, int):
            warnings.warn(f"Modulo by {other} does not make sense. You may want to use 12 instead.")
            return Melody([event % other for event in self.events])
        else:
            return NotImplemented

    def __repr__(self) -> str:
        if len(self.events) <= 2:
            return f"Melody([{self.events.__repr__()}])"
        elif len(self.events) <= 50:
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
    def __getitem__(self, key: int | slice) -> "MelEvent | Melody":
        if isinstance(key, slice):
            return Melody(self.events[key])
        else:
            return self.events[key]

    def __iter__(self) -> Iterator[MelEvent]:
        return iter(self.events)

    def nearest(self, time: float) -> MelEvent:
        """
        Find the nearest event to the given time.
        """
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

    def save(self, path: PathLike) -> None:
        save_melody(self.events, path)

    def plot(self, save_path: PathLike | None = None, show_plot: bool = True) -> None:
        from exmel.vis import plot_melody
        plot_melody(self, save_path, show_plot)

    def diff(self) -> list[float]:
        return [self[i+1].time - self[i].time for i in range(len(self)-1)]

    def mean_diff(self) -> float:
        return statistics.geometric_mean(self.diff())

    def split(self, threshold: float = 16) -> list["Melody"]:
        """
        Separate the melody into multiple melodies based on the threshold.
        """
        diffs = self.diff()
        threshold = statistics.geometric_mean(diffs) * threshold
        indices = [i+1 for i, d in enumerate(diffs) if d > threshold]
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
            self.events_by_note = {event.note: [(event.time, event.velocity)] for event in source}
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

    def nearest_global(self, mel_event: EventLike, n: int = 1) -> list[MidiEvent]:
        """
        Find n nearest events across all notes to the given event.
        
        Time Complexity: O(log N + n) where N is the total number of events
        """
        if n <= 0:
            return []
        
        ref_time = mel_event.time
        
        # Binary search on global time array
        idx = bisect.bisect_left(self.global_times, ref_time)
        
        # Use two-pointer expansion on global events
        left = idx - 1
        right = idx
        results = []
        
        # Collect n nearest events globally
        while len(results) < n and (left >= 0 or right < len(self.global_events)):
            left_dist = float('inf')
            right_dist = float('inf')
            
            # Calculate distances
            if left >= 0:
                left_dist = abs(self.global_events[left].time - ref_time)
            if right < len(self.global_events):
                right_dist = abs(self.global_events[right].time - ref_time)
            
            # Choose the closer event
            if left_dist <= right_dist and left >= 0:
                results.append(self.global_events[left])
                left -= 1
            elif right < len(self.global_events):
                results.append(self.global_events[right])
                right += 1
            else:
                break
        
        # Sort by distance for consistent ordering
        results.sort(key=lambda event: abs(event.time - ref_time))
        
        return results

    def plot(self, save_path: PathLike | None = None, show_plot: bool = True) -> None:
        from exmel.vis import plot_performance
        plot_performance(self, save_path, show_plot)

    def between(self, from_: MelEvent, to_: MelEvent) -> list[MidiEvent]:
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

type PerformanceLike = Performance | PathLike | Iterable[MidiEvent]