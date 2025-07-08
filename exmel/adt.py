from functools import cached_property
from pathlib import Path
import bisect
from typing import overload
import mido
from dataclasses import dataclass
from warnings import deprecated
from functools import cache
from typing import Literal

@cache
def midi_to_note_name(midi_note):
    note_names = 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
    note_index = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{note_names[note_index]}{octave}"

@cache
def note_name_to_midi(note_name: str) -> int:
    note_names = 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
    note_index = note_names.index(note_name[:-1])
    octave = int(note_name[-1])
    return note_index + 12 * (octave + 1)

@dataclass(frozen=True, slots=True)
class MelEvent:
    time: float  # Time in seconds
    note: int    # MIDI note number (0-127)

    @property
    def note_name(self) -> str:
        return midi_to_note_name(self.note)

    def __sub__(self, other: 'MelEvent') -> 'MelEvent':
        # "a - b": time and note shift
        if isinstance(other, MelEvent):
            return MelEvent(self.time - other.time, self.note - other.note)
        return NotImplemented
    
    def __add__(self, other: 'MelEvent') -> 'MelEvent':
        # "a + b": time and note shift
        if isinstance(other, MelEvent):
            return MelEvent(self.time + other.time, self.note + other.note)
        return NotImplemented

    def __rshift__(self, other: float) -> 'MelEvent':
        # "a >> float": time shift
        if isinstance(other, float):
            return MelEvent(self.time + other, self.note)
        return NotImplemented

    def __lshift__(self, other: float) -> 'MelEvent':
        # "a << float": time shift
        if isinstance(other, float):
            return MelEvent(self.time - other, self.note)
        return NotImplemented

    def __xor__(self, other: int) -> 'MelEvent':
        # "a ^ int": note shift
        if isinstance(other, int):
            return MelEvent(self.time, self.note + other)
        return NotImplemented

    def __truediv__(self, other: 'MelEvent') -> float:
        # "a / b": time ratio
        if isinstance(other, MelEvent):
            return self.time / other.time
        return NotImplemented
    
    def __mul__(self, other: float) -> 'MelEvent':
        # "a * float": time scale
        if isinstance(other, float):
            return MelEvent(self.time * other, self.note)
        return NotImplemented

    def __repr__(self) -> str:
        # Convert note to note name
        return f"MelEvent_(time={self.time:.1f}, note='{self.note_name}')"

def MelEvent_(time: float, note: int | str) -> MelEvent:
    if isinstance(note, str):
        note = note_name_to_midi(note)
    return MelEvent(time, note)

@dataclass(frozen=True, slots=True)
class MidiEvent(MelEvent):
    time: float  # Time in seconds
    note: int    # MIDI note number (0-127)
    velocity: int  # Velocity (0-127)

    def __repr__(self) -> str:
        return f"MidiEvent_(time={self.time:.1f}, note='{self.note_name}', velocity={self.velocity})"

def MidiEvent_(time: float, note: int | str, velocity: int) -> MidiEvent:
    if isinstance(note, str):
        note = note_name_to_midi(note)
    return MidiEvent(time, note, velocity)

class Melody:

    def __init__(self,
                source: "Path | Melody | list[MelEvent] | list[MidiEvent]") -> None:
        """
        Initialize the melody with a MIDI file.
        """
        if isinstance(source, Path):
            self.events: list[MelEvent] = []
            self._load_midi(source)
        elif isinstance(source, Melody):
            self.events = source.events
        elif isinstance(source, list):
            self.events = [MelEvent(event.time, event.note) for event in source]
        else:
            raise TypeError(f"Invalid source type: {type(source)}")

        self.event_to_index = {event: i for i, event in enumerate(self.events)}
        self.time_to_index = {event.time: i for i, event in enumerate(self.events)}
    
    def _load_midi(self, midi_file: Path) -> None:
        """
        Load the MIDI file and convert ticks to seconds, then sort events by time.
        """
        mid = mido.MidiFile(str(midi_file))
        
        # Track tempo changes and convert ticks to seconds
        tempo = 500000  # Default tempo (120 BPM)
        ticks_per_beat = mid.ticks_per_beat
        events = []
        
        # Convert ticks to seconds for all tracks
        for track in mid.tracks:
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
        if isinstance(time_shift, float):
            return Melody([event << time_shift for event in self.events])
        return NotImplemented
    
    def __rshift__(self, time_shift: float) -> "Melody":
        """
        Shift the melody by a number of seconds.
        """
        if isinstance(time_shift, float):
            return Melody([event >> time_shift for event in self.events])
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
    
    def index(self, value: MelEvent | float) -> int:
        """Find the index of a value. O(1) time complexity if index_map is built."""
        if isinstance(value, MelEvent):
            return self.event_to_index[value]
        elif isinstance(value, float):
            return self.time_to_index[value]
        else:
            raise TypeError(f"Invalid value type: {type(value)}")

    @property
    def duration(self) -> float:
        return self.events[-1].time - self.events[0].time

@deprecated("This implementation is deprecated. Use PianoMidi with precomputed indices for better performance.")
class _PianoMidi:
    def __init__(self, midi_file: Path) -> None:
        """
        Initialize the tracker with a MIDI file.
        Time Complexity: O(n log n) where n is the number of note events
        """
        self.events_by_note: dict[int, list[tuple[float, int]]] = {}
        #                    dict[note, list[tuple[time_seconds, velocity]]]
        self._load_midi(midi_file)
    
    def _load_midi(self, midi_file: Path) -> None:
        """Load MIDI file, convert ticks to seconds, and organize events by note."""
        mid = mido.MidiFile(str(midi_file))
        
        # Track tempo changes and convert ticks to seconds
        tempo = 500000  # Default tempo (120 BPM)
        ticks_per_beat = mid.ticks_per_beat
        
        # Convert to absolute time in seconds and extract note_on events
        all_events = []
        
        for track in mid.tracks:
            track_time_ticks = 0
            for msg in track:
                track_time_ticks += msg.time
                
                # Handle tempo changes
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                
                if msg.type == 'note_on' and msg.velocity > 0:  # Note onset
                    # Convert ticks to seconds using current tempo
                    time_seconds = track_time_ticks * tempo / (ticks_per_beat * 1000000)
                    all_events.append(MidiEvent(time_seconds, msg.note, msg.velocity))
        
        # Organize events by note
        for event in all_events:
            if event.note not in self.events_by_note:
                self.events_by_note[event.note] = []
            self.events_by_note[event.note].append((event.time, event.velocity))
        
        # Sort each note's events by time
        for note in self.events_by_note:
            self.events_by_note[note].sort(key=lambda x: x[0])

    @cached_property
    def duration(self) -> float:
        """
        Calculate the duration of the MIDI file.
        Returns the time of the last event in seconds.
        """
        if not self.events_by_note:
            return 0.0
        
        # Find the maximum time across all events
        max_time = 0.0
        for note_events in self.events_by_note.values():
            if note_events:
                max_time = max(max_time, note_events[-1][0])
        
        return max_time

    def right_nearest(self, mel_event: MelEvent) -> MidiEvent | None:
        """
        Find the nearest occurrence of the note strictly to the right of the mel_event.
        This is equivalent to finding the leftmost occurrence of the note after the mel_event.
        """
        note = mel_event.note
        start_time = mel_event.time
        if note not in self.events_by_note:
            return None
        
        events = self.events_by_note[note]
        # Find the leftmost event strictly to the right of start_time
        idx = bisect.bisect_right(events, (start_time, float('inf')))
        
        if idx < len(events):
            time, velocity = events[idx]
            return MidiEvent(time, note, velocity)
        return None
    
    def left_nearest(self, mel_event: MelEvent) -> MidiEvent | None:
        """
        Find the nearest occurrence of the note strictly to the left of the mel_event.
        This is equivalent to finding the rightmost occurrence of the note before the mel_event.
        """
        note = mel_event.note
        end_time = mel_event.time
        if note not in self.events_by_note:
            return None
        
        events = self.events_by_note[note]
        # Find the rightmost event strictly to the left of end_time
        idx = bisect.bisect_left(events, (end_time, 0)) - 1
        
        if idx >= 0:
            time, velocity = events[idx]
            return MidiEvent(time, note, velocity)
        return None

    def nearest(self, mel_event: MelEvent) -> MidiEvent | None:
        """
        Find the nearest occurrence of the note to the mel_event.
        """
        note = mel_event.note
        ref_time = mel_event.time
        if note not in self.events_by_note:
            return None
        
        events = self.events_by_note[note]
        if not events:
            return None
        
        # Binary search for insertion point
        idx = bisect.bisect_left(events, (ref_time, 0))
        
        # Check neighbors to find the nearest
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(events):
            candidates.append(idx)
        
        if not candidates:
            return None
        
        # Find the nearest by absolute time difference
        nearest_idx = min(candidates, key=lambda i: abs(events[i][0] - ref_time))
        time, velocity = events[nearest_idx]
        return MidiEvent(time, note, velocity)
    
    def right_nearest_multi(self, mel_event: MelEvent, n: int) -> list[MidiEvent]:
        """
        Find n nearest occurrences of the note strictly to the right of the mel_event.
        Ordered from nearest to furthest (left to right).
        Time Complexity: O(log k + n) where k is the number of occurrences of the note
        """
        note = mel_event.note
        start_time = mel_event.time
        if note not in self.events_by_note:
            return []
        
        events = self.events_by_note[note]
        idx = bisect.bisect_right(events, (start_time, float('inf')))
        
        results = []
        for i in range(idx, min(idx + n, len(events))):
            time, velocity = events[i]
            results.append(MidiEvent(time, note, velocity))
        
        return results
    
    def left_nearest_multi(self, mel_event: MelEvent, n: int) -> list[MidiEvent]:
        """
        Find n nearest occurrences of the note strictly to the left of the mel_event.
        Ordered from nearest to furthest (right to left).
        Time Complexity: O(log k + n) where k is the number of occurrences of the note
        """
        note = mel_event.note
        end_time = mel_event.time
        if note not in self.events_by_note:
            return []
        
        events = self.events_by_note[note]
        idx = bisect.bisect_left(events, (end_time, 0)) - 1
        
        results = []
        for i in range(idx, max(idx - n, -1), -1):
            if i >= 0:
                time, velocity = events[i]
                results.append(MidiEvent(time, note, velocity))
        
        return results
    
    def nearest_multi(self, mel_event: MelEvent, n: int) -> list[MidiEvent]:
        """
        Find n occurrences of the note nearest to the mel_event.
        Ordered from nearest to furthest.
        Time Complexity: O(log k + n) where k is the number of occurrences of the note
        """
        note = mel_event.note
        ref_time = mel_event.time
        if note not in self.events_by_note or n <= 0:
            return []
        
        events = self.events_by_note[note]
        if not events:
            return []
        
        # Use binary search to find insertion point
        idx = bisect.bisect_left(events, (ref_time, 0))
        
        # Use two pointers to expand outwards from the insertion point
        left = idx - 1
        right = idx
        results = []
        
        # Collect n nearest events by expanding left and right
        while len(results) < n and (left >= 0 or right < len(events)):
            left_dist = float('inf')
            right_dist = float('inf')
            
            # Calculate distances
            if left >= 0:
                left_dist = abs(events[left][0] - ref_time)
            if right < len(events):
                right_dist = abs(events[right][0] - ref_time)
            
            # Choose the closer event
            if left_dist <= right_dist and left >= 0:
                time, velocity = events[left]
                results.append(MidiEvent(time, note, velocity))
                left -= 1
            elif right < len(events):
                time, velocity = events[right]
                results.append(MidiEvent(time, note, velocity))
                right += 1
            else:
                break
        
        # Sort results by distance to maintain nearest-to-furthest ordering
        results.sort(key=lambda event: abs(event.time - ref_time))
        
        return results

type time_velocity_tuple = tuple[float, int]

@dataclass(slots=True)
class Match:
    current_miss: int
    sum_miss: int
    sum_error: float
    events: list[MidiEvent]

class PianoMidi:
    """
    Optimized MIDI event tracker that uses precomputed indices and lookup tables
    to support efficient nearest neighbor queries on note events.
    """
    
    def __init__(self, midi_file: Path) -> None:
        """
        Initialize the tracker by loading MIDI data and building search indices.
        
        Time Complexity: O(n log n) where n is the number of note events
        """
        
        self.events_by_note: dict[int, list[time_velocity_tuple]] = {}
        self._load_midi(midi_file)
        self._build_indices()
    
    def _load_midi(self, midi_file: Path) -> None:
        """
        Load MIDI file, convert ticks to seconds, and organize events by note.
        
        Time Complexity: O(n log n) where n is the number of note events
        """
        mid = mido.MidiFile(str(midi_file))
        
        # Track tempo changes and convert ticks to seconds
        tempo = 500000  # Default tempo (120 BPM)
        ticks_per_beat = mid.ticks_per_beat
        
        # Convert to absolute time in seconds and extract note_on events
        all_events: list[MidiEvent] = []
        
        for track in mid.tracks:
            track_time_ticks = 0
            for msg in track:
                track_time_ticks += msg.time
                
                # Handle tempo changes
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                
                if msg.type == 'note_on' and msg.velocity > 0:  # Note onset
                    # Convert ticks to seconds using current tempo
                    time_seconds = track_time_ticks * tempo / (ticks_per_beat * 1000000)
                    all_events.append(MidiEvent(time_seconds, msg.note, msg.velocity))
        
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
        self.global_events: list[tuple[float, int, int]] = []  # (time, note, velocity)
        for note, events in self.events_by_note.items():
            for time, velocity in events:
                self.global_events.append((time, note, velocity))
        
        self.global_events.sort()  # Sort by time
        self.global_times = [event[0] for event in self.global_events]
        
        # 3. Precompute common query patterns using lookup tables
        self.note_time_to_index: dict[int, dict[float, int]] = {}
        # note -> {time -> index}
        for note, events in self.events_by_note.items():
            self.note_time_to_index[note] = {event[0]: i for i, event in enumerate(events)}

    @property
    def duration(self) -> float:
        """
        Calculate the duration of the MIDI file.
        Returns the time of the last event in seconds.
        
        Time Complexity: O(1) - cached property
        """
        if not self.global_events:
            return 0.0
        return self.global_events[-1][0]

    def right_nearest(self, mel_event: MelEvent, use_lookup: bool = True) -> MidiEvent | None:
        """
        Find the nearest occurrence of the note strictly to the right of the given event.
        
        Time Complexity: O(log k) where k is the number of events for the note
        """
        note = mel_event.note
        start_time = mel_event.time
        
        if note not in self.note_times:
            return None

        if use_lookup:
            idx = self.note_time_to_index[note].get(start_time, None)
            if idx is not None:
                if idx < len(self.note_velocities[note]) - 1:
                    t, v = self.events_by_note[note][idx + 1]
                    return MidiEvent(t, note, v)
                else:
                    return None
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        # Binary search on the precomputed time array
        idx = bisect.bisect_right(times, start_time)
        
        if idx < len(times):
            return MidiEvent(times[idx], note, velocities[idx])
        return None
    
    def left_nearest(self, mel_event: MelEvent, use_lookup: bool = True) -> MidiEvent | None:
        """
        Find the nearest occurrence of the note strictly to the left of the given event.
        
        Time Complexity: O(log k) where k is the number of events for the note
        """
        note = mel_event.note
        end_time = mel_event.time
        
        if note not in self.note_times:
            return None

        if use_lookup:
            idx = self.note_time_to_index[note].get(end_time, None)
            if idx is not None:
                if idx > 0:
                    t, v = self.events_by_note[note][idx - 1]
                    return MidiEvent(t, note, v)
                else:
                    return None
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        # Binary search on the precomputed time array
        idx = bisect.bisect_left(times, end_time) - 1
        
        if idx >= 0:
            return MidiEvent(times[idx], note, velocities[idx])
        return None

    def nearest(self, mel_event: MelEvent) -> MidiEvent | None:
        """
        Find the nearest occurrence of the note to the given event.
        
        Args:
            mel_event: The reference event to find nearest neighbor for
            use_buckets: Whether to use bucket lookup optimization (default: True)
        
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
        
        # Fallback to binary search for edge cases
        idx = bisect.bisect_left(times, ref_time)
        
        # Check neighbors to find the nearest
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(times):
            candidates.append(idx)
        
        if not candidates:
            return None
        
        # Find the nearest by absolute time difference
        nearest_idx = min(candidates, key=lambda i: abs(times[i] - ref_time))
        return MidiEvent(times[nearest_idx], note, velocities[nearest_idx])
    
    def right_nearest_multi(self, mel_event: MelEvent, n: int, use_lookup: bool = True) -> list[MidiEvent]:
        """
        Find n nearest occurrences of the note strictly to the right of the given event.
        
        Time Complexity: O(log k + n) where k is the number of events for the note
        """
        note = mel_event.note
        start_time = mel_event.time
        
        if note not in self.note_times or n <= 0:
            return []

        if use_lookup:
            idx = self.note_time_to_index[note].get(start_time, None)
            if idx is not None:
                list_tv = self.events_by_note[note][idx + 1:idx + n + 1]
                return [MidiEvent(t, note, v) for t, v in list_tv]
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        idx = bisect.bisect_right(times, start_time)
        
        # Use list slicing for bulk operations
        end_idx = min(idx + n, len(times))
        result_times = times[idx:end_idx]
        result_velocities = velocities[idx:end_idx]
        
        return [MidiEvent(time, note, vel) for time, vel in zip(result_times, result_velocities)]
    
    def left_nearest_multi(self, mel_event: MelEvent, n: int, use_lookup: bool = True) -> list[MidiEvent]:
        """
        Find n nearest occurrences of the note strictly to the left of the given event.
        
        Time Complexity: O(log k + n) where k is the number of events for the note
        """
        note = mel_event.note
        end_time = mel_event.time
        
        if note not in self.note_times or n <= 0:
            return []

        if use_lookup:
            idx = self.note_time_to_index[note].get(end_time, None)
            if idx is not None:
                if idx - n - 1 >= 0:
                    list_tv = self.events_by_note[note][idx-1:idx-n-1:-1]
                elif idx - 1 >= 0:
                    list_tv = self.events_by_note[note][idx-1::-1]
                else:
                    list_tv = []
                return [MidiEvent(t, note, v) for t, v in list_tv]
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        idx = bisect.bisect_left(times, end_time) - 1
        
        # Use list slicing for bulk operations
        start_idx = max(idx - n + 1, 0)
        result_times = times[start_idx:idx + 1]
        result_velocities = velocities[start_idx:idx + 1]
        
        # Reverse to get nearest-to-furthest order
        result_times.reverse()
        result_velocities.reverse()
        
        return [MidiEvent(time, note, vel) for time, vel in zip(result_times, result_velocities)]
    
    def nearest_multi(self, mel_event: MelEvent, n: int) -> list[MidiEvent]:
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

    def nearest_global(self, mel_event: MelEvent, n: int = 1) -> list[MidiEvent]:
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
                left_dist = abs(self.global_events[left][0] - ref_time)
            if right < len(self.global_events):
                right_dist = abs(self.global_events[right][0] - ref_time)
            
            # Choose the closer event
            if left_dist <= right_dist and left >= 0:
                time, note, velocity = self.global_events[left]
                results.append(MidiEvent(time, note, velocity))
                left -= 1
            elif right < len(self.global_events):
                time, note, velocity = self.global_events[right]
                results.append(MidiEvent(time, note, velocity))
                right += 1
            else:
                break
        
        # Sort by distance for consistent ordering
        results.sort(key=lambda event: abs(event.time - ref_time))
        
        return results

    def match_chunk(self,
        chunk: Melody,
        direction: Literal["l2r", "r2l"],
        local_tolerance: float = 0.1,
        miss_tolerance: int = 2,
        tolerate_start: bool = True) -> list[Match]:
        """
        Match a chunk of melody events to the most similar MIDI events across the timeline.
        
        Args:
            chunk: A Melody object containing the sequence of events to match against
            direction: The search direction for building matches. "l2r" for left-to-right, "r2l" for right-to-left.
            local_tolerance: Maximum allowed time difference for individual event matches (seconds)
            miss_tolerance: Maximum allowed number of missed events
        """
        if len(chunk) == 0:
            return []
        if len(chunk) == 1:
            tv_list = self.events_by_note[chunk[0].note]
            return [Match(0, 0, 0, [MidiEvent(tv[0], chunk[0].note, tv[1])])
                    for tv in tv_list]
        
        for i in range(miss_tolerance+1):
            if direction == "l2r":
                candidates = self.match_chunk(chunk[i:len(chunk)-1], "l2r", local_tolerance, miss_tolerance, False)
            elif direction == "r2l":
                candidates = self.match_chunk(chunk[1:len(chunk)-i], "r2l", local_tolerance, miss_tolerance, False)
            else:
                raise ValueError(f"Invalid direction: {direction}")
            if not tolerate_start:
                break
            if len(candidates) > 0:
                for c in candidates:
                    c.sum_miss += i
                break
        
        if len(candidates) == 0:
            return []

        output = []
        for candidate in candidates:

            if direction == "l2r":
                melody_shift = chunk[-1] - chunk[-2-candidate.current_miss]
                predicted_event = candidate.events[-1] + melody_shift
            elif direction == "r2l":
                melody_shift = chunk[0] - chunk[1+candidate.current_miss]
                predicted_event = candidate.events[0] + melody_shift
            else:
                assert False, "unreachable"

            nearest = self.nearest(predicted_event)
            if nearest is not None:
                diff = nearest - predicted_event
                if abs(diff.time) / abs(melody_shift.time) <= local_tolerance:

                    if direction == "l2r":
                        candidate.events.append(nearest)
                    elif direction == "r2l":
                        candidate.events.insert(0, nearest)
                    else:
                        assert False, "unreachable"

                    candidate.current_miss = 0
                    candidate.sum_error += abs(diff.time)
                else:
                    candidate.current_miss += 1
                    candidate.sum_miss += 1
            else:
                candidate.current_miss += 1
                candidate.sum_miss += 1
            if candidate.current_miss <= miss_tolerance:
                output.append(candidate)
        return output

type MelodyLike = Melody | list[MelEvent] | list[MidiEvent]

def save_melody(melody: MelodyLike, path: Path) -> None:
    """
    Save a melody to a MIDI file.
    Each event's duration extends until the start of the next event.
    
    Args:
        melody: A Melody object or list of events to save
        path: Path where to save the MIDI file
    """
    # Convert to list of events
    if isinstance(melody, Melody):
        events = melody.events
    elif isinstance(melody, list):
        events = melody
    else:
        raise TypeError(f"Invalid melody type: {type(melody)}")
    
    if not events:
        return
    
    # Create MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Add tempo message
    track.append(mido.MetaMessage('set_tempo', tempo=500000))  # 120 BPM
    
    # Convert seconds to ticks
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # microseconds per beat
    
    def seconds_to_ticks(seconds):
        return int(seconds * ticks_per_beat * 1000000 / tempo)
    
    # Process events
    current_time = 0.0
    
    for i, event in enumerate(events):
        # Get velocity - default to 64 if not available
        velocity = event.velocity if isinstance(event, MidiEvent) else 64
        
        # Calculate note duration (until next event or default)
        if i < len(events) - 1:
            duration = events[i + 1].time - event.time
        else:
            duration = 1.0  # Default duration for last note
        
        # Add note_on event
        delta_time = event.time - current_time
        delta_ticks = seconds_to_ticks(delta_time)
        track.append(mido.Message('note_on', note=event.note, velocity=velocity, time=delta_ticks))
        current_time = event.time
        
        # Add note_off event
        duration_ticks = seconds_to_ticks(duration)
        track.append(mido.Message('note_off', note=event.note, velocity=velocity, time=duration_ticks))
        current_time += duration
    
    # Save the file
    mid.save(str(path))