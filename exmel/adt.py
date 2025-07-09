from pathlib import Path
import bisect
import mido
from dataclasses import dataclass
from functools import cache
from typing import Literal, Iterator, overload
import warnings

warnings.simplefilter("once")

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
        return f"MelEvent_(time={self.time:.4f}, note='{self.note_name}')"

    def __floordiv__(self, other: float) -> 'MelEvent':
        # Quantize time to the nearest multiple of other
        if isinstance(other, float):
            return MelEvent(round(self.time / other) * other, self.note)
        return NotImplemented
    
    def __mod__(self, other: Literal[12]) -> 'MelEvent':
        # "a % b": note modulo
        if other == 12:
            return MelEvent(self.time, self.note % 12)
        elif isinstance(other, int):
            warnings.warn(f"Modulo by {other} does not make sense. You may want to use 12 instead.")
            return MelEvent(self.time, self.note % other)
        else:
            return NotImplemented

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
        return f"MidiEvent_(time={self.time:.4f}, note='{self.note_name}', velocity={self.velocity})"

def MidiEvent_(time: float, note: int | str, velocity: int) -> MidiEvent:
    if isinstance(note, str):
        note = note_name_to_midi(note)
    return MidiEvent(time, note, velocity)

class Melody:

    def __init__(self,
                source: "Path | Melody | list[MelEvent] | list[MidiEvent]",
                track_idx: int | None = None) -> None:
        """
        Initialize the melody with a MIDI file.
        """
        if isinstance(source, Path):
            self.events: list[MelEvent] = []
            self._load_midi(source, track_idx)
        elif isinstance(source, Melody):
            self.events = source.events
        elif isinstance(source, list):
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

    def __floordiv__(self, other: float) -> "Melody":
        """
        Quantize the melody to the nearest multiple of other.
        """
        if isinstance(other, float):
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

type time_velocity_tuple = tuple[float, int]
type MelodyLike = Melody | list[MelEvent] | list[MidiEvent]

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
        direction: Literal["l2r", "r2l"] = "l2r",
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
            tolerate_start: Whether to tolerate at the start of the search (based on direction)
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

    def match(self,
        melody: Melody,
        local_tolerance: float = 0.1,
        miss_tolerance: int = 2,
        tolerate_start: bool = True,
        length_threshold: int = 10) -> Match:
        """
        Match a melody to the most similar MIDI events across the timeline.
        """
        raise NotImplementedError("Not implemented")