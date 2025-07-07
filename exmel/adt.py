from functools import cached_property
from pathlib import Path
import bisect
import mido
from dataclasses import dataclass
from warnings import deprecated

@dataclass(frozen=True)
class MelEvent:
    time: float  # Time in seconds
    note: int    # MIDI note number (0-127)

    def __sub__(self, other: 'MelEvent') -> 'MelEvent':
        return MelEvent(self.time - other.time, self.note - other.note)
    
    def __add__(self, other: 'MelEvent') -> 'MelEvent':
        return MelEvent(self.time + other.time, self.note + other.note)

    def __div__(self, other: 'MelEvent') -> float:
        return self.time / other.time
    
    def __mul__(self, other: float) -> 'MelEvent':
        return MelEvent(self.time * other, self.note)

@dataclass(frozen=True)
class MidiEvent(MelEvent):
    time: float  # Time in seconds
    note: int    # MIDI note number (0-127)
    velocity: int  # Velocity (0-127)

class Melody:
    def __init__(self, midi_file: Path) -> None:
        """
        Initialize the melody with a MIDI file.
        """
        self.events: tuple[MelEvent] = tuple()
        self._load_midi(midi_file)
        self.index_map: dict[MelEvent, int] = {event: i for i, event in enumerate(self.events)}
    
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
        self.events = tuple(events)

    def __repr__(self) -> str:
        return self.events.__repr__()

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> MelEvent:
        return self.events[index]
    
    def index(self, value: MelEvent) -> int:
        """Find the index of a value. O(1) time complexity."""
        return self.index_map[value]

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
        self.events_by_note: dict[int, list[tuple[float, int]]] = {}
        self._load_midi(midi_file)
        self._build_optimized_indices()
    
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
    
    def _build_optimized_indices(self) -> None:
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
        self._build_lookup_tables()
    
    def _build_lookup_tables(self) -> None:
        """
        Build time-bucketed lookup tables to accelerate nearest neighbor queries.
        
        Time Complexity: O(n) where n is the number of note events
        """
        # For each note, build a lookup table for fast nearest neighbor queries
        self.nearest_lookup: dict[int, dict[int, int]] = {}  # note -> {time_bucket -> closest_event_idx}
        
        for note, times in self.note_times.items():
            if not times:
                continue
                
            # Create time buckets for constant-time approximate lookups
            min_time = times[0]
            max_time = times[-1]
            
            if max_time == min_time:
                # Single event case
                self.nearest_lookup[note] = {0: 0}
                continue
            
            # Use 1000 buckets for good granularity vs memory tradeoff
            num_buckets = min(1000, len(times) * 2)
            bucket_size = (max_time - min_time) / num_buckets
            
            lookup_table = {}
            for bucket in range(num_buckets + 1):
                bucket_time = min_time + bucket * bucket_size
                
                # Find the closest event index for this bucket
                idx = bisect.bisect_left(times, bucket_time)
                
                # Check both neighbors to find the nearest
                candidates = []
                if idx > 0:
                    candidates.append(idx - 1)
                if idx < len(times):
                    candidates.append(idx)
                
                if candidates:
                    nearest_idx = min(candidates, key=lambda i: abs(times[i] - bucket_time))
                    lookup_table[bucket] = nearest_idx
            
            self.nearest_lookup[note] = lookup_table
    
    def _get_bucket(self, note: int, time: float) -> int:
        """
        Get the bucket index for a given note and time.
        
        Time Complexity: O(1)
        """
        if note not in self.note_times or not self.note_times[note]:
            return 0
        
        times = self.note_times[note]
        min_time = times[0]
        max_time = times[-1]
        
        if max_time == min_time:
            return 0
        
        num_buckets = min(1000, len(times) * 2)
        bucket_size = (max_time - min_time) / num_buckets
        bucket = int((time - min_time) / bucket_size)
        
        return max(0, min(bucket, num_buckets))

    @cached_property
    def duration(self) -> float:
        """
        Calculate the duration of the MIDI file.
        Returns the time of the last event in seconds.
        
        Time Complexity: O(1) - cached property
        """
        if not self.global_events:
            return 0.0
        return self.global_events[-1][0]

    def right_nearest(self, mel_event: MelEvent) -> MidiEvent | None:
        """
        Find the nearest occurrence of the note strictly to the right of the given event.
        
        Time Complexity: O(log k) where k is the number of events for the note
        """
        note = mel_event.note
        start_time = mel_event.time
        
        if note not in self.note_times:
            return None
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        # Binary search on the precomputed time array
        idx = bisect.bisect_right(times, start_time)
        
        if idx < len(times):
            return MidiEvent(times[idx], note, velocities[idx])
        return None
    
    def left_nearest(self, mel_event: MelEvent) -> MidiEvent | None:
        """
        Find the nearest occurrence of the note strictly to the left of the given event.
        
        Time Complexity: O(log k) where k is the number of events for the note
        """
        note = mel_event.note
        end_time = mel_event.time
        
        if note not in self.note_times:
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
        
        Time Complexity: O(1) for lookup table hits, O(log k) for misses
        where k is the number of events for the note
        """
        note = mel_event.note
        ref_time = mel_event.time
        
        if note not in self.note_times:
            return None
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        if not times:
            return None
        
        # Try lookup table first for constant-time access
        if note in self.nearest_lookup:
            bucket = self._get_bucket(note, ref_time)
            if bucket in self.nearest_lookup[note]:
                lookup_idx = self.nearest_lookup[note][bucket]
                
                # Verify the lookup result is reasonable (within one bucket)
                if 0 <= lookup_idx < len(times):
                    lookup_time = times[lookup_idx]
                    
                    # Check if lookup is good enough or if we need to refine
                    min_time = times[0]
                    max_time = times[-1]
                    bucket_size = (max_time - min_time) / min(1000, len(times) * 2) if max_time != min_time else 0
                    
                    # If we're within the bucket tolerance, use the lookup result
                    if abs(lookup_time - ref_time) <= bucket_size:
                        return MidiEvent(lookup_time, note, velocities[lookup_idx])
        
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
    
    def right_nearest_multi(self, mel_event: MelEvent, n: int) -> list[MidiEvent]:
        """
        Find n nearest occurrences of the note strictly to the right of the given event.
        
        Time Complexity: O(log k + n) where k is the number of events for the note
        """
        note = mel_event.note
        start_time = mel_event.time
        
        if note not in self.note_times or n <= 0:
            return []
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        idx = bisect.bisect_right(times, start_time)
        
        # Use list slicing for bulk operations
        end_idx = min(idx + n, len(times))
        result_times = times[idx:end_idx]
        result_velocities = velocities[idx:end_idx]
        
        return [MidiEvent(time, note, vel) for time, vel in zip(result_times, result_velocities)]
    
    def left_nearest_multi(self, mel_event: MelEvent, n: int) -> list[MidiEvent]:
        """
        Find n nearest occurrences of the note strictly to the left of the given event.
        
        Time Complexity: O(log k + n) where k is the number of events for the note
        """
        note = mel_event.note
        end_time = mel_event.time
        
        if note not in self.note_times or n <= 0:
            return []
        
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
        
        Time Complexity: O(log k + n) where k is the number of events for the note
        """
        note = mel_event.note
        ref_time = mel_event.time
        
        if note not in self.note_times or n <= 0:
            return []
        
        times = self.note_times[note]
        velocities = self.note_velocities[note]
        
        if not times:
            return []
        
        # Binary search to find insertion point
        idx = bisect.bisect_left(times, ref_time)
        
        # Use two-pointer expansion
        left = idx - 1
        right = idx
        results = []
        
        # Collect n nearest events by expanding left and right
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