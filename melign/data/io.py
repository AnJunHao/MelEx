from pathlib import Path
from typing import overload, Literal, Iterable, cast
import mido
from warnings import warn

from melign.data.event import MidiEvent, MelEvent, EventLike

type PathLike = Path | str

@overload
def load_midi(  # type: ignore[reportOverlappingOverload]
    midi_file: PathLike,
    track_idx: int | None = None,
    include_velocity: Literal[True] = True,
) -> list[MidiEvent]: ...
@overload
def load_midi(
    midi_file: PathLike,
    track_idx: int | None = None,
    include_velocity: Literal[False] = False,
) -> list[MelEvent]: ...
def load_midi(
    midi_file: PathLike,
    track_idx: int | None = None,
    include_velocity: bool = True,
) -> list[MidiEvent] | list[MelEvent]:

    try:
        mid = mido.MidiFile(midi_file)
    except OSError as e:
        raise OSError(f"Error loading MIDI file: {midi_file}") from e
        
    # Track tempo changes and convert ticks to seconds
    tempo = 500000  # Default tempo (120 BPM)
    ticks_per_beat = mid.ticks_per_beat
    events: list[MidiEvent] | list[MelEvent] = []

    # Get the specified track or all tracks
    if track_idx is None:
        iter_tracks = mid.tracks
    else:
        iter_tracks = [mid.tracks[track_idx]]

    # Process all specified tracks to get note events with their exact timing (and velocity)
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
                if include_velocity:
                    events = cast(list[MidiEvent], events)
                    events.append(MidiEvent(time_seconds, msg.note, msg.velocity))
                else:
                    events = cast(list[MelEvent], events)
                    events.append(MelEvent(time_seconds, msg.note))

    events.sort(key=lambda x: x.time)

    return events

def load_note(
    note_file: PathLike,
) -> list[MelEvent]:
    with open(note_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    events: list[MelEvent] = []
    for line in lines:
        start, end, note = line.split()
        events.append(MelEvent(float(start), int(note)))
    return events

def melody_to_midi(melody: Iterable[EventLike] | PathLike, path: PathLike) -> None:
    
    # Convert to list of events
    if isinstance(melody, (Path, str)):
        mel_path = melody
        melody = Path(melody)
        if melody.suffix in (".mid", ".midi"):
            events = load_midi(melody)
        elif melody.suffix in (".note", ".txt"):
            events = load_note(melody)
        else:
            raise ValueError(f"Invalid file type: {melody.suffix}")
    else:
        mel_path = None
        events = list(melody)
    
    if not events:
        if mel_path is not None:
            warn(f"No event in {mel_path}, not saving to {path}")
        else:
            warn(f"No event in melody, not saving to {path}")
        return
    
    # Create MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Add tempo message
    track.append(mido.MetaMessage('set_tempo', tempo=500000))  # 120 BPM

    # Add time signature
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))  # 120 BPM
    
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

def melody_to_note(melody: Iterable[EventLike] | PathLike, path: PathLike) -> None:
    if isinstance(melody, (Path, str)):
        melody = Path(melody)
        if melody.suffix in (".mid", ".midi"):
            events = load_midi(melody)
        elif melody.suffix in (".note", ".txt"):
            events = load_note(melody)
        else:
            raise ValueError(f"Invalid file type: {melody.suffix}")
    else:
        events = list(melody)
    
    with open(path, "w", encoding="utf-8") as f:
        for i, event in enumerate(events):
            # Calculate note duration (until next event or default)
            if i < len(events) - 1:
                duration = events[i + 1].time - event.time
            else:
                duration = 1.0  # Default duration for last note
            
            # Write in format: start end note
            end_time = event.time + duration
            f.write(f"{event.time} {end_time} {event.note}\n")

def extract_original_events(
    melody: Iterable[EventLike] | PathLike,
    original_midi: PathLike,
    output_path: PathLike,
    time_tolerance: float = 0.001,
) -> None:
    """
    Extract original MIDI events from a MIDI file based on an extracted melody.
    
    This function finds the original MIDI events that correspond to the melody events
    based on timing alignment, preserving the original velocity, duration, and other
    MIDI properties.
    
    Args:
        melody: A Melody object or list of events representing the extracted melody
        original_midi: Path to the original MIDI file
        output_path: Path where to save the extracted MIDI file
    """
    # Convert to list of events
    if isinstance(melody, (Path, str)):
        melody = Path(melody)
        if melody.suffix in (".mid", ".midi"):
            events = load_midi(melody)
        elif melody.suffix in (".note", ".txt"):
            events = load_note(melody)
        else:
            raise ValueError(f"Invalid file type: {melody.suffix}")
    else:
        events = list(melody)
    
    if not events:
        return
    
    # Load the original MIDI file
    original_mid = mido.MidiFile(str(original_midi))
    
    # Extract all original MIDI events with timing information
    original_events = []
    tempo = 500000  # Default tempo (120 BPM)
    ticks_per_beat = original_mid.ticks_per_beat
    
    # Process all tracks to get note events with their exact timing
    for track in original_mid.tracks:
        track_time_ticks = 0
        for msg in track:
            track_time_ticks += msg.time
            
            # Handle tempo changes
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            
            # Store note_on and note_off events separately
            if msg.type in ['note_on', 'note_off']:
                time_seconds = track_time_ticks * tempo / (ticks_per_beat * 1000000)
                original_events.append({
                    'time': time_seconds,
                    'time_ticks': track_time_ticks,
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'type': msg.type,
                    'message': msg
                })
    
    # Sort original events by time
    original_events.sort(key=lambda x: x['time'])
    
    # Create a new MIDI file for the extracted events
    new_mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    new_track = mido.MidiTrack()
    new_mid.tracks.append(new_track)
    
    # Add tempo message (use the same tempo as original)
    new_track.append(mido.MetaMessage('set_tempo', tempo=500000))

    # Add time signature
    new_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))  # 120 BPM
    
    # For each melody event, find the corresponding original events
    selected_events = []
    
    for mel_event in events:
        # Find the original note_on event that matches this melody event
        for orig_event in original_events:
            if (orig_event['type'] == 'note_on' and 
                orig_event['note'] == mel_event.note and
                orig_event['velocity'] > 0 and  # Only consider actual note onsets
                abs(orig_event['time'] - mel_event.time) < time_tolerance):
                
                # Find the corresponding note_off event (either note_off or note_on with velocity=0)
                note_off_event = None
                for off_event in original_events:
                    if (off_event['note'] == mel_event.note and
                        off_event['time'] > orig_event['time'] and
                        ((off_event['type'] == 'note_off') or 
                         (off_event['type'] == 'note_on' and off_event['velocity'] == 0))):
                        note_off_event = off_event
                        break
                
                # If we found both on and off events, add them to selection
                if note_off_event:
                    selected_events.append(orig_event)
                    selected_events.append(note_off_event)
                break
    
    # Sort selected events by time
    selected_events.sort(key=lambda x: x['time'])
    
    # Convert selected events to MIDI messages and add to track
    current_time_ticks = 0
    
    for event in selected_events:
        # Calculate delta time in ticks
        delta_ticks = event['time_ticks'] - current_time_ticks
        
        # Create the message with the delta time
        msg = mido.Message(
            event['type'],
            note=event['note'],
            velocity=event['velocity'],
            time=delta_ticks
        )
        
        new_track.append(msg)
        current_time_ticks = event['time_ticks']
    
    # Save the new MIDI file
    new_mid.save(str(output_path))