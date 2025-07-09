from pathlib import Path
import mido

from exmel.adt import Melody, MidiEvent, MelodyLike

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

def extract_original_events(melody: MelodyLike, original_midi: Path, output_path: Path) -> None:
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
    if isinstance(melody, Melody):
        events = melody.events
    elif isinstance(melody, list):
        events = melody
    else:
        raise TypeError(f"Invalid melody type: {type(melody)}")
    
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
    time_tolerance = 0.001  # 1ms tolerance for timing matching
    
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