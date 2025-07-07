
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import pretty_midi
import os

def display_midi_piano_roll(midi_file_path: str, 
                           figsize: tuple[int, int] = (15, 10),
                           note_height: float = 0.8,
                           show_velocity: bool = True,
                           color_map: str = 'viridis',
                           title: str | None = None,
                           save_path: str | None = None,
                           time_range: tuple[float, float] | None = None) -> None:
    """
    Display a MIDI piano roll as a pretty, readable graph with note annotations.
    
    Args:
        midi_file_path: Path to the MIDI file
        figsize: Figure size (width, height)
        note_height: Height of each note rectangle (0-1)
        time_resolution: Time resolution in seconds for the x-axis
        show_velocity: Whether to color notes by velocity
        color_map: Matplotlib colormap for velocity coloring
        title: Optional title for the plot
        save_path: Optional path to save the plot
        time_range: Optional tuple of (start_time, end_time) in seconds to limit display range
    """
    
    # Load MIDI file using pretty_midi for better note extraction
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return
    
    # Extract all notes from all instruments
    all_notes = []
    for instrument in midi_data.instruments:
        all_notes.extend(instrument.notes)
    
    if not all_notes:
        print("No notes found in the MIDI file.")
        return
    
    # Get time range
    end_time = max(note.end for note in all_notes)
    
    # Apply time range filter if specified
    if time_range is not None:
        start_time, end_time = time_range
        # Filter notes that fall within the specified time range
        all_notes = [note for note in all_notes if note.start < end_time and note.end > start_time]
        if not all_notes:
            print("No notes found in the specified time range.")
            return
    else:
        start_time = 0
    
    # Get note range from the filtered notes
    note_numbers = [note.pitch for note in all_notes]
    min_note = min(note_numbers)
    max_note = max(note_numbers)
    
    # Add some padding to the note range for better visualization
    note_padding = 3
    min_note = max(0, min_note - note_padding)
    max_note = min(127, max_note + note_padding)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up the plot
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(min_note - 0.5, max_note + 0.5)
    
    # Create note name mapping
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Function to get note name
    def get_note_name(note_number):
        octave = (note_number // 12) - 1
        note_name = note_names[note_number % 12]
        return f"{note_name}{octave}"
    
    # Draw notes
    for note in all_notes:
        # Calculate color based on velocity if enabled
        if show_velocity:
            # Normalize velocity to 0-1 range
            velocity_norm = note.velocity / 127.0
            color = plt.cm.get_cmap(color_map)(velocity_norm)
        else:
            color = 'steelblue'
        
        # Create rectangle for the note
        rect = Rectangle(
            (note.start, note.pitch - note_height/2),
            note.end - note.start,
            note_height,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(rect)
    
    # Set up y-axis with note names
    y_ticks = []
    y_labels = []
    
    # Only show notes that are actually used or are octave boundaries
    for note_num in range(min_note, max_note + 1):
        # Show all notes in the range, but we'll filter some out later
        y_ticks.append(note_num)
        y_labels.append(get_note_name(note_num))
    
    # Filter y-axis to show fewer labels for better readability
    # Show every 4th note to avoid overcrowding
    step = max(1, len(y_ticks) // 20)  # Aim for ~20 labels
    filtered_ticks = y_ticks[::step]
    filtered_labels = y_labels[::step]
    
    ax.set_yticks(filtered_ticks)
    ax.set_yticklabels(filtered_labels, fontsize=10)
    
    # Set up x-axis
    # Convert time to seconds and format nicely
    x_ticks = np.arange(start_time, end_time + 1, max(1, (end_time - start_time) // 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t:.1f}s" for t in x_ticks], fontsize=10)
    
    # Customize the plot
    ax.grid(True, alpha=0.3, axis='both')
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Notes', fontsize=12, fontweight='bold')
    
    # Set title
    if title is None:
        title = f"Piano Roll: {os.path.basename(midi_file_path)}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add velocity legend if enabled
    if show_velocity:
        # Create a colorbar for velocity
        norm = plt.Normalize(0, 127) # type: ignore
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Velocity', fontsize=10, fontweight='bold')
        cbar.set_ticks([0, 32, 64, 96, 127])
        cbar.set_ticklabels(['0', '32', '64', '96', '127'])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Piano roll saved to: {save_path}")
    
    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Test with one of the MIDI files
    midi_file = "mid_files/weiyi_melody.mid"
    if os.path.exists(midi_file):
        print(f"Displaying piano roll for: {midi_file}")
        display_midi_piano_roll(midi_file)
    else:
        print(f"MIDI file not found: {midi_file}")

