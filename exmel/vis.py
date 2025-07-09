
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
import pretty_midi
import os
from exmel.adt import MelodyLike, Melody
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class EvaluationResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    plot: Figure

def evaluate_melody(gt: MelodyLike, pred: MelodyLike, tolerance: float = 0.1, save_path: str | None = None) -> EvaluationResult:
    """
    Evaluate the accuracy of a predicted melody against a ground truth melody.
    Creates a piano roll visualization with color coding for tp/fp/fn.
    
    Args:
        gt: Ground truth melody
        pred: Predicted melody
        tolerance: Time tolerance for matching notes (seconds)
        save_path: Optional path to save the visualization plot
    """
    gt = Melody(gt)
    pred = Melody(pred)
    
    # Track which notes are tp, fp, fn for visualization
    tp_notes = []
    fp_notes = []
    fn_notes = []
    
    # Find true positives and mark matched gt notes
    matched_gt_indices = set()
    for i, p_event in enumerate(pred):
        matched = False
        for j, g_event in enumerate(gt):
            if p_event.note == g_event.note:
                if abs(p_event.time - g_event.time) < tolerance:
                    tp_notes.append(p_event)
                    matched_gt_indices.add(j)
                    matched = True
                    break
        if not matched:
            fp_notes.append(p_event)
    
    # Find false negatives (unmatched gt notes)
    for j, g_event in enumerate(gt):
        if j not in matched_gt_indices:
            fn_notes.append(g_event)
    
    # Calculate metrics
    tp = len(tp_notes)
    fp = len(fp_notes)
    fn = len(fn_notes)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create piano roll visualization
    plot = _create_evaluation_piano_roll(tp_notes, fp_notes, fn_notes, gt, pred, save_path=save_path)
    
    return EvaluationResult(tp, fp, fn, precision, recall, f1_score, plot)

def _create_evaluation_piano_roll(tp_notes, fp_notes, fn_notes, gt, pred, 
                                  save_path: str | None = None, note_height=0.8):
    """
    Create a piano roll visualization showing tp/fp/fn with color coding.
    Automatically adjusts figure size based on melody length and duration.
    """
    # Get all notes to determine plot range
    all_notes = list(gt.events) + list(pred.events)
    if not all_notes:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No notes to display', ha='center', va='center', transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Get time and note ranges
    start_time = min(event.time for event in all_notes)
    end_time = max(event.time for event in all_notes)
    min_note = min(event.note for event in all_notes)
    max_note = max(event.note for event in all_notes)
    
    # Calculate optimal figure size
    duration = end_time - start_time
    note_range = max_note - min_note
    total_events = len(all_notes)
    
    # Auto-adjust width based on duration and number of events
    # Base width on duration, with constraints
    base_width = max(8, min(30, duration * 2))  # 2 seconds per inch, min 8", max 30"
    
    # Adjust width based on event density
    if total_events > 0:
        event_density = total_events / max(duration, 1)
        if event_density > 10:  # High density
            base_width = max(base_width, total_events * 0.2)  # More space for dense melodies
        elif event_density < 2:  # Low density
            base_width = max(8, base_width * 0.8)  # Less space for sparse melodies
    
    # Auto-adjust height based on note range
    base_height = max(6, min(15, note_range * 0.3 + 4))  # Scale with note range
    
    figsize = (base_width, base_height)
    
    # Add padding
    time_padding = (end_time - start_time) * 0.05
    note_padding = 3
    
    start_time = max(0, start_time - time_padding)
    end_time = end_time + time_padding
    min_note = max(0, min_note - note_padding)
    max_note = min(127, max_note + note_padding)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(min_note - 0.5, max_note + 0.5)
    
    # Color scheme
    colors = {
        'tp': '#2E8B57',    # Sea Green - True Positives
        'fp': '#DC143C',    # Crimson - False Positives  
        'fn': '#4169E1'     # Royal Blue - False Negatives
    }
    
    # Draw notes with color coding
    def draw_note(event, color, label, alpha=0.8):
        # For visualization, we'll show notes as short rectangles
        # Scale duration based on figure width to ensure visibility
        duration = max(0.05, (end_time - start_time) / 100)  # Adaptive duration
        rect = Rectangle(
            (event.time, event.note - note_height/2),
            duration,
            note_height,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=alpha,
            label=label
        )
        ax.add_patch(rect)
    
    # Draw all note categories
    labels_added = set()

    for event in fn_notes:
        label = 'False Negatives' if 'False Negatives' not in labels_added else ""
        if label: labels_added.add(label)
        draw_note(event, colors['fn'], label)
    
    for event in fp_notes:
        label = 'False Positives' if 'False Positives' not in labels_added else ""
        if label: labels_added.add(label)
        draw_note(event, colors['fp'], label)
    
    for event in tp_notes:
        label = 'True Positives' if 'True Positives' not in labels_added else ""
        if label: labels_added.add(label)
        draw_note(event, colors['tp'], label)
    
    # Set up y-axis with note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def get_note_name(note_number):
        octave = (note_number // 12) - 1
        note_name = note_names[note_number % 12]
        return f"{note_name}{octave}"
    
    # Filter y-axis to show fewer labels for better readability
    y_range = max_note - min_note
    step = max(1, y_range // 20)  # Aim for ~20 labels
    y_ticks = list(range(min_note, max_note + 1, step))
    y_labels = [get_note_name(note) for note in y_ticks]
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Set up x-axis with adaptive spacing
    time_range = end_time - start_time
    # Adjust x-axis tick spacing based on duration
    if time_range <= 10:
        x_step = 1
    elif time_range <= 60:
        x_step = 5
    elif time_range <= 300:
        x_step = 30
    else:
        x_step = 60
    
    x_ticks = np.arange(start_time, end_time + x_step, x_step)
    ax.set_xticks(x_ticks)
    
    # Format x-axis labels based on duration
    if time_range <= 60:
        ax.set_xticklabels([f"{t:.1f}s" for t in x_ticks], fontsize=10)
    else:
        ax.set_xticklabels([f"{int(t//60)}:{int(t%60):02d}" for t in x_ticks], fontsize=10)
        ax.set_xlabel('Time (mm:ss)', fontsize=12, fontweight='bold')
    
    if time_range <= 60:
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    
    # Customize the plot
    ax.grid(True, alpha=0.3, axis='both')
    ax.set_ylabel('Notes', fontsize=12, fontweight='bold')
    
    # Add title with duration info
    title = f'Melody Evaluation: Ground Truth vs Prediction\n'
    title += f'Duration: {duration:.1f}s, Events: {total_events}, Figure: {figsize[0]:.1f}"x{figsize[1]:.1f}"'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    if labels_added:
        ax.legend(loc='upper right', framealpha=0.9)
    
    # Add metrics text
    tp_count = len(tp_notes)
    fp_count = len(fp_notes)
    fn_count = len(fn_notes)
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"TP: {tp_count}, FP: {fp_count}, FN: {fn_count}\n"
    metrics_text += f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
    
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plot saved to: {save_path}")
    
    return fig

def display_evaluation_result(result: EvaluationResult, save_path: str | None = None, show_plot: bool = True) -> None:
    """
    Display the evaluation result with metrics and piano roll visualization.
    
    Args:
        result: EvaluationResult object containing metrics and plot
        save_path: Optional additional path to save the plot (if not already saved)
        show_plot: Whether to display the plot interactively
    """
    print("Melody Evaluation Results:")
    print(f"  True Positives:  {result.tp}")
    print(f"  False Positives: {result.fp}")
    print(f"  False Negatives: {result.fn}")
    print(f"  Precision:       {result.precision:.3f}")
    print(f"  Recall:          {result.recall:.3f}")
    print(f"  F1-Score:        {result.f1_score:.3f}")
    print()
    
    if save_path:
        result.plot.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plot saved to: {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()

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
    
    # Example of melody evaluation
    gt_file = "mid_files/weiyi_melody.mid"
    pred_file = "mid_files/weiyi_transcription.mid"
    
    if os.path.exists(gt_file) and os.path.exists(pred_file):
        from pathlib import Path
        
        print(f"\nEvaluating melody comparison:")
        print(f"  Ground Truth: {gt_file}")
        print(f"  Prediction:   {pred_file}")
        
        # Load melodies
        gt_melody = Melody(Path(gt_file))
        pred_melody = Melody(Path(pred_file))
        
        # Evaluate with auto-sized figure and save to file
        save_path = "melody_evaluation_result.png"
        result = evaluate_melody(gt_melody, pred_melody, tolerance=0.1, save_path=save_path)
        
        # Display results (plot already saved)
        display_evaluation_result(result, show_plot=True)
    else:
        print("Required MIDI files not found for evaluation example")

