
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from dataclasses import dataclass

from exmel.sequence import MelodyLike, Melody, PerformanceLike, Performance
from exmel.alignment import Alignment
from exmel.io import PathLike

def plot_alignment(
    alignment: Alignment,
    ref_melody: MelodyLike,
    performance: PerformanceLike | None = None,
    ground_truth: MelodyLike | None = None,
    save_path: str | None = None
) -> None:
    """
    Visualize the alignment by showing the source of each match chunk.
    
    Creates a two-panel visualization:
    - Upper panel: Piano roll of the alignment (and the performance if provided)
    - Lower panel: Piano roll of the original melody
    - Connecting lines between matched chunks
    
    Args:
        alignment: Alignment object containing matches and aligned events
        ref_melody: Original melody (MelodyLike)
        performance: Performance (PerformanceLike)
        save_path: Optional path to save the visualization
    """
    # Convert ref_melody to Melody object if needed
    if not isinstance(ref_melody, Melody):
        ref_melody = Melody(ref_melody)
    
    # Convert performance to Performance object if needed
    performance_events = []
    if performance is not None:
        if not isinstance(performance, Performance):
            performance = Performance(performance)
        performance_events = performance.global_events
    
    # Convert ground truth to Melody object if needed
    ground_truth_melody = None
    if ground_truth is not None:
        if not isinstance(ground_truth, Melody):
            ground_truth_melody = Melody(ground_truth)
        else:
            ground_truth_melody = ground_truth
    
    # Get all events to determine plot ranges
    alignment_events = alignment.events
    melody_events = ref_melody.events
    ground_truth_events = ground_truth_melody.events if ground_truth_melody else []
    
    if not alignment_events and not melody_events and not performance_events and not ground_truth_events:
        print("No events to visualize")
        return
    
    # Calculate time and note ranges
    all_times = []
    all_notes = []
    
    if alignment_events:
        all_times.extend([event.time for event in alignment_events])
        all_notes.extend([event.note for event in alignment_events])
    
    if melody_events:
        all_times.extend([event.time for event in melody_events])
        all_notes.extend([event.note for event in melody_events])
    
    if performance_events:
        all_times.extend([event.time for event in performance_events])
        all_notes.extend([event.note for event in performance_events])
    
    if ground_truth_events:
        all_times.extend([event.time for event in ground_truth_events])
        all_notes.extend([event.note for event in ground_truth_events])
    
    time_min, time_max = min(all_times), max(all_times)
    note_min, note_max = min(all_notes), max(all_notes)
    
    # Add padding
    time_padding = (time_max - time_min) * 0.05
    note_padding = 2
    time_min = max(0, time_min - time_padding)
    time_max = time_max + time_padding
    note_min = max(0, note_min - note_padding)
    note_max = min(127, note_max + note_padding)
    
    # Calculate figure size based on content
    duration = time_max - time_min
    note_range = note_max - note_min
    
    # Auto-size figure
    fig_width = max(10, min(20, duration * 1.5))
    fig_height = max(8, min(12, note_range * 0.2 + 6))
    
    # Create figure with two subplots
    fig, (ax_alignment, ax_melody) = plt.subplots(2, 1, figsize=(fig_width, fig_height), 
                                                   sharex=True, gridspec_kw={'hspace': 0.3})
    
    # Color scheme
    alignment_color = '#2E8B57'    # Sea Green
    melody_color = '#4169E1'       # Royal Blue
    performance_color = '#FF8C00'  # Dark Orange
    
    # Ground truth evaluation colors - standard and informative
    tp_color = '#228B22'    # Forest Green - True Positives (standard success color)
    fp_color = '#DC143C'    # Crimson - False Positives (standard error color)
    fn_color = '#4169E1'    # Royal Blue - False Negatives (standard missing color)
    
    # Generate distinct colors for each match
    import matplotlib.cm as cm
    num_matches = len(alignment.matches)
    if num_matches > 0:
        # Use a colormap to generate distinct colors for each match
        colormap = plt.colormaps.get_cmap('tab10' if num_matches <= 10 else 'tab20')
        match_colors = [colormap(i / max(1, num_matches - 1)) for i in range(num_matches)]
    else:
        match_colors = []
    
    note_height = 0.6
    
    # Ground truth evaluation if provided
    tp_events, fp_events, fn_events = [], [], []
    if ground_truth_melody:
        # Evaluate alignment against ground truth (similar to evaluate_melody)
        tolerance = 0.1  # Default tolerance
        modulo = True    # Default modulo
        
        # Keep original events for visualization
        gt_original = ground_truth_melody
        pred_original = Melody(alignment_events)
        
        # Create moduloed versions for matching logic only
        gt_modulo = gt_original % 12 if modulo else gt_original
        pred_modulo = pred_original % 12 if modulo else pred_original
        
        # Find true positives and mark matched gt notes
        matched_gt_indices = set()
        for i, p_event_mod in enumerate(pred_modulo):
            matched = False
            for j, g_event_mod in enumerate(gt_modulo):
                if p_event_mod.note == g_event_mod.note:
                    if abs(p_event_mod.time - g_event_mod.time) < tolerance:
                        # Store original events for visualization
                        tp_events.append(pred_original[i])
                        matched_gt_indices.add(j)
                        matched = True
                        break
            if not matched:
                # Store original event for visualization
                fp_events.append(pred_original[i])
        
        # Find false negatives (unmatched gt notes)
        for j, g_event_mod in enumerate(gt_modulo):
            if j not in matched_gt_indices:
                # Store original event for visualization
                fn_events.append(gt_original[j])
    
    # Helper function to calculate match statistics
    def get_match_stats(match):
        duration = match.end - match.start
        num_events = len(match.events)
        num_misses = match.sum_miss
        sum_error = match.sum_error
        score = match.score
        
        # Calculate average velocity from events
        if match.events:
            avg_velocity = sum(event.velocity for event in match.events) / len(match.events)
        else:
            avg_velocity = 0
        
        return {
            'duration': duration,
            'num_events': num_events,
            'num_misses': num_misses,
            'score': score,
            'avg_velocity': avg_velocity,
            'sum_error': sum_error
        }
    
    # Helper function to draw piano roll
    def draw_piano_roll(ax, events, color, title, show_matches=False, ground_truth_eval=None):
        if not events:
            ax.text(0.5, 0.5, 'No events', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Highlight match regions if requested
        if show_matches:
            for i, match in enumerate(alignment.matches):
                match_color = match_colors[i] if i < len(match_colors) else '#DC143C'
                # Create a background highlight for the match region
                match_rect = Rectangle(
                    (match.start, note_min),
                    match.end - match.start,
                    note_max - note_min,
                    facecolor=match_color,
                    alpha=0.2,
                    edgecolor=match_color,
                    linewidth=2,
                    linestyle='--'
                )
                ax.add_patch(match_rect)
                
                # Get match statistics
                stats = get_match_stats(match)
                
                # Create detailed match label with statistics
                match_text = f'M{i+1}\n'
                match_text += f'Score: {stats["score"]:.2f}\n'
                match_text += f'Dur: {stats["duration"]:.1f}s\n'
                match_text += f'Vel: {stats["avg_velocity"]:.0f}\n'
                match_text += f'Events: {stats["num_events"]}\n'
                match_text += f'Misses: {stats["num_misses"]}\n'
                match_text += f"Err: {stats["sum_error"]:.2f}"
                
                # Position the label at the center of the match region
                label_x = match.start + (match.end - match.start) / 2
                label_y = note_max - 80
                
                ax.text(label_x, label_y, match_text, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                       fontsize=7, fontweight='bold')

        # Draw events with ground truth evaluation coloring if provided
        if ground_truth_eval:
            tp_events, fp_events, fn_events = ground_truth_eval
            
            # Enhanced visual settings for better distinction
            enhanced_note_height = note_height * 1.2  # Make events taller
            enhanced_duration_factor = 120  # Make events wider
            
            # Draw events with evaluation colors - enhanced visibility
            # Draw FN events first (background layer)
            for event in fn_events:
                duration = max(0.08, (time_max - time_min) / enhanced_duration_factor)
                # Add subtle glow effect with a larger, more transparent background rectangle
                glow_rect = Rectangle(
                    (event.time - duration*0.1, event.note - enhanced_note_height*0.7),
                    duration*1.2,
                    enhanced_note_height*1.4,
                    facecolor=fn_color,
                    edgecolor='none',
                    alpha=0.3
                )
                ax.add_patch(glow_rect)
                
                # Main event rectangle
                rect = Rectangle(
                    (event.time, event.note - enhanced_note_height/2),
                    duration,
                    enhanced_note_height,
                    facecolor=fn_color,
                    edgecolor='darkblue',
                    linewidth=0.1,
                    alpha=0.95
                )
                ax.add_patch(rect)
            
            # Draw FP events (middle layer)
            for event in fp_events:
                duration = max(0.08, (time_max - time_min) / enhanced_duration_factor)
                # Add subtle glow effect
                glow_rect = Rectangle(
                    (event.time - duration*0.1, event.note - enhanced_note_height*0.7),
                    duration*1.2,
                    enhanced_note_height*1.4,
                    facecolor=fp_color,
                    edgecolor='none',
                    alpha=0.3
                )
                ax.add_patch(glow_rect)
                
                # Main event rectangle
                rect = Rectangle(
                    (event.time, event.note - enhanced_note_height/2),
                    duration,
                    enhanced_note_height,
                    facecolor=fp_color,
                    edgecolor='darkred',
                    linewidth=0.1,
                    alpha=0.95
                )
                ax.add_patch(rect)
            
            # Draw TP events last (foreground layer)
            for event in tp_events:
                duration = max(0.08, (time_max - time_min) / enhanced_duration_factor)
                # Add subtle glow effect
                glow_rect = Rectangle(
                    (event.time - duration*0.1, event.note - enhanced_note_height*0.7),
                    duration*1.2,
                    enhanced_note_height*1.4,
                    facecolor=tp_color,
                    edgecolor='none',
                    alpha=0.3
                )
                ax.add_patch(glow_rect)
                
                # Main event rectangle
                rect = Rectangle(
                    (event.time, event.note - enhanced_note_height/2),
                    duration,
                    enhanced_note_height,
                    facecolor=tp_color,
                    edgecolor='darkgreen',
                    linewidth=0.1,
                    alpha=0.95
                )
                ax.add_patch(rect)
        else:
            # Draw events with default color
            for event in events:
                duration = max(0.05, (time_max - time_min) / 100)  # Adaptive duration for visibility
                rect = Rectangle(
                    (event.time, event.note - note_height/2),
                    duration,
                    note_height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )
                ax.add_patch(rect)
        
        # Set limits and grid
        ax.set_xlim(time_min, time_max)
        ax.set_ylim(note_min - 0.5, note_max + 0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Draw alignment panel (upper)
    title_parts = [f'Alignment ({len(alignment_events)} events)']
    if performance_events:
        title_parts.append(f'Performance ({len(performance_events)} events)')
    if ground_truth_melody:
        title_parts.append(f'Ground Truth Evaluation')
    alignment_title = ' + '.join(title_parts)

    # Draw performance events in the alignment panel if provided
    if performance_events:
        for event in performance_events:
            duration = max(0.02, (time_max - time_min) / 150)  # Smaller duration for performance events
            rect = Rectangle(
                (event.time, event.note - note_height/4),  # Smaller height
                duration,
                note_height/2,  # Half the height of alignment events
                facecolor=performance_color,
                edgecolor='black',
                linewidth=0.3,
                alpha=0.4  # More transparent
            )
            ax_alignment.add_patch(rect)
    
    # Prepare ground truth evaluation data for alignment panel
    ground_truth_eval_data = None
    if ground_truth_melody:
        ground_truth_eval_data = (tp_events, fp_events, fn_events)
    
    draw_piano_roll(ax_alignment, alignment_events, alignment_color, 
                   alignment_title, show_matches=True, ground_truth_eval=ground_truth_eval_data)
    
    # Draw discarded matches in a minimal way
    if alignment.discarded_matches:
        # Preprocess discarded matches to merge overlapping ones into chunks
        def merge_overlapping_discards(discarded_matches):
            if not discarded_matches:
                return []
            
            # Sort by start time
            sorted_matches = sorted(discarded_matches, key=lambda m: m.start)
            chunks = []
            current_chunk = [sorted_matches[0]]
            
            for match in sorted_matches[1:]:
                # Check if current match overlaps with the last match in current chunk
                last_match = current_chunk[-1]
                if match.start <= last_match.end:
                    # Overlapping - extend current chunk
                    current_chunk.append(match)
                else:
                    # No overlap - finalize current chunk and start new one
                    chunks.append(current_chunk)
                    current_chunk = [match]
            
            # Don't forget the last chunk
            chunks.append(current_chunk)
            return chunks
        
        discarded_chunks = merge_overlapping_discards(alignment.discarded_matches)
        
        for i, chunk in enumerate(discarded_chunks):
            # Calculate chunk boundaries
            chunk_start = min(match.start for match in chunk)
            chunk_end = max(match.end for match in chunk)
            
            # Calculate statistics for the chunk
            scores = [match.score for match in chunk]
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            
            # Create a single background highlight for the entire chunk
            discarded_rect = Rectangle(
                (chunk_start, note_min),
                chunk_end - chunk_start,
                note_max - note_min,
                facecolor='gray',
                alpha=0.15,  # Slightly more visible for chunks
                edgecolor='gray',
                linewidth=1,
                linestyle=':'
            )
            ax_alignment.add_patch(discarded_rect)
            
            # Add a compact label with aggregated information
            label_x = chunk_start + (chunk_end - chunk_start) / 2
            label_y = note_max - 2
            label_text = f'D{i+1}({len(chunk)})\n{min_score:.1f}-{avg_score:.1f}-{max_score:.1f}'
            ax_alignment.text(label_x, label_y, label_text, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.8),
                           fontsize=6, fontweight='bold')
    
    
    
    # Draw melody panel (lower)
    draw_piano_roll(ax_melody, melody_events, melody_color, 
                   f'Original Melody ({len(melody_events)} events)')
    
    # Highlight corresponding melody regions for each match
    for i, match in enumerate(alignment.matches):
        match_color = match_colors[i] if i < len(match_colors) else '#DC143C'
        # Create a background highlight for the melody source region
        melody_rect = Rectangle(
            (match.melody_start, note_min),
            match.melody_end - match.melody_start,
            note_max - note_min,
            facecolor=match_color,
            alpha=0.2,
            edgecolor=match_color,
            linewidth=2,
            linestyle='-'
        )
        ax_melody.add_patch(melody_rect)
        
        # Add simplified match number label in melody panel
        ax_melody.text(match.melody_start + (match.melody_end - match.melody_start) / 2, note_min + 1,
                      f'M{i+1}', ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                      fontsize=8, fontweight='bold')
    
    # Draw connecting lines between matches using ConnectionPatch for accurate alignment
    for i, match in enumerate(alignment.matches):
        line_color = match_colors[i] if i < len(match_colors) else '#FF6347'
        
        # Create connection lines that precisely connect match boundaries
        # Line from melody start to alignment start
        con_start = ConnectionPatch(
            xyA=(match.melody_start, note_max), coordsA='data', axesA=ax_melody,
            xyB=(match.start, note_min), coordsB='data', axesB=ax_alignment,
            arrowstyle='-', color=line_color, linewidth=2, alpha=0.7
        )
        fig.add_artist(con_start)
        
        # Line from melody end to alignment end
        con_end = ConnectionPatch(
            xyA=(match.melody_end, note_max), coordsA='data', axesA=ax_melody,
            xyB=(match.end, note_min), coordsB='data', axesB=ax_alignment,
            arrowstyle='-', color=line_color, linewidth=2, alpha=0.7
        )
        fig.add_artist(con_end)
    
    # Set up note names for y-axis
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def get_note_name(note_number):
        octave = (note_number // 12) - 1
        note_name = note_names[note_number % 12]
        return f"{note_name}{octave}"
    
    # Set y-axis ticks for both panels
    y_step = max(1, (note_max - note_min) // 15)  # Aim for ~15 labels
    y_ticks = list(range(note_min, note_max + 1, y_step))
    y_labels = [get_note_name(note) for note in y_ticks]
    
    for ax in [ax_alignment, ax_melody]:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_ylabel('Notes', fontsize=10, fontweight='bold')
    
    # Set up x-axis for both panels
    time_range = time_max - time_min
    if time_range <= 10:
        x_step = 1
    elif time_range <= 60:
        x_step = 5
    else:
        x_step = 10
    
    x_ticks = np.arange(time_min, time_max + x_step, x_step)
    
    # Set ticks for both panels
    ax_alignment.set_xticks(x_ticks)
    ax_melody.set_xticks(x_ticks)
    
    # Format labels based on time range
    if time_range <= 60:
        x_labels = [f"{t:.1f}s" for t in x_ticks]
        xlabel_text = 'Time (seconds)'
    else:
        x_labels = [f"{int(t//60)}:{int(t%60):02d}" for t in x_ticks]
        xlabel_text = 'Time (mm:ss)'
    
    # Apply labels to both panels
    ax_alignment.set_xticklabels(x_labels, fontsize=10)
    ax_melody.set_xticklabels(x_labels, fontsize=10)
    
    # Make sure ticks are visible on both panels (override sharex behavior)
    ax_alignment.tick_params(axis='x', which='both', labelbottom=True)
    ax_melody.tick_params(axis='x', which='both', labelbottom=True)
    
    # Add x-axis labels to both panels
    ax_alignment.set_xlabel(xlabel_text, fontsize=10, fontweight='bold')
    ax_melody.set_xlabel(xlabel_text, fontsize=10, fontweight='bold')
    
    # Add overall title and metadata
    title = f'Melody Alignment Visualization'
    if performance_events:
        title += f' (with Performance)'
    title += f'\nMatches: {len(alignment.matches)}, Score: {alignment.score:.2f}, '
    title += f'Duration: {duration:.1f}s'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    
    # Add legend - simplified based on what's available
    legend_elements = []
    
    # Case 1: Ground truth is provided - show only TP/FP/FN, performance, and discarded chunks
    if ground_truth_melody:
        legend_elements.extend([
            Line2D([0], [0], color=tp_color, lw=4, label='True Positives'),
            Line2D([0], [0], color=fp_color, lw=4, label='False Positives'),
            Line2D([0], [0], color=fn_color, lw=4, label='False Negatives'),
        ])
        
        # Add performance if present
        if performance_events:
            legend_elements.append(
                Line2D([0], [0], color=performance_color, lw=2, alpha=0.4, label='Performance')
            )
        
        # Add discarded matches if present
        if alignment.discarded_matches:
            legend_elements.append(
                Line2D([0], [0], color='gray', lw=1, linestyle=':', alpha=0.5, label='Discarded Chunks')
            )
    
    # Case 2: No ground truth but performance is provided - show alignment, performance, and discarded chunks
    elif performance_events:
        legend_elements.append(
            Line2D([0], [0], color=alignment_color, lw=4, label='Alignment')
        )
        legend_elements.append(
            Line2D([0], [0], color=performance_color, lw=2, alpha=0.4, label='Performance')
        )
        
        # Add discarded matches if present
        if alignment.discarded_matches:
            legend_elements.append(
                Line2D([0], [0], color='gray', lw=1, linestyle=':', alpha=0.5, label='Discarded Chunks')
            )
    
    # Case 3: Neither ground truth nor performance - show only alignment and discarded chunks
    else:
        legend_elements.append(
            Line2D([0], [0], color=alignment_color, lw=4, label='Alignment')
        )
        
        # Add discarded matches if present
        if alignment.discarded_matches:
            legend_elements.append(
                Line2D([0], [0], color='gray', lw=1, linestyle=':', alpha=0.5, label='Discarded Chunks')
            )
    
    ax_alignment.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add statistics text
    stats_text = f"Alignment Stats:\n"

    if ground_truth_melody:
        tp_count = len(tp_events)
        fp_count = len(fp_events)
        fn_count = len(fn_events)
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        stats_text += f"TP: {tp_count}, FP: {fp_count}, FN: {fn_count}\n"
        stats_text += f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n"
    stats_text += f"Matches: {len(alignment.matches)}\n"
    stats_text += f"Score: {alignment.score:.3f}\n"
    stats_text += f"Sum Miss: {alignment.sum_miss}\n"
    stats_text += f"Sum Error: {alignment.sum_error:.3f}"
    
    ax_melody.text(0.02, 0.98, stats_text, transform=ax_melody.transAxes,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.1),
                  fontsize=9, fontweight='bold')
    
    # plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Alignment visualization saved to: {save_path}")
    else:
        plt.show()
    

@dataclass(frozen=True, slots=True)
class EvaluationResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    plot: Figure | None = None

def evaluate_melody(
    gt: MelodyLike,
    pred: MelodyLike,
    tolerance: float = 0.1,
    modulo: bool = True,
    plot: bool = True,
    save_path: str | None = None,
) -> EvaluationResult:
    """
    Evaluate the accuracy of a predicted melody against a ground truth melody.
    Creates a piano roll visualization with color coding for tp/fp/fn.
    
    Args:
        gt: Ground truth melody
        pred: Predicted melody
        tolerance: Time tolerance for matching notes (seconds)
        modulo: Whether to use modulo 12 for note matching
        plot: Whether to plot the evaluation result
        save_path: Optional path to save the visualization plot
    """
    gt = Melody(gt)
    pred = Melody(pred)

    if modulo:
        gt %= 12
        pred %= 12
    
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
    if plot:
        fig = _create_evaluation_piano_roll(tp_notes, fp_notes, fn_notes, gt, pred, save_path=save_path)
    else:
        fig = None
    
    return EvaluationResult(tp, fp, fn, precision, recall, f1_score, fig)

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
    
    if save_path and result.plot:
        result.plot.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plot saved to: {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()

def plot_melody(melody: MelodyLike, save_path: PathLike | None = None, show_plot: bool = True) -> None:
    """
    Plot the melody with a piano roll visualization.
    
    Args:
        melody: Melody to plot
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot interactively
    """
    melody = Melody(melody)
    events = melody.events
    
    if not events:
        print("No events to visualize in melody")
        return
    
    # Calculate time and note ranges
    start_time = min(event.time for event in events)
    end_time = max(event.time for event in events)
    min_note = min(event.note for event in events)
    max_note = max(event.note for event in events)
    
    # Calculate optimal figure size
    duration = end_time - start_time
    note_range = max_note - min_note
    total_events = len(events)
    
    # Auto-adjust width based on duration and number of events
    base_width = max(8, min(30, duration * 1.5))  # 1.5 seconds per inch, min 8", max 30"
    
    # Adjust width based on event density
    if total_events > 0:
        event_density = total_events / max(duration, 1)
        if event_density > 10:  # High density
            base_width = max(base_width, total_events * 0.15)
        elif event_density < 2:  # Low density
            base_width = max(8, base_width * 0.8)
    
    # Auto-adjust height based on note range
    base_height = max(6, min(15, note_range * 0.3 + 4))
    
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
    melody_color = '#4169E1'  # Royal Blue
    note_height = 0.8
    
    # Draw notes
    for event in events:
        # Scale duration based on figure width to ensure visibility
        duration_vis = max(0.05, (end_time - start_time) / 100)
        rect = Rectangle(
            (event.time, event.note - note_height/2),
            duration_vis,
            note_height,
            facecolor=melody_color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(rect)
    
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
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    else:
        ax.set_xticklabels([f"{int(t//60)}:{int(t%60):02d}" for t in x_ticks], fontsize=10)
        ax.set_xlabel('Time (mm:ss)', fontsize=12, fontweight='bold')
    
    # Customize the plot
    ax.grid(True, alpha=0.3, axis='both')
    ax.set_ylabel('Notes', fontsize=12, fontweight='bold')
    
    # Add title with metadata
    title = f'Melody Visualization\n'
    title += f'Duration: {duration:.1f}s, Events: {total_events}, Figure: {figsize[0]:.1f}"x{figsize[1]:.1f}"'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics text
    stats_text = f"Melody Stats:\n"
    stats_text += f"Events: {total_events}\n"
    stats_text += f"Duration: {duration:.1f}s\n"
    stats_text += f"Note Range: {min_note}-{max_note}\n"
    stats_text += f"Event Density: {total_events/max(duration, 1):.1f} events/s"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Melody plot saved to: {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()

def plot_performance(performance: PerformanceLike, save_path: PathLike | None = None, show_plot: bool = True) -> None:
    """
    Plot the performance with a piano roll visualization.
    
    Args:
        performance: Performance to plot
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot interactively
    """
    performance = Performance(performance)
    events = performance.global_events
    
    if not events:
        print("No events to visualize in performance")
        return
    
    # Calculate time and note ranges
    start_time = min(event.time for event in events)
    end_time = max(event.time for event in events)
    min_note = min(event.note for event in events)
    max_note = max(event.note for event in events)
    
    # Calculate optimal figure size
    duration = end_time - start_time
    note_range = max_note - min_note
    total_events = len(events)
    
    # Auto-adjust width based on duration and number of events
    base_width = max(8, min(30, duration * 1.5))  # 1.5 seconds per inch, min 8", max 30"
    
    # Adjust width based on event density
    if total_events > 0:
        event_density = total_events / max(duration, 1)
        if event_density > 10:  # High density
            base_width = max(base_width, total_events * 0.15)
        elif event_density < 2:  # Low density
            base_width = max(8, base_width * 0.8)
    
    # Auto-adjust height based on note range
    base_height = max(6, min(15, note_range * 0.3 + 4))
    
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
    performance_color = '#FF8C00'  # Dark Orange
    note_height = 0.8
    
    # Draw notes
    for event in events:
        # Scale duration based on figure width to ensure visibility
        duration_vis = max(0.05, (end_time - start_time) / 100)
        rect = Rectangle(
            (event.time, event.note - note_height/2),
            duration_vis,
            note_height,
            facecolor=performance_color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(rect)
    
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
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    else:
        ax.set_xticklabels([f"{int(t//60)}:{int(t%60):02d}" for t in x_ticks], fontsize=10)
        ax.set_xlabel('Time (mm:ss)', fontsize=12, fontweight='bold')
    
    # Customize the plot
    ax.grid(True, alpha=0.3, axis='both')
    ax.set_ylabel('Notes', fontsize=12, fontweight='bold')
    
    # Add title with metadata
    title = f'Performance Visualization\n'
    title += f'Duration: {duration:.1f}s, Events: {total_events}, Figure: {figsize[0]:.1f}"x{figsize[1]:.1f}"'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics text
    stats_text = f"Performance Stats:\n"
    stats_text += f"Events: {total_events}\n"
    stats_text += f"Duration: {duration:.1f}s\n"
    stats_text += f"Note Range: {min_note}-{max_note}\n"
    stats_text += f"Event Density: {total_events/max(duration, 1):.1f} events/s"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to: {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()