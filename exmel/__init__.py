from exmel.sequence import Melody, Performance
from exmel.event import MelEvent_, MidiEvent_, MidiEvent, MelEvent
from exmel.alignment import scan, align, AlignConfig
from exmel.vis import (
    evaluate_melody, plot_alignment,
    plot_melody, plot_performance
)
from exmel.io import extract_original_events, save_melody
from exmel.score import (
    sum_velocity, weighted_sum_velocity,
    duration_adjusted_weighted_sum_velocity,
)