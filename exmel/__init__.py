from exmel.sequence import Melody, Performance
from exmel.event import MelEvent_, MidiEvent_, MidiEvent, MelEvent
from exmel.alignment import scan, align, AlignConfig
from exmel.eval import (
    evaluate_melody, plot_alignment
)
from exmel.io import extract_original_events, melody_to_midi, melody_to_note
from exmel.score import (
    sum_velocity, weighted_sum_velocity,
    duration_adjusted_weighted_sum_velocity,
)
from exmel.pipeline import eval_pipeline, inference_pipeline, Dataset