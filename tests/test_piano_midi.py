import unittest
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exmel.adt import _PianoMidi, PianoMidi, MelEvent, MidiEvent

class TestPianoMidiQuerying(unittest.TestCase):
    """Comprehensive tests for PianoMidi querying methods."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Create a mock PianoMidi instance with test data
        self.piano_midi = Mock(spec=_PianoMidi)
        self.piano_midi.events_by_note = {
            60: [(1.0, 80), (2.5, 90), (4.0, 85), (6.0, 75)],  # C4
            64: [(1.5, 70), (3.0, 80), (5.0, 85)],  # E4
            72: [(0.5, 95), (2.0, 88), (3.5, 92)],  # C5
            48: [(1.2, 75)],  # C3
        }
        
        # Bind the methods to the mock instance
        self.piano_midi.right_nearest = _PianoMidi.right_nearest.__get__(self.piano_midi, _PianoMidi)
        self.piano_midi.left_nearest = _PianoMidi.left_nearest.__get__(self.piano_midi, _PianoMidi)
        self.piano_midi.nearest = _PianoMidi.nearest.__get__(self.piano_midi, _PianoMidi)
        self.piano_midi.right_nearest_multi = _PianoMidi.right_nearest_multi.__get__(self.piano_midi, _PianoMidi)
        self.piano_midi.left_nearest_multi = _PianoMidi.left_nearest_multi.__get__(self.piano_midi, _PianoMidi)
        self.piano_midi.nearest_multi = _PianoMidi.nearest_multi.__get__(self.piano_midi, _PianoMidi)

    def test_right_nearest_basic(self):
        """Test basic right_nearest functionality."""
        # Test finding event to the right
        result = self.piano_midi.right_nearest(MelEvent(2.0, 60))
        self.assertEqual(result, MidiEvent(2.5, 60, 90))
        
        # Test finding event to the right when query time matches an event
        result = self.piano_midi.right_nearest(MelEvent(2.5, 60))
        self.assertEqual(result, MidiEvent(4.0, 60, 85))

    def test_right_nearest_edge_cases(self):
        """Test right_nearest edge cases."""
        # Test when no events exist to the right
        result = self.piano_midi.right_nearest(MelEvent(7.0, 60))
        self.assertIsNone(result)
        
        # Test when query time is exactly at the last event
        result = self.piano_midi.right_nearest(MelEvent(6.0, 60))
        self.assertIsNone(result)
        
        # Test when note doesn't exist
        result = self.piano_midi.right_nearest(MelEvent(2.0, 99))
        self.assertIsNone(result)
        
        # Test when query time is before first event
        result = self.piano_midi.right_nearest(MelEvent(0.5, 60))
        self.assertEqual(result, MidiEvent(1.0, 60, 80))

    def test_left_nearest_basic(self):
        """Test basic left_nearest functionality."""
        # Test finding event to the left
        result = self.piano_midi.left_nearest(MelEvent(5.0, 60))
        self.assertEqual(result, MidiEvent(4.0, 60, 85))
        
        # Test finding event to the left when query time matches an event
        result = self.piano_midi.left_nearest(MelEvent(4.0, 60))
        self.assertEqual(result, MidiEvent(time=2.5, note=60, velocity=90))

    def test_left_nearest_edge_cases(self):
        """Test left_nearest edge cases."""
        # Test when no events exist to the left
        result = self.piano_midi.left_nearest(MelEvent(0.5, 60))
        self.assertIsNone(result)
        
        # Test when query time is exactly at the first event
        result = self.piano_midi.left_nearest(MelEvent(1.0, 60))
        self.assertIsNone(result)
        
        # Test when note doesn't exist
        result = self.piano_midi.left_nearest(MelEvent(2.0, 99))
        self.assertIsNone(result)
        
        # Test when query time is after last event
        result = self.piano_midi.left_nearest(MelEvent(7.0, 60))
        self.assertEqual(result, MidiEvent(6.0, 60, 75))

    def test_nearest_basic(self):
        """Test basic nearest functionality."""
        # Test finding nearest event when query is between two events
        result = self.piano_midi.nearest(MelEvent(3.0, 60))
        self.assertEqual(result, MidiEvent(2.5, 60, 90))  # 2.5 is closer than 4.0
        
        # Test finding nearest event when query is closer to later event
        result = self.piano_midi.nearest(MelEvent(3.5, 60))
        self.assertEqual(result, MidiEvent(4.0, 60, 85))  # 4.0 is closer than 2.5
        
        # Test finding nearest event when query exactly matches an event
        result = self.piano_midi.nearest(MelEvent(2.5, 60))
        self.assertEqual(result, MidiEvent(2.5, 60, 90))

    def test_nearest_edge_cases(self):
        """Test nearest edge cases."""
        # Test when note doesn't exist
        result = self.piano_midi.nearest(MelEvent(2.0, 99))
        self.assertIsNone(result)
        
        # Test when note has no events
        self.piano_midi.events_by_note[99] = []
        result = self.piano_midi.nearest(MelEvent(2.0, 99))
        self.assertIsNone(result)
        
        # Test when query is before first event
        result = self.piano_midi.nearest(MelEvent(0.5, 60))
        self.assertEqual(result, MidiEvent(1.0, 60, 80))
        
        # Test when query is after last event
        result = self.piano_midi.nearest(MelEvent(7.0, 60))
        self.assertEqual(result, MidiEvent(6.0, 60, 75))

    def test_right_nearest_multi_basic(self):
        """Test basic right_nearest_multi functionality."""
        # Test finding multiple events to the right
        result = self.piano_midi.right_nearest_multi(MelEvent(2.0, 60), 2)
        expected = [MidiEvent(2.5, 60, 90), MidiEvent(4.0, 60, 85)]
        self.assertEqual(result, expected)
        
        # Test finding more events than available
        result = self.piano_midi.right_nearest_multi(MelEvent(1.0, 60), 5)
        expected = [MidiEvent(2.5, 60, 90), MidiEvent(4.0, 60, 85), MidiEvent(6.0, 60, 75)]
        self.assertEqual(result, expected)

    def test_right_nearest_multi_edge_cases(self):
        """Test right_nearest_multi edge cases."""
        # Test when no events exist to the right
        result = self.piano_midi.right_nearest_multi(MelEvent(7.0, 60), 2)
        self.assertEqual(result, [])
        
        # Test when note doesn't exist
        result = self.piano_midi.right_nearest_multi(MelEvent(2.0, 99), 2)
        self.assertEqual(result, [])
        
        # Test with n=0
        result = self.piano_midi.right_nearest_multi(MelEvent(2.0, 60), 0)
        self.assertEqual(result, [])
        
        # Test with negative n
        result = self.piano_midi.right_nearest_multi(MelEvent(2.0, 60), -1)
        self.assertEqual(result, [])

    def test_left_nearest_multi_basic(self):
        """Test basic left_nearest_multi functionality."""
        # Test finding multiple events to the left
        result = self.piano_midi.left_nearest_multi(MelEvent(5.0, 60), 2)
        expected = [MidiEvent(4.0, 60, 85), MidiEvent(2.5, 60, 90)]
        self.assertEqual(result, expected)
        
        # Test finding more events than available
        result = self.piano_midi.left_nearest_multi(MelEvent(6.0, 60), 5)
        expected = [MidiEvent(4.0, 60, 85), MidiEvent(2.5, 60, 90), MidiEvent(1.0, 60, 80)]
        self.assertEqual(result, expected)

    def test_left_nearest_multi_edge_cases(self):
        """Test left_nearest_multi edge cases."""
        # Test when no events exist to the left
        result = self.piano_midi.left_nearest_multi(MelEvent(0.5, 60), 2)
        self.assertEqual(result, [])
        
        # Test when note doesn't exist
        result = self.piano_midi.left_nearest_multi(MelEvent(2.0, 99), 2)
        self.assertEqual(result, [])
        
        # Test with n=0
        result = self.piano_midi.left_nearest_multi(MelEvent(5.0, 60), 0)
        self.assertEqual(result, [])
        
        # Test with negative n
        result = self.piano_midi.left_nearest_multi(MelEvent(5.0, 60), -1)
        self.assertEqual(result, [])

    def test_nearest_multi_basic(self):
        """Test basic nearest_multi functionality."""
        # Test finding multiple nearest events
        result = self.piano_midi.nearest_multi(MelEvent(3.0, 60), 2)
        expected = [MidiEvent(2.5, 60, 90), MidiEvent(4.0, 60, 85)]
        self.assertEqual(result, expected)
        
        # Test finding more events than available
        result = self.piano_midi.nearest_multi(MelEvent(3.0, 60), 5)
        expected = [MidiEvent(2.5, 60, 90), MidiEvent(4.0, 60, 85), MidiEvent(1.0, 60, 80), MidiEvent(6.0, 60, 75)]
        self.assertEqual(result, expected)

    def test_nearest_multi_edge_cases(self):
        """Test nearest_multi edge cases."""
        # Test when note doesn't exist
        result = self.piano_midi.nearest_multi(MelEvent(2.0, 99), 2)
        self.assertEqual(result, [])
        
        # Test when note has no events
        result = self.piano_midi.nearest_multi(MelEvent(2.0, 99), 2)
        self.assertEqual(result, [])
        
        # Test with n=0
        result = self.piano_midi.nearest_multi(MelEvent(3.0, 60), 0)
        self.assertEqual(result, [])
        
        # Test with negative n
        result = self.piano_midi.nearest_multi(MelEvent(3.0, 60), -1)
        self.assertEqual(result, [])

    def test_nearest_multi_ordering(self):
        """Test that nearest_multi returns events in correct order (nearest to furthest)."""
        # Test with query time exactly between two events
        result = self.piano_midi.nearest_multi(MelEvent(3.2, 60), 3)
        expected = [MidiEvent(2.5, 60, 90), MidiEvent(4.0, 60, 85), MidiEvent(1.0, 60, 80)]
        self.assertEqual(result, expected)

    def test_cross_note_queries(self):
        """Test queries across different notes to ensure isolation."""
        # Test that queries for different notes don't interfere
        result_60 = self.piano_midi.right_nearest(MelEvent(2.0, 60))
        result_64 = self.piano_midi.right_nearest(MelEvent(2.0, 64))
        
        self.assertEqual(result_60, MidiEvent(2.5, 60, 90))
        self.assertEqual(result_64, MidiEvent(3.0, 64, 80))

    def test_empty_piano_midi(self):
        """Test behavior with empty PianoMidi instance."""
        empty_piano = Mock(spec=_PianoMidi)
        empty_piano.events_by_note = {}
        
        # Bind methods
        empty_piano.right_nearest = _PianoMidi.right_nearest.__get__(empty_piano, _PianoMidi)
        empty_piano.left_nearest = _PianoMidi.left_nearest.__get__(empty_piano, _PianoMidi)
        empty_piano.nearest = _PianoMidi.nearest.__get__(empty_piano, _PianoMidi)
        empty_piano.right_nearest_multi = _PianoMidi.right_nearest_multi.__get__(empty_piano, _PianoMidi)
        empty_piano.left_nearest_multi = _PianoMidi.left_nearest_multi.__get__(empty_piano, _PianoMidi)
        empty_piano.nearest_multi = _PianoMidi.nearest_multi.__get__(empty_piano, _PianoMidi)
        
        # All queries should return None or empty list
        self.assertIsNone(empty_piano.right_nearest(MelEvent(2.0, 60)))
        self.assertIsNone(empty_piano.left_nearest(MelEvent(2.0, 60)))
        self.assertIsNone(empty_piano.nearest(MelEvent(2.0, 60)))
        self.assertEqual(empty_piano.right_nearest_multi(MelEvent(2.0, 60), 2), [])
        self.assertEqual(empty_piano.left_nearest_multi(MelEvent(2.0, 60), 2), [])
        self.assertEqual(empty_piano.nearest_multi(MelEvent(2.0, 60), 2), [])

    def test_single_event_note(self):
        """Test behavior with notes that have only one event."""
        # Test with note 48 which has only one event
        result = self.piano_midi.right_nearest(MelEvent(1.0, 48))
        self.assertEqual(result, MidiEvent(1.2, 48, 75))
        
        result = self.piano_midi.left_nearest(MelEvent(3.0, 48))
        self.assertEqual(result, MidiEvent(1.2, 48, 75))
        
        result = self.piano_midi.left_nearest(MelEvent(1.2, 48))
        self.assertIsNone(result)

        result = self.piano_midi.right_nearest(MelEvent(1.2, 48))
        self.assertIsNone(result)

        result = self.piano_midi.nearest(MelEvent(1.2, 48))
        self.assertEqual(result, MidiEvent(1.2, 48, 75))

    def test_boundary_conditions(self):
        """Test boundary conditions for all methods."""
        # Test exactly at event times
        result = self.piano_midi.right_nearest(MelEvent(1.0, 60))
        self.assertEqual(result, MidiEvent(2.5, 60, 90))
        
        result = self.piano_midi.left_nearest(MelEvent(1.0, 60))
        self.assertIsNone(result)
        
        result = self.piano_midi.nearest(MelEvent(1.0, 60))
        self.assertEqual(result, MidiEvent(1.0, 60, 80))


class TestOptimizedPianoMidiQuerying(TestPianoMidiQuerying):
    """Comprehensive tests for OptimizedPianoMidi querying methods."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Create a mock OptimizedPianoMidi instance with test data
        self.piano_midi = Mock(spec=PianoMidi)
        self.piano_midi.events_by_note = {
            60: [(1.0, 80), (2.5, 90), (4.0, 85), (6.0, 75)],  # C4
            64: [(1.5, 70), (3.0, 80), (5.0, 85)],  # E4
            72: [(0.5, 95), (2.0, 88), (3.5, 92)],  # C5
            48: [(1.2, 75)],  # C3
        }
        
        # Bind the methods to the mock instance
        self.piano_midi.right_nearest = PianoMidi.right_nearest.__get__(self.piano_midi, PianoMidi)
        self.piano_midi.left_nearest = PianoMidi.left_nearest.__get__(self.piano_midi, PianoMidi)
        self.piano_midi.nearest = PianoMidi.nearest.__get__(self.piano_midi, PianoMidi)
        self.piano_midi.right_nearest_multi = PianoMidi.right_nearest_multi.__get__(self.piano_midi, PianoMidi)
        self.piano_midi.left_nearest_multi = PianoMidi.left_nearest_multi.__get__(self.piano_midi, PianoMidi)
        self.piano_midi.nearest_multi = PianoMidi.nearest_multi.__get__(self.piano_midi, PianoMidi)
        self.piano_midi.nearest_global = PianoMidi.nearest_global.__get__(self.piano_midi, PianoMidi)
        
        # Bind the private methods and reuse the actual optimization logic
        self.piano_midi._build_optimized_indices = PianoMidi._build_optimized_indices.__get__(self.piano_midi, PianoMidi)
        self.piano_midi._build_lookup_tables = PianoMidi._build_lookup_tables.__get__(self.piano_midi, PianoMidi)
        self.piano_midi._get_bucket = PianoMidi._get_bucket.__get__(self.piano_midi, PianoMidi)
        
        # Build the optimized indices using the actual class methods
        self.piano_midi._build_optimized_indices()

    def test_empty_piano_midi(self):
        """Test behavior with empty OptimizedPianoMidi instance."""
        empty_optimized = Mock(spec=PianoMidi)
        empty_optimized.events_by_note = {}
        
        # Bind methods
        empty_optimized.right_nearest = PianoMidi.right_nearest.__get__(empty_optimized, PianoMidi)
        empty_optimized.left_nearest = PianoMidi.left_nearest.__get__(empty_optimized, PianoMidi)
        empty_optimized.nearest = PianoMidi.nearest.__get__(empty_optimized, PianoMidi)
        empty_optimized.right_nearest_multi = PianoMidi.right_nearest_multi.__get__(empty_optimized, PianoMidi)
        empty_optimized.left_nearest_multi = PianoMidi.left_nearest_multi.__get__(empty_optimized, PianoMidi)
        empty_optimized.nearest_multi = PianoMidi.nearest_multi.__get__(empty_optimized, PianoMidi)
        empty_optimized.nearest_global = PianoMidi.nearest_global.__get__(empty_optimized, PianoMidi)
        
        # Bind the private methods and initialize using actual class methods
        empty_optimized._build_optimized_indices = PianoMidi._build_optimized_indices.__get__(empty_optimized, PianoMidi)
        empty_optimized._build_lookup_tables = PianoMidi._build_lookup_tables.__get__(empty_optimized, PianoMidi)
        empty_optimized._get_bucket = PianoMidi._get_bucket.__get__(empty_optimized, PianoMidi)
        
        # Build the optimized indices using the actual class methods
        empty_optimized._build_optimized_indices()
        
        # All queries should return None or empty list
        self.assertIsNone(empty_optimized.right_nearest(MelEvent(2.0, 60)))
        self.assertIsNone(empty_optimized.left_nearest(MelEvent(2.0, 60)))
        self.assertIsNone(empty_optimized.nearest(MelEvent(2.0, 60)))
        self.assertEqual(empty_optimized.right_nearest_multi(MelEvent(2.0, 60), 2), [])
        self.assertEqual(empty_optimized.left_nearest_multi(MelEvent(2.0, 60), 2), [])
        self.assertEqual(empty_optimized.nearest_multi(MelEvent(2.0, 60), 2), [])
        self.assertEqual(empty_optimized.nearest_global(MelEvent(2.0, 60), 2), [])

if __name__ == '__main__':
    unittest.main(verbosity=2) 