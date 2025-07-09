from functools import cache
from dataclasses import dataclass
from typing import Literal
import warnings

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
        if isinstance(other, (float, int)):
            return MelEvent(self.time + other, self.note)
        return NotImplemented

    def __lshift__(self, other: float) -> 'MelEvent':
        # "a << float": time shift
        if isinstance(other, (float, int)):
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
        if isinstance(other, (float, int)):
            return MelEvent(self.time * other, self.note)
        return NotImplemented

    def __repr__(self) -> str:
        # Convert note to note name
        return f"MelEvent_(time={self.time:.4f}, note='{self.note_name}')"

    def __floordiv__(self, other: float) -> 'MelEvent':
        # Quantize time to the nearest multiple of other
        if isinstance(other, (float, int)):
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

type EventLike = MelEvent | MidiEvent