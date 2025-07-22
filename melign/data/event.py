from functools import cache
from dataclasses import dataclass
from typing import Literal, overload
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

    @overload
    def __truediv__(self, other: float) -> 'MelEvent': ...
    @overload
    def __truediv__(self, other: 'MelEvent') -> float: ...
    def __truediv__(self, other: 'float | MelEvent') -> 'MelEvent | float':
        # "a / b": time ratio
        if isinstance(other, MelEvent):
            return self.time / other.time
        elif isinstance(other, (float, int)):
            return MelEvent(self.time / other, self.note)
        return NotImplemented
    
    def __mul__(self, other: float) -> 'MelEvent':
        # "a * float": time scale
        if isinstance(other, (float, int)):
            return MelEvent(self.time * other, self.note)
        return NotImplemented

    def __repr__(self) -> str:
        # Convert note to note name
        if self.note > 12:
            return f"MelEvent_(time={self.time:.2f}, note='{self.note_name}')"
        else:
            return f"MelEvent_(time={self.time:.2f}, note={self.note})"

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
    velocity: int  # Velocity (0-127)

    @overload
    def __sub__(self, other: 'MidiEvent') -> 'MidiEvent': ...
    @overload
    def __sub__(self, other: 'MelEvent') -> 'MelEvent': ...
    def __sub__(self, other: 'EventLike') -> 'EventLike':
        # "a - b": time, note, and velocity shift
        if isinstance(other, MidiEvent):
            return MidiEvent(self.time - other.time, self.note - other.note, self.velocity - other.velocity)
        elif isinstance(other, MelEvent):
            return super(MidiEvent, self).__sub__(other)
        return NotImplemented
    
    @overload
    def __add__(self, other: 'MidiEvent') -> 'MidiEvent': ...
    @overload
    def __add__(self, other: 'MelEvent') -> 'MelEvent': ...
    def __add__(self, other: 'EventLike') -> 'EventLike':
        # "a + b": time, note, and velocity shift
        if isinstance(other, MidiEvent):
            return MidiEvent(self.time + other.time, self.note + other.note, self.velocity + other.velocity)
        elif isinstance(other, MelEvent):
            return super(MidiEvent, self).__add__(other)
        return NotImplemented

    def __rshift__(self, other: float) -> 'MidiEvent':
        # "a >> float": time shift
        if isinstance(other, (float, int)):
            return MidiEvent(self.time + other, self.note, self.velocity)
        return NotImplemented

    def __lshift__(self, other: float) -> 'MidiEvent':
        # "a << float": time shift
        if isinstance(other, (float, int)):
            return MidiEvent(self.time - other, self.note, self.velocity)
        return NotImplemented

    def __xor__(self, other: int) -> 'MidiEvent':
        # "a ^ int": note shift
        if isinstance(other, int):
            return MidiEvent(self.time, self.note + other, self.velocity)
        return NotImplemented

    @overload
    def __truediv__(self, other: float) -> 'MidiEvent': ...
    @overload
    def __truediv__(self, other: 'EventLike') -> float: ...
    def __truediv__(self, other: 'float | EventLike') -> 'MidiEvent | float':
        # "a / b": time ratio
        if isinstance(other, (MidiEvent, MelEvent)):
            return self.time / other.time
        elif isinstance(other, (float, int)):
            return MidiEvent(self.time / other, self.note, self.velocity)
        return NotImplemented
    
    def __mul__(self, other: float) -> 'MidiEvent':
        # "a * float": time scale
        if isinstance(other, (float, int)):
            return MidiEvent(self.time * other, self.note, self.velocity)
        return NotImplemented

    def __floordiv__(self, other: float) -> 'MidiEvent':
        # Quantize time to the nearest multiple of other
        if isinstance(other, (float, int)):
            return MidiEvent(round(self.time / other) * other, self.note, self.velocity)
        return NotImplemented
    
    def __mod__(self, other: Literal[12]) -> 'MidiEvent':
        # "a % b": note modulo
        if other == 12:
            return MidiEvent(self.time, self.note % 12, self.velocity)
        elif isinstance(other, int):
            warnings.warn(f"Modulo by {other} does not make sense. You may want to use 12 instead.")
            return MidiEvent(self.time, self.note % other, self.velocity)
        else:
            return NotImplemented

    @property
    def note_name(self) -> str:
        return midi_to_note_name(self.note)

    def __repr__(self) -> str:
        if self.note > 12:
            return f"MidiEvent_(time={self.time:.2f}, note='{self.note_name}', velocity={self.velocity})"
        else:
            return f"MidiEvent_(time={self.time:.2f}, note={self.note}, velocity={self.velocity})"

def MidiEvent_(time: float, note: int | str, velocity: int) -> MidiEvent:
    if isinstance(note, str):
        note = note_name_to_midi(note)
    return MidiEvent(time, note, velocity)

type EventLike = MelEvent | MidiEvent