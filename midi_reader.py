#!/usr/bin/env python3
"""
https://arxiv.org/abs/1808.03715v1
"""
from typing import List, Iterable, Optional

import mido


class MidiProcessor:
    """TODO
    TODO maybe remove <time_shift: 0> and <velocity: 0> events?

    TODO sequence start / end / padding?

    Turn a midi file into tokens:
    -> load midi file from disk
    -> generate events (parse_midi)
    -> get tokens (events_to_indices)

    Turn tokens (from model) into midi file:
    -> get tokens sampled from model
    -> turn into events (indices_to_events)
    -> turn into midi (events_to_midi)
    """
    num_notes = 128
    num_velocities = 32

    sustain_control_channel = 64
    sustain_threshold = 64

    def __init__(self,
                 time_granularity: int = 100,
                 piece_start: bool = False,
                 piece_end: bool = False):
        """Create a MidiProcessor with specific settings.

        :param time_granularity: Size of smallest time unit, per second.
        A time_granularity of 100 corresponds to 1s / 100 = 10ms increments,
        and time_granularity of 125 corresponds to 1s / 125 = 8ms increments.

        :param piece_start: Whether to append a <START> event before the sequence
        :param piece_end: Whether to append an <END> event after the sequence
        """
        self.time_granularity = time_granularity
        self.piece_start = piece_start
        self.piece_end = piece_end
        self._pitch_augmentation = None
        self._time_augmentation = None

        num_extra_tokens = int(piece_start) + int(piece_end)
        self.vocab_size = 2 * self.num_notes + self.time_granularity + self.num_velocities + num_extra_tokens

        self._reset_state()

    def _reset_state(self):
        self._events = []
        self._seconds_elapsed = 0.0
        self._last_velocity = -1
        self._sustain_pressed = False
        self._sustained_notes = set()

    def parse_midi(self,
                   midi: mido.MidiFile,
                   pitch_augmentation: int = 0,
                   time_augmentation: float = 1.0) -> List["Event"]:
        """Parse a midi file to a sequence, optionally augmenting pitch and time values.

        :param midi: The MidiFile to parse
        :param pitch_augmentation: Pitch added to every note in the sequence, in semitones.
        Augmenting pitch by 0 semitones corresponds to no augmenation.
        :param time_augmentation: Multiplier for time values.
        Augmenting time by 1.0x corresponds to no augmentation.

        :return: The events that make up the processed input.
        """
        self._reset_state()
        self._pitch_augmentation = pitch_augmentation
        self._time_augmentation = time_augmentation

        if self.piece_start:
            self._add_event('START')
            raise NotImplementedError

        for msg in midi:
            self._handle_message(msg)

        if self.piece_end:
            self._add_event('END')
            raise NotImplementedError

        return self._events

    def _handle_message(self, msg):
        self._seconds_elapsed += self._augment_time(msg.time)

        if (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off':
            self._handle_note_off(msg)
        elif msg.type == 'note_on':
            self._handle_note_on(msg)
        elif msg.type == 'control_change' and msg.control == self.sustain_control_channel:
            # sustain on -> off
            if msg.value < self.sustain_threshold and self._sustain_pressed:
                self._sustain_pressed = False
                self._handle_sustain_release()
            # sustain off -> on
            elif msg.value >= self.sustain_threshold and not self._sustain_pressed:
                self._sustain_pressed = True

    def _handle_note_on(self, msg):
        pitch = self._augment_pitch(msg.note)
        self._sustained_notes.discard(pitch)
        self._handle_time_shift()
        velo = msg.velocity // 4  # quantize velocity TODO
        if velo != self._last_velocity:
            self._add_event('VELOCITY', velo)
        self._add_event('NOTE_ON', pitch)

    def _handle_note_off(self, msg):
        pitch = self._augment_pitch(msg.note)
        if self._sustain_pressed:
            self._sustained_notes.add(pitch)
        else:
            self._handle_time_shift()
            self._add_event('NOTE_OFF', pitch)

    def _handle_sustain_release(self):
        self._handle_time_shift()
        for pitch in self._sustained_notes:
            self._add_event('NOTE_OFF', pitch)
        self._sustained_notes.clear()

    def _handle_time_shift(self):
        # TODO augment time in here, instead of in _handle_message() ?
        steps = round(self._seconds_elapsed * self.time_granularity)
        self._seconds_elapsed = 0.0  # TODO handle rounding error?
        while steps > 0:
            shift = min(self.time_granularity - 1, steps)
            self._add_event('TIME_SHIFT', shift)
            steps -= shift

    def _add_event(self, event_type: str, event_value: Optional[int] = None):
        index = self._event_index(event_type, event_value)
        event = Event(index, event_type, event_value)
        self._events.append(event)

    def _event_index(self, event_type: str, event_value: Optional[int]) -> int:
        idx = 0
        if event_type == 'NOTE_ON':
            return event_value
        idx += self.num_notes
        if event_type == 'NOTE_OFF':
            return idx + event_value
        idx += self.num_notes
        if event_type == 'TIME_SHIFT':
            return idx + event_value
        idx += self.time_granularity
        if event_type == 'VELOCITY':
            return idx + event_value

        raise NotImplementedError('No support for START / END yet. TODO')
        if event_type == 'START':
            pass
        if event_type == 'END':
            pass
        raise ValueError(f'Could not determine index for event <{event_type}: {event_value}>')

    def _augment_pitch(self, pitch: int):
        pitch += self._pitch_augmentation
        pitch = min(self.num_notes - 1, max(0, pitch))
        return pitch

    def _augment_time(self, time: float):
        time *= self._time_augmentation
        return time

    def events_to_midi(self, events: List["Event"]) -> mido.MidiFile:
        """Turn a list of events into a playable midi file"""
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)

        ticks_elapsed = 0
        current_velocity = 64  # TODO which default velocity?
        for event in events:

            if event.type == 'NOTE_ON':
                track.append(mido.Message('note_on',
                                          note=event.value,
                                          velocity=current_velocity,
                                          time=ticks_elapsed))
                ticks_elapsed = 0
            elif event.type == 'NOTE_OFF':
                track.append(mido.Message('note_off',
                                          note=event.value,
                                          velocity=current_velocity,
                                          time=ticks_elapsed))
                ticks_elapsed = 0
            elif event.type == 'TIME_SHIFT':
                ticks_elapsed += int(event.value / self.time_granularity * 1042)  # TODO which number here?
            elif event.type == 'VELOCITY':
                current_velocity = event.value * 4
            elif event.type == 'START' or event.type == 'END':  # TODO stop sample on generated <END> ?
                pass
            else:
                raise ValueError

        return midi

    def events_to_indices(self, events: List["Event"]) -> Iterable[int]:
        """Turn a list of events into a indices"""
        return map(lambda e: e.index, events)

    def indices_to_events(self, indices: Iterable[int]) -> List["Event"]:
        """Turn indices into a list of events"""
        events = []

        for idx in indices:
            cat = int(idx)  # cast from possible tensor or numpy integer types
            if cat < 0:
                raise ValueError('Event index cannot be negative')
            if cat < self.num_notes:
                events.append(Event(idx, 'NOTE_ON', cat))
                continue
            cat -= self.num_notes
            if cat < self.num_notes:
                events.append(Event(idx, 'NOTE_OFF', cat))
                continue
            cat -= self.num_notes
            if cat < self.time_granularity:
                events.append(Event(idx, 'TIME_SHIFT', cat))
                continue
            cat -= self.time_granularity
            if cat < self.num_velocities:
                events.append(Event(idx, 'VELOCITY', cat))
                continue
            cat -= self.num_velocities
            # TODO handle START / END
            raise NotImplementedError
            raise ValueError(f'Event index ({idx}) cannot be larger than vocab_size ({self.vocab_size})')
        return events


class Event:
    """A single event as created by MidiProcessor.
    An event has a type in {'NOTE_ON', 'NOTE_OFF', 'TIME_SHIFT', 'VELOCITY', 'START', 'END'}
    and a value corresponding to a pitch, time step or velocity.

    This event representation was proposed by Oore et al. (2018)
    and extended to allow for different time granularity as well as optional START and END tokens
    before and after a musical piece. (See :class:`MidiProcessor`)
    """

    def __init__(self, index: int, event_type: str, event_value: Optional[int]):
        self.index = index
        self.type = event_type
        self.value = event_value

    def __repr__(self):
        if self.value is None:
            return f'<{self.type.lower()} [{self.index}]>'
        return f'<{self.type.lower()}: {self.value} [{self.index}]>'

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (other.index == self.index
                    and other.type == self.type
                    and other.value == self.value)
        return False


# class Event:
#     """
#     models the 413 events that make up the performance_rnn vocabulary
#     """
# 
#     vocab_size = 128 + 128 + 100 + 32
# 
#     def __init__(self,
#                  message_type: str,
#                  message_value: int):
#         # sanity check first
#         if message_type == 'NOTE_ON':
#             assert 0 <= message_value < 128
#         elif message_type == 'NOTE_OFF':
#             assert 0 <= message_value < 128
#         elif message_type == 'TIME_SHIFT':
#             assert 0 <= message_value < 100
#         elif message_type == 'VELOCITY':
#             assert 0 <= message_value < 32
#         else:
#             raise ValueError
# 
#         self.type = message_type
#         self.value = message_value
# 
#     def __repr__(self):
#         return f'<{self.type.lower()}: {self.value}>'
# 
#     @property
#     def category(self) -> int:
#         """
#         The category represents an event by a single unique integer.
#         """
#         if self.type == 'NOTE_ON':
#             return self.value
#         elif self.type == 'NOTE_OFF':
#             return 128 + self.value
#         elif self.type == 'TIME_SHIFT':
#             return 128 + 128 + self.value
#         elif self.type == 'VELOCITY':
#             return 128 + 128 + 100 + self.value
#         raise ValueError
# 
#     def __eq__(self, other) -> bool:
#         if isinstance(other, Event):
#             return self.category == other.category
#         return False
# 
#     @staticmethod
#     def from_category(cat) -> 'Event':
#         cat = int(cat)  # cast from possible tensor or numpy int types
#         if cat < 0:
#             raise ValueError('Event category cannot be negative')
#         if cat < 128:
#             return Event('NOTE_ON', cat)
#         cat -= 128
#         if cat < 128:
#             return Event('NOTE_OFF', cat)
#         cat -= 128
#         if cat < 100:
#             return Event('TIME_SHIFT', cat)
#         cat -= 100
#         if cat < 32:
#             return Event('VELOCITY', cat)
#         raise ValueError(f'Event category should be less than {128*2+100+32}')
# 
#     @staticmethod
#     def from_elapsed(seconds: float) -> List['Event']:
#         """
#         Returns a list of time shifts corresponding to waiting {seconds}s.
#         """
#         time_shifts = []
#         # Quantize the elapsed time
#         steps: int = round(seconds * 100)
#         while steps > 0:
#             to_add = min(99, steps)
#             time_shifts.append(Event('TIME_SHIFT', to_add))
#             steps -= to_add
#         return time_shifts


# def midi_to_events(midi: mido.MidiFile, augment_pitch: int = 0, augment_time: float = 1.0) -> List[Event]:
#     """
#     Parse a MidiFile to obtain a list of Events
#     Set augment_pitch to augment by a number of semitones,
#     set augment_time as event time multiplier
#     """
# 
#     events = []
#     seconds_elapsed = 0.0
#     last_velocity = -1
#     sustain_pedal_pressed = False
#     sustained_notes = set()
# 
#     for msg in midi:
# 
#         seconds_elapsed += msg.time * augment_time  # augment time
# 
#         # Note off message (piano key released)
#         if (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off':
#             note_value = msg.note + augment_pitch  # add pitch augmentation
#             note_value = min(127, max(0, note_value))
# 
#             # Note is sustained or stopped
#             if sustain_pedal_pressed:
#                 sustained_notes.add(note_value)
#             else:
#                 events += Event.from_elapsed(seconds_elapsed)
#                 seconds_elapsed = 0.0
#                 events.append(Event('NOTE_OFF', note_value))
# 
#         # Note on message (piano key pressed)
#         elif msg.type == 'note_on':
#             note_value = msg.note + augment_pitch  # add pitch augmentation
#             note_value = min(127, max(0, note_value))
# 
#             sustained_notes.discard(note_value)
#             events += Event.from_elapsed(seconds_elapsed)
#             seconds_elapsed = 0.0
#             velo = msg.velocity // 4  # quantize velocity
#             if velo != last_velocity:
#                 events.append(Event('VELOCITY', velo))
#             events.append(Event('NOTE_ON', note_value))
# 
#         # Sustain pedal
#         elif msg.type == 'control_change' and msg.control == 64:
#             # sustain on -> off
#             if msg.value < 64 and sustain_pedal_pressed:
#                 sustain_pedal_pressed = False
#                 events += Event.from_elapsed(seconds_elapsed)
#                 seconds_elapsed = 0.0
#                 for note in sustained_notes:
#                     events.append(Event('NOTE_OFF', note))
#                 sustained_notes.clear()
# 
#             # sustain pedal off -> on
#             elif msg.value >= 64 and not sustain_pedal_pressed:
#                 sustain_pedal_pressed = True
# 
#         # Possibly to add: Damper = control 67
#         else:
#             pass
# 
#     return events


# def events_to_midi(events: Iterable[Event]) -> mido.MidiFile:
#     """
#     Turn a list of events into a MIDI file
#     For evaluating model performance
#     """
#     result_midi = mido.MidiFile()
#     track = mido.MidiTrack()
#     result_midi.tracks.append(track)
# 
#     ticks_elapsed = 0
#     current_velocity = 64
#     for event in events:
# 
#         if event.type == 'NOTE_ON':
#             track.append(mido.Message('note_on',
#                                       note=event.value,
#                                       velocity=current_velocity,
#                                       time=ticks_elapsed))
#             ticks_elapsed = 0
#         elif event.type == 'NOTE_OFF':
#             track.append(mido.Message('note_off',
#                                       note=event.value,
#                                       velocity=current_velocity,
#                                       time=ticks_elapsed))
#             ticks_elapsed = 0
#         elif event.type == 'TIME_SHIFT':
#             ticks_elapsed += int(event.value / 100 * 1042)  # TODO which number here?
#         elif event.type == 'VELOCITY':
#             current_velocity = event.value * 4
#         else:
#             raise ValueError
# 
#     return result_midi


def test_conversions(path):
    """
    Test to check if converting an event list to midi and back to event list
    changes the event list. It shouldn't.
    """
    midi_before = mido.MidiFile(path)
    events_before = midi_to_events(midi_before)
    midi_after = events_to_midi(events_before)
    events_after = midi_to_events(midi_after)

    # TODO test fails, probably because the TIME_SHIFTs are not translated well to midi
    # The values differ slightly
    assert events_before == events_after


def main():
    path = '../datasets/maestro/maestro-v3.0.0/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi'
    midi = mido.MidiFile(path)
    # test_conversions(path)
    events = midi_to_events(midi)

    [print(e) for e in events]
    print(len(events))

    return
    midi = events_to_midi(events)

    with mido.open_output('FLUID') as port:
        i = 0
        for msg in midi.play():
            print(i, ':', msg)
            i += 1
            port.send(msg)


if __name__ == '__main__':
    main()


