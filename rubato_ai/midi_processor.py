"""
https://arxiv.org/abs/1808.03715v1
"""
from typing import List, Iterable, Optional

from .registry import register_param

import mido


@register_param('time_granularity', int,
                'Number of midi processor <TIME_SHIFT> events per second')
@register_param('piece_start', bool,
                'Whether to prepend <START> events to sequences')
@register_param('piece_end', bool,
                'Whether to append <END> events to sequences')
class MidiProcessor:
    """Class for generating tokens from a midi file and turning tokens back into a midi file.

    A MidiProcessor can parse midi files into an event-based representation.
    This representation is based on the suggested midi representation by Oore et al. (2018),
    but allows control over the time granularity.

    Additionally, optional <START> and <END> tokens are pre- and appended to processed midi sequences
    to supply the model with information on when a piece starts; these tokens are added
    only at the beginning and end of a piece, not at sliding windows in the middle of the piece.

    TODO maybe remove <time_shift: 0> and <velocity: 0> events?

    FIXME bug in _handle_time_shift (see :func:`test_conversions` below)

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

    def __init__(self, **config):
        """Create a MidiProcessor with specific settings.

        :param time_granularity: Size of smallest time unit, per second.
        A time_granularity of 100 corresponds to 1s / 100 = 10ms increments, as used by Huang et al. (2018)
        and time_granularity of 125 corresponds to 1s / 125 = 8ms increments, as used by Oore et al. (2018).

        :param piece_start: Whether to append a <START> event before the sequence
        :param piece_end: Whether to append an <END> event after the sequence
        """
        self.time_granularity = config['time_granularity']
        self.piece_start = config['piece_start']
        self.piece_end = config['piece_end']
        self._pitch_augmentation = None
        self._time_augmentation = None

        num_extra_tokens = int(self.piece_start) + int(self.piece_end)
        self.vocab_size = 2 * self.num_notes + self.time_granularity + self.num_velocities + num_extra_tokens

        self._reset_state()

    @property
    def start_token(self) -> int:
        event_type = 'START'
        return self._event_index(event_type)

    @property
    def end_token(self) -> int:
        event_type = 'END'
        return self._event_index(event_type)

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
        """
        Parse a midi file to a sequence, optionally augmenting pitch and time values.

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

        for msg in midi:
            self._handle_message(msg)

        if self.piece_end:
            self._add_event('END')

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

    def _add_event(self, event_type: str, event_value: int = -1):
        index = self._event_index(event_type, event_value)
        event = Event(index, event_type, event_value)
        self._events.append(event)

    def _event_index(self, event_type: str, event_value: int) -> int:
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
        idx += self.num_velocities

        if self.piece_start:
            if event_type == 'START':
                return idx
            idx += 1
        if self.piece_end:
            if event_type == 'END':
                return idx
        raise ValueError(f'Could not determine index for event <{event_type}: {event_value}>')

    def _augment_pitch(self, pitch: int):
        assert self._pitch_augmentation is not None
        pitch += self._pitch_augmentation
        pitch = min(self.num_notes - 1, max(0, pitch))
        return pitch

    def _augment_time(self, time: float):
        assert self._time_augmentation is not None
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
                ticks_elapsed += int(event.value / self.time_granularity * 956)  # TODO which number here?
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
            if self.piece_start:
                if cat == 0:
                    events.append(Event(idx, 'START'))
                    continue
                cat -= 1
            if self.piece_end:
                if cat == 0:
                    events.append(Event(idx, 'END'))
                    continue
            raise ValueError(f'Event index ({idx}) cannot be larger than vocab_size ({self.vocab_size})')
        return events


class Event:
    """A single event as created by MidiProcessor.
    An event has a type in {'NOTE_ON', 'NOTE_OFF', 'TIME_SHIFT', 'VELOCITY', 'START', 'END'}
    and a value corresponding to a pitch, time step or velocity.

    A value of -1 is used for <START> or <END> tokens.

    This event representation was proposed by Oore et al. (2018)
    and extended to allow for different time granularity as well as optional START and END tokens
    before and after a musical piece. (See :class:`MidiProcessor`)
    """

    def __init__(self, index: int, event_type: str, event_value: int = -1):
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


def test_conversions(path: str, midi_processor: MidiProcessor):
    """
    Test to check if converting an event list to midi and back to event list
    changes the event list.

    FIXME: time_shift events are not translated well, especially where there
        are many consecutive control_change messages in the input.
        This most likely comes from the rounding of time values
        in MidiProcessor._handle_time_shift().
    """
    midi_before = mido.MidiFile(path)
    events_before = midi_processor.parse_midi(midi_before)
    midi_after = midi_processor.events_to_midi(events_before)
    events_after = midi_processor.parse_midi(midi_after)

    assert len(events_before) == len(events_after)

    for i in range(len(events_before)):
        print(f'i: {i}\n\tbefore: {events_before[i]}\n\tafter: {events_after[i]}')
        assert events_before[i] == events_after[i]


def main():
    path = '../datasets/maestro/maestro-v3.0.0/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi'
    midi_processor = MidiProcessor(time_granularity=100, piece_start=True, piece_end=True)
    test_conversions(path, midi_processor)


if __name__ == '__main__':
    main()


