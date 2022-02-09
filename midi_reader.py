#!/usr/bin/env python3
"""
https://arxiv.org/abs/1808.03715v1
"""
import os
from typing import List, Iterable

import pdb

import mido


class Event:
    """
    models the 413 events that make up the performance_rnn vocabulary
    """

    vocab_size = 128 + 128 + 125 + 32

    def __init__(self,
                 message_type: str,
                 message_value: int):
        # sanity check first
        if message_type == 'NOTE_ON':
            assert 0 <= message_value < 128
        elif message_type == 'NOTE_OFF':
            assert 0 <= message_value < 128
        elif message_type == 'TIME_SHIFT':
            assert 0 <= message_value < 125
        elif message_type == 'VELOCITY':
            assert 0 <= message_value < 32
        else:
            raise ValueError

        self.type = message_type
        self.value = message_value

    def __repr__(self):
        return f'<{self.type.lower()}: {self.value}>'

    @property
    def category(self) -> int:
        """
        The category represents an event by a single unique integer.
        """
        if self.type == 'NOTE_ON':
            return self.value
        elif self.type == 'NOTE_OFF':
            return 128 + self.value
        elif self.type == 'TIME_SHIFT':
            return 128 + 128 + self.value
        elif self.type == 'VELOCITY':
            return 128 + 128 + 125 + self.value
        raise ValueError

    def __eq__(self, other) -> bool:
        if isinstance(other, Event):
            return self.category == other.category
        return False

    @staticmethod
    def from_category(cat) -> 'Event':
        cat = int(cat)  # cast from possible tensor or numpy int types
        if cat < 0:
            raise ValueError('Event category cannot be negative')
        if cat < 128:
            return Event('NOTE_ON', cat)
        cat -= 128
        if cat < 128:
            return Event('NOTE_OFF', cat)
        cat -= 128
        if cat < 125:
            return Event('TIME_SHIFT', cat)
        cat -= 125
        if cat < 32:
            return Event('VELOCITY', cat)
        raise ValueError(f'Event category should be less than {128*2+125+32}')

    @staticmethod
    def from_elapsed(seconds: float) -> List['Event']:
        """
        Returns a list of time shifts corresponding to waiting {seconds}s.
        """
        time_shifts = []
        # Quantize the elapsed time
        # 0.000 -> 0
        # 0.008 -> 1
        # 0.016 -> 2
        # 1.000 -> 125
        steps: int = round(seconds * 125)
        while steps > 0:
            to_add = min(124, steps)
            time_shifts.append(Event('TIME_SHIFT', to_add))
            steps -= to_add
        return time_shifts


def midi_to_events(midi: mido.MidiFile, augment_pitch: int = 0, augment_time: float = 1.0) -> List[Event]:
    """
    Parse a MidiFile to obtain a list of Events
    Set augment_pitch to augment by a number of semitones,
    set augment_time as event time multiplier
    """

    events = []
    seconds_elapsed = 0.0
    last_velocity = -1
    sustain_pedal_pressed = False
    sustained_notes = set()

    for msg in midi:

        seconds_elapsed += msg.time * augment_time  # augment time

        # Note off message (piano key released)
        if (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off':
            note_value = msg.note + augment_pitch  # add pitch augmentation
            note_value = min(127, max(0, note_value))

            # Note is sustained or stopped
            if sustain_pedal_pressed:
                sustained_notes.add(note_value)
            else:
                events += Event.from_elapsed(seconds_elapsed)
                seconds_elapsed = 0.0
                events.append(Event('NOTE_OFF', note_value))

        # Note on message (piano key pressed)
        elif msg.type == 'note_on':
            note_value = msg.note + augment_pitch  # add pitch augmentation
            note_value = min(127, max(0, note_value))

            sustained_notes.discard(note_value)
            events += Event.from_elapsed(seconds_elapsed)
            seconds_elapsed = 0.0
            velo = msg.velocity // 4  # quantize velocity
            if velo != last_velocity:
                events.append(Event('VELOCITY', velo))
            events.append(Event('NOTE_ON', note_value))

        # Sustain pedal
        elif msg.type == 'control_change' and msg.control == 64:
            # sustain on -> off
            if msg.value < 64 and sustain_pedal_pressed:
                sustain_pedal_pressed = False
                events += Event.from_elapsed(seconds_elapsed)
                seconds_elapsed = 0.0
                for note in sustained_notes:
                    events.append(Event('NOTE_OFF', note))
                sustained_notes.clear()

            # sustain pedal off -> on
            elif msg.value >= 64 and not sustain_pedal_pressed:
                sustain_pedal_pressed = True

        # Possibly to add: Damper = control 67
        else:
            pass

    return events


def events_to_midi(events: Iterable[Event]) -> mido.MidiFile:
    """
    Turn a list of events into a MIDI file
    For evaluating model performance
    """
    result_midi = mido.MidiFile()
    track = mido.MidiTrack()
    result_midi.tracks.append(track)

    ticks_elapsed = 0
    current_velocity = 64
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
            ticks_elapsed += int(event.value / 125 * 1042)  # TODO which number here?
        elif event.type == 'VELOCITY':
            current_velocity = event.value * 4
        else:
            raise ValueError

    return result_midi


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


def get_primer(filename, length):
    filename = "2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_08_R1_2013_wav--1.midi"
    length = 128

    path = os.path.join('data/maestro-v3.0.0', filename)
    midi = mido.MidiFile(path)

    import numpy as np
    return np.array(midi_to_events(midi))[:length]


if __name__ == '__main__':
    print([x for x in map(lambda x: x.category, get_primer(None, None))])
    # main()


