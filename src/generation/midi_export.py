"""Convert piano-roll arrays to MIDI files."""
import numpy as np
import pretty_midi


def piano_roll_to_midi(piano_roll, output_path, fs=8, threshold=0.5):
    """Convert a 128 x T piano-roll numpy array into a MIDI file.

    Args:
        piano_roll: Binary array (pitches x time) or (time x pitches)
        output_path: Path to save the .mid file
        fs: Sampling rate (frames per second)
        threshold: Activation threshold for note detection
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    # Ensure shape is (pitches, time_steps)
    if piano_roll.shape[0] != 128 and piano_roll.shape[1] == 128:
        piano_roll = piano_roll.T

    pitches, frames = piano_roll.shape
    for pitch in range(pitches):
        i = 0
        while i < frames:
            if piano_roll[pitch, i] > threshold:
                start = i / fs
                j = i
                while j < frames and piano_roll[pitch, j] > threshold:
                    j += 1
                end = j / fs
                inst.notes.append(pretty_midi.Note(
                    velocity=90, pitch=pitch, start=start, end=end))
                i = j
            else:
                i += 1

    pm.instruments.append(inst)
    pm.write(output_path)
