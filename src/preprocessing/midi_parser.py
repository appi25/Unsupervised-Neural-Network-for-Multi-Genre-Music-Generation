"""MIDI preprocessing pipeline for Tasks 1 and 2.

Converts raw MIDI files into binarized piano-roll representations
at 8 Hz sampling rate.
"""
import os
import numpy as np
import pretty_midi


def process_midi(file_path, fs=8):
    """Convert a single MIDI file to binarized piano roll."""
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        piano_roll = midi_data.get_piano_roll(fs=fs)
        return (piano_roll > 0).astype(np.int8)
    except Exception:
        return None


def process_dataset(data_dir, output_dir, limit=7000):
    """Process all MIDI files in a directory to .npy piano rolls."""
    os.makedirs(output_dir, exist_ok=True)
    midi_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, f))

    count = 0
    for path in midi_files[:limit]:
        roll = process_midi(path)
        if roll is not None and roll.shape[1] >= 64:
            np.save(os.path.join(output_dir, f'roll_{count}.npy'), roll.T)
            count += 1

    print(f"Processed {count} files to {output_dir}")
    return count
