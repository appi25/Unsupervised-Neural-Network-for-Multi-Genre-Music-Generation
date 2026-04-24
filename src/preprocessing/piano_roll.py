"""Piano-roll segmentation and normalization for Tasks 1 and 2.

Segments piano-roll matrices into fixed-length windows of 64 time steps
(8 seconds at 8 Hz) for batch training.
"""
import numpy as np
import os


def segment_piano_roll(piano_roll, seq_len=64):
    """Split a piano roll into fixed-length windows."""
    n_frames = piano_roll.shape[0] if piano_roll.ndim == 2 else piano_roll.shape[1]
    segments = []
    for start in range(0, n_frames - seq_len + 1, seq_len):
        if piano_roll.ndim == 2 and piano_roll.shape[0] == 128:
            segments.append(piano_roll[:, start:start + seq_len].T)
        else:
            segments.append(piano_roll[start:start + seq_len])
    return np.array(segments) if segments else np.array([])


def normalize_and_window(input_dir, output_dir, window_size=64, batch_size=5000):
    """Process all .npy files: segment into windows and save as batches."""
    os.makedirs(output_dir, exist_ok=True)
    all_segments = []

    for f in sorted(os.listdir(input_dir)):
        if not f.endswith('.npy'):
            continue
        roll = np.load(os.path.join(input_dir, f))
        segs = segment_piano_roll(roll, seq_len=window_size)
        if len(segs) > 0:
            all_segments.append(segs)

    if all_segments:
        all_data = np.concatenate(all_segments, axis=0)
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i + batch_size]
            np.save(os.path.join(output_dir, f'batch_{i // batch_size}.npy'), batch)
        print(f"Saved {len(all_data)} segments in {len(all_data) // batch_size + 1} batches")
    return len(all_segments)
