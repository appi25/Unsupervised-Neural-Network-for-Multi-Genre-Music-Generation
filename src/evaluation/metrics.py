"""Evaluation metrics from project specification Section 5.

Pitch Histogram Similarity: H(p,q) = sum_i |p_i - q_i|
Rhythm Diversity Score: D = #unique_durations / #total_notes
Repetition Ratio: R = #repeated_patterns / #total_patterns
Human Listening Score: Score_human in [1, 5]
"""
import numpy as np
import pretty_midi
import os
import glob


def pitch_histogram_similarity(pm, baseline=None):
    """H(p,q) = sum_i=1..12 |p_i - q_i| against uniform baseline."""
    if baseline is None:
        baseline = np.ones(12) / 12
    hist = pm.get_pitch_class_histogram()
    p = hist / (np.sum(hist) + 1e-8)
    return np.sum(np.abs(p - baseline))


def rhythm_diversity(pm):
    """D = #unique durations / #total notes."""
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return 0.0
    durations = [round(n.end - n.start, 3) for n in notes]
    return len(set(durations)) / len(notes)


def repetition_ratio(pm, k=4):
    """R = #repeated k-grams / #total k-grams."""
    notes = [n for inst in pm.instruments for n in inst.notes]
    pitches = [n.pitch for n in notes]
    if len(pitches) < k:
        return 0.0
    ngrams = [tuple(pitches[i:i+k]) for i in range(len(pitches) - k + 1)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def calculate_note_density(piano_roll_binary):
    """Fraction of active cells in a binary piano roll."""
    return np.mean(piano_roll_binary)


def evaluate_folder(folder_path):
    """Run all three metrics on every MIDI file in a folder."""
    if not os.path.exists(folder_path):
        return None
    files = glob.glob(f'{folder_path}/**/*.mid', recursive=True)
    if not files:
        return None

    H, D, R = [], [], []
    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(f)
            H.append(pitch_histogram_similarity(pm))
            D.append(rhythm_diversity(pm))
            R.append(repetition_ratio(pm))
        except Exception:
            continue

    return {
        'n_files': len(H),
        'H': float(np.mean(H)) if H else 0.0,
        'D': float(np.mean(D)) if D else 0.0,
        'R': float(np.mean(R)) if R else 0.0,
    }
