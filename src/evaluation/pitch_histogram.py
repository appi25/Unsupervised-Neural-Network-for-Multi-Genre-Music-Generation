"""Pitch Histogram Similarity metric (Section 5).

H(p, q) = sum_i=1..12 |p_i - q_i|
"""
import numpy as np


def pitch_histogram_similarity(p, q=None):
    """L1 distance between two normalized pitch-class distributions."""
    if q is None:
        q = np.ones(12) / 12
    return np.sum(np.abs(p - q))
