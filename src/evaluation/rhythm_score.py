"""Rhythm Diversity Score metric (Section 5).

D_rhythm = #unique_durations / #total_notes
"""


def rhythm_diversity(durations):
    """Ratio of unique note durations to total note count."""
    if not durations:
        return 0.0
    return len(set(durations)) / len(durations)
