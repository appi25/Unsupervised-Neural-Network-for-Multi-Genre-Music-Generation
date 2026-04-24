"""REMI tokenizer setup for Tasks 3 and 4.

Uses miditok library to convert MIDI files into REMI event tokens.
Vocabulary size: 397 tokens (note-on, velocity, time-shift, bar markers).
"""
from miditok import REMI, TokenizerConfig
from pathlib import Path


def get_tokenizer():
    """Return a configured REMI tokenizer."""
    config = TokenizerConfig(num_velocities=16, use_programs=True)
    return REMI(config)


def tokenize_midi_files(midi_dir, tokenizer, max_files=50):
    """Tokenize a collection of MIDI files into REMI sequences."""
    midi_root = Path(midi_dir)
    midi_paths = list(midi_root.rglob('*.mid'))[:max_files]

    tokenized_data = []
    for path in midi_paths:
        try:
            tokens = tokenizer(path)
            if isinstance(tokens, list):
                tokenized_data.append(tokens[0].ids)
            else:
                tokenized_data.append(tokens.ids)
        except Exception:
            continue

    print(f"Tokenized {len(tokenized_data)} files, vocab size: {len(tokenizer)}")
    return tokenized_data
