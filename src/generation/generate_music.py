"""Autoregressive generation for the Music Transformer (Tasks 3, 4)."""
import torch
import torch.nn.functional as F


def generate_music(model, tokenizer, seed_seq=None, max_len=1000,
                   temperature=0.9, device='cpu'):
    """Sample autoregressively from a trained Music Transformer.

    Args:
        model: Trained MusicTransformer
        tokenizer: REMI tokenizer (for token-to-MIDI conversion)
        seed_seq: Starting token IDs (default [0])
        max_len: Maximum tokens to generate
        temperature: Sampling temperature (0.9 recommended)
        device: cuda or cpu

    Returns:
        Generated token sequence as list of ints
    """
    if seed_seq is None:
        seed_seq = [0]

    model.eval()
    generated = torch.tensor([seed_seq]).to(device)
    with torch.no_grad():
        for _ in range(max_len):
            input_idx = generated[:, -128:]
            logits = model(input_idx)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

    return generated.squeeze().tolist()
