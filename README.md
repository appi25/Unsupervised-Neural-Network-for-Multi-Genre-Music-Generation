# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 — Neural Networks (Spring 2026)  
**Authors:** [Your Name(s)]

## Overview
Four progressive unsupervised neural architectures for symbolic music generation:
1. **Task 1 (Easy):** LSTM Autoencoder — piano-roll reconstruction
2. **Task 2 (Medium):** Variational Autoencoder — KL regularization + latent interpolation
3. **Task 3 (Hard):** Music Transformer — REMI tokens, Perplexity 1.4093
4. **Task 4 (Advanced):** RLHF fine-tuning — 10-participant preference survey

## Dataset
Lakh MIDI Dataset — `clean_midi` subset (17,162 MIDI files)
No genre labels used (unsupervised).

## Repository Structure
```
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/                       # MIDI data (not included due to size)
│   ├── raw_midi/
│   ├── processed/
│   └── train_test_split/
├── notebooks/                  # Project notebooks
│   ├── NN_Project_clean.ipynb  # Main project notebook (all 4 tasks)
│   ├── preprocessing.ipynb
│   └── baseline_markov.ipynb
├── src/                        # Source modules
│   ├── config.py
│   ├── preprocessing/          # MIDI parsing + tokenization
│   ├── models/                 # AE, VAE, Transformer, Diffusion
│   ├── training/               # Training scripts per task
│   ├── evaluation/             # Metrics (H, D, R, Perplexity)
│   └── generation/             # Music generation + MIDI export
├── outputs/
│   ├── generated_midis/        # 56+ generated MIDI files
│   ├── plots/                  # Loss curves + comparison charts
│   └── survey_results/         # 4 survey CSVs (1,750 total ratings)
└── report/
    ├── final_report.tex        # LaTeX source (IEEE format)
    ├── architecture_diagrams/
    └── references.bib
```

## Key Results

| Model               | Loss | Perplexity | Rhythm D ↑ | Repetition R ↓ | Human (1-5) ↑ |
|---------------------|------|------------|------------|----------------|---------------|
| Random Generator    | –    | –          | 0.007      | 0.000          | 1.24          |
| Markov Chain        | –    | –          | 0.124      | 0.149          | 2.36          |
| Task 1: LSTM AE     | 0.11 | –          | 0.293      | 0.881          | 3.20          |
| Task 2: VAE         | 0.31 | –          | 0.003      | 0.890          | 3.80          |
| Task 3: Transformer | –    | 1.41       | 0.034      | 0.427          | 4.30          |
| Task 4: RLHF-Tuned  | –    | 1.29       | 0.035      | 0.119          | **4.58**      |

All human scores measured from 4 independent surveys (10 participants each).
Monotonic progression: 1.24 → 2.36 → 3.20 → 3.80 → 4.30 → 4.58

## Human Evaluation
- Survey 1 (Tasks 1-3): 10 participants × 10 tracks × 7 dims = 700 ratings
- Survey 2 (Task 4 RLHF): 10 participants × 5 tracks × 7 dims = 350 ratings
- Survey 3 (Random baseline): 10 participants × 5 tracks × 7 dims = 350 ratings
- Survey 4 (Markov baseline): 10 participants × 5 tracks × 7 dims = 350 ratings
- **Total: 1,750 human ratings**

## Running
1. Open `notebooks/NN_Project_clean.ipynb` in Google Colab
2. Mount Google Drive and update paths if needed
3. Run all cells top to bottom (~10 min)

## References
[1] Huang et al., Music Transformer, ICLR 2019 — https://arxiv.org/abs/1809.04281
[2] Roberts et al., MusicVAE, ICML 2018 — https://arxiv.org/abs/1803.05428
[3] Ouyang et al., RLHF, NeurIPS 2022 — https://arxiv.org/abs/2203.02155
[4] Raffel, Lakh MIDI Dataset, 2016 — https://colinraffel.com/projects/lmd/
[5] Huang & Yang, Pop Music Transformer, 2020 — https://arxiv.org/abs/2002.00212
[6] van den Oord et al., WaveNet, 2016 — https://arxiv.org/abs/1609.03499
