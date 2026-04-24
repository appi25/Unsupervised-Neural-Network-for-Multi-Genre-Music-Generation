# 🎵 Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 — Neural Networks (Spring 2026)  
**Authors:** Mohammed Asifur Rahman

---

## 📌 Overview
Four progressive unsupervised neural architectures for symbolic music generation:

1. **Task 1 (Easy):** LSTM Autoencoder — piano-roll reconstruction  
2. **Task 2 (Medium):** Variational Autoencoder — KL regularization + latent interpolation  
3. **Task 3 (Hard):** Music Transformer — REMI tokens, Perplexity 1.4093  
4. **Task 4 (Advanced):** RLHF fine-tuning — human preference optimization  

---

## 🎼 Dataset
- **Lakh MIDI Dataset** (`clean_midi` subset)  
- **17,162 MIDI files**  
- Fully unsupervised (no genre labels)

---

## 🗂️ Repository Structure
```bash
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_midi/
│   ├── processed/
│   └── train_test_split/
├── notebooks/
│   ├── NN_Project_clean.ipynb
│   ├── preprocessing.ipynb
│   └── baseline_markov.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── generation/
├── outputs/
│   ├── generated_midis/
│   ├── plots/
│   └── survey_results/
└── report/
    ├── final_report.tex
    ├── architecture_diagrams/
    └── references.bib
| Model            | Loss | Perplexity | Rhythm D ↑ | Repetition R ↓ | Human ↑  |
| ---------------- | ---- | ---------- | ---------- | -------------- | -------- |
| Random Generator | –    | –          | 0.007      | 0.000          | 1.24     |
| Markov Chain     | –    | –          | 0.124      | 0.149          | 2.36     |
| LSTM AE          | 0.11 | –          | 0.293      | 0.881          | 3.20     |
| VAE              | 0.31 | –          | 0.003      | 0.890          | 3.80     |
| Transformer      | –    | 1.41       | 0.034      | 0.427          | 4.30     |
| RLHF-Tuned       | –    | **1.29**   | 0.035      | 0.119          | **4.58** |
Human Evaluation
Total: 1,750 ratings
4 surveys (10 participants each)
Survey 1 (Tasks 1–3): 700 ratings
Survey 2 (RLHF): 350 ratings
Survey 3 (Random): 350 ratings
Survey 4 (Markov): 350 ratings

Running
Open notebooks/NN_Project_clean.ipynb in Google Colab
Mount Google Drive
Update dataset paths
Run all cells (~10 min)

References
Huang et al., Music Transformer (ICLR 2019)
Roberts et al., MusicVAE (ICML 2018)
Ouyang et al., RLHF (NeurIPS 2022)
Raffel, Lakh MIDI Dataset (2016)
Huang & Yang, Pop Music Transformer (2020)
van den Oord et al., WaveNet (2016)

---

---
 
