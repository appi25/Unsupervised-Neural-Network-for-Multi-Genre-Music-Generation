"""Project configuration and hyperparameters."""

# Dataset
DATASET_PATH = 'data/raw_midi/clean_midi'
PROCESSED_PATH = 'data/processed'
OUTPUT_PATH = 'outputs'

# Piano-roll settings (Tasks 1, 2)
FS = 8                # frames per second
SEQ_LEN = 64          # time steps per window
INPUT_DIM = 128       # MIDI pitch range

# LSTM Autoencoder (Task 1)
HIDDEN_DIM_AE = 256
LATENT_DIM = 128

# VAE (Task 2)
HIDDEN_DIM_VAE = 512
BETA = 0.1            # KL divergence weight

# Transformer (Tasks 3, 4)
VOCAB_SIZE = 397      # REMI token vocabulary
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FF = 1024
MAX_SEQ_LEN = 128

# Training
BATCH_SIZE_AE = 64
BATCH_SIZE_TR = 16
LR_AE = 1e-3
LR_VAE = 1e-3
LR_TR = 1e-4
LR_RLHF = 1e-5
