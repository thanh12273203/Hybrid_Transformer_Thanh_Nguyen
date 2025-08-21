# Event Classification with Masked Transformer Autoencoders <img src='assets/pics/gsoc_icon.png' alt="GSoC" width='30'/>

A Particle Transformer model with EquiLinear layers for jet physics tasks on JetClass dataset, including jet classification and masked particle reconstruction with momentum-conservation losses.

## Overview

High-energy physics jets are complex objects composed of many particles. This repo provides implementations of Particle Transformer-based models and training utilities to tackle:
- Jet classification (multi-class) using attention over per-particle inputs
- Masked particle reconstruction (predicting a masked particle’s kinematics) with conservation-aware losses

![Cutaway diagram of CMS detector (retrieved from https://cds.cern.ch/record/2665537/files/)](assets/pics/cms_160312_02.png)
*Cutaway diagram of CMS detector (retrieved from https://cds.cern.ch/record/2665537/files/)*

Core components:
- Models: `ParticleTransformer` with `EquiLinear` layers.
- Engine: Training/eval loops, logging, checkpointing (see `src/engine/trainer.py`)
- Configs: YAML-driven experiment config (see `configs/`)
- Datasets: Utilities for loading the ROOT files from JetClass dataset (see `src/utils/data`)

## Get Started

### Prerequisites

- Python 3.13
- Git

### Installation

1. **Clone the repository**

    ```bash
    git clone <repository-url>
    cd Hybrid_Transformer_Thanh_Nguyen
    ```

2. **Create and activate a virtual environment**

    ```bash
    python -m venv venv

    # On Windows
    venv\Scripts\activate

    # On macOS/Linux
    source venv/bin/activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install PyTorch (GPU or CPU):**

    See: https://pytorch.org/get-started/locally/

## Data

The JetClass dataset is publicly available: https://zenodo.org/records/6619768

Install and place the ROOT files under `./data/` in split folders. Example:

```
data/
├── train_100M/
├── val_5M/
│   ├── HToBB_120.root
│   ├── ...
│   └── ZToQQ_124.root
└── test_20M/
```

Utilities to read ROOT and build numpy arrays/live datasets are under `src/utils/data`. Per-particle features are typically `[pT, eta, phi, energy]`. Some configs use a mask to hide one particle during training and reconstruct it.

## Configuration

Experiments are defined in YAML. See `configs/train_ParT.yaml` (excerpt):

```yaml
model:
    num_classes: 10
    embed_dim: 128
    num_layers: 8
    max_num_particles: 128
    num_particle_features: 4
    mask: True  # enable masked-particle reconstruction mode
    inference: False

train:
    batch_size: 128
    criterion:
        name: 'conservation_loss'
        kwargs:
            loss_coef: [0.25, 0.25, 0.25, 0.25]
            reduction: 'mean'
    optimizer:
        name: 'adam'
        kwargs: {lr: 0.0001}
    scheduler:
        name: exponential_lr
        kwargs: {gamma: 0.95}
    num_epochs: 20
    logging_dir: logs
    device: cuda
```

Model and training configs are parsed and fed into the trainers.

## Train and Evaluate

**TODO**: Write Python scripts for HPC.

## Model and Losses

- `src/models/particle_transformer.py`: Particle Transformer backbone and heads
- `src/loss/`: Losses including `conservation_loss` for masked reconstruction

## Notebooks

See `notebooks/01_ParT_demo.ipynb` for a quick, interactive demonstration of loading data and running the model.

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Project Structure

```
Hybrid_Transformer_Thanh_Nguyen/
├── src/
│   ├── configs/           # Dataclasses and loaders for YAML configs
│   ├── engine/            # Training/evaluation engine
│   ├── loss/              # Loss functions and registry
│   ├── models/            # Particle Transformer and variants
│   ├── optim/             # Optimizers/schedulers registries
│   └── utils/             # Callbacks, metrics, data, viz helpers
├── scripts/               # CLI tools: train/evaluate
├── tests/                 # Unit tests
├── logs/                  # Runs: best weights and CSV logs
├── data/                  # ROOT files (train/val/test splits)
└── assets/                # Figures and demo assets
```

## Notes

- Device selection is controlled by the YAML (`train.device`) and falls back to CUDA if available.
- YAML values should retain their numeric types; if you edit configs programmatically, ensure numbers aren’t coerced to strings (e.g., learning rates).