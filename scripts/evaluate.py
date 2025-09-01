import os
import yaml
import argparse
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import torch

from src.configs import LorentzParTConfig, TrainConfig
from src.engine import Trainer, MaskedModelTrainer
from src.models import LorentzParT
from src.utils import accuracy_metric_ce
from src.utils.data import JetClassDataset, compute_norm_stats, read_file
from src.utils.viz import plot_particle_reconstruction, plot_confusion_matrix, plot_roc_curve


def _load_split(
    split_dir: str,
    num_files: int,
    stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    filepaths = [f for f in os.listdir(split_dir) if f.endswith('.root')]
    selected = filepaths[:num_files * stride:stride]
    all_xp, all_xj, all_y = [], [], []
    
    for fname in selected:
        xp, xj, y = read_file(os.path.join(split_dir, fname))
        all_xp.append(xp)
        all_xj.append(xj)
        all_y.append(y)
        
    X_particles = np.concatenate(all_xp, axis=0)
    X_jets = np.concatenate(all_xj, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    return X_particles, X_jets, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate ParticleTransformer from YAML config')

    # Model and configurations arguments
    parser.add_argument('--config', type=str, default='./configs/train_LorentzParT.yaml', help='Path to YAML config')
    parser.add_argument('--best-model', type=str, default='./logs/LorentzParT/best/pretrained_equilinear_clf.pt', help='Path to best model weights (.pt)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to restore trainer state')

    # Data loading arguments
    parser.add_argument('--data-root', type=str, default='./data', help='Dataset root folder')
    parser.add_argument('--test-split', type=str, default='val_5M', help='Split to evaluate on')
    parser.add_argument('--num-files', type=int, default=10, help='Limit number of ROOT files to read (0=all)')
    parser.add_argument('--stride', type=int, default=5, help='Stride when selecting ROOT files')

    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    # Parse command-line arguments
    args = parse_args()

    # Reproducibility settings
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the YAML files
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    model_cfg = LorentzParTConfig.from_dict(cfg['model'])
    train_cfg = TrainConfig.from_dict(cfg['train'])

    # Read in the data
    X_particles, _, y = _load_split(os.path.join(args.data_root, args.test_split), args.num_files, args.stride)
    normalize = [True, False, False, True]
    norm_dict = compute_norm_stats(X_particles)

    # Randomly split the data into training, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_particles, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    # Create the dataset
    if model_cfg.mask:
        train_dataset = JetClassDataset(X_train, y_train, normalize, norm_dict, mask_mode='random')
        val_dataset = JetClassDataset(X_val, y_val, normalize, norm_dict, mask_mode='random')
        test_dataset = JetClassDataset(X_test, y_test, normalize, norm_dict, mask_mode='first')
    else:
        train_dataset = JetClassDataset(X_train, y_train, normalize, norm_dict, mask_mode=None)
        val_dataset = JetClassDataset(X_val, y_val, normalize, norm_dict, mask_mode=None)
        test_dataset = JetClassDataset(X_test, y_test, normalize, norm_dict, mask_mode=None)

    # Initialize the model
    device = torch.device(train_cfg.device) if train_cfg.device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = LorentzParT(config=model_cfg).to(device)

    # Trainer stub for evaluation convenience
    if model_cfg.mask:
        trainer = MaskedModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            config=train_cfg
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            metric=accuracy_metric_ce,
            config=train_cfg
        )

    # Restore weights
    if model_cfg.weights is not None and os.path.exists(model_cfg.weights):
        model_path = model_cfg.weights
        print(f"Loading best model: {model_path}")
        trainer.load_best_model(model_path)
    elif args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Restoring from checkpoint: {args.checkpoint}")

        try:
            trainer.load_checkpoint(args.checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Evaluate the model
    if model_cfg.mask:
        test_loss, test_metric, y_true, y_pred = trainer.evaluate(plot_particle_reconstruction)
    else:
        test_loss, test_metric, y_true, y_pred = trainer.evaluate(
            loss_type='cross_entropy',
            plot=[plot_roc_curve, plot_confusion_matrix]
        )
        
    # Add custom visualization here


if __name__ == '__main__':
    main()