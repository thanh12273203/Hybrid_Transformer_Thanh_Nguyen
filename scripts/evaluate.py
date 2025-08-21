import os
import argparse
from typing import Tuple, Dict, List

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

from src.configs import ParticleTransformerConfig, TrainConfig
from src.engine import Trainer, MaskedModelTrainer
from src.models import ParticleTransformer
from src.utils import accuracy_metric_ce
from src.utils.data import JetClassDataset, read_file
from src.utils.viz import plot_particle_reconstruction, plot_confusion_matrix, plot_roc_curve


def _load_split(
	split_dir: str,
	num_files: int,
	stride: int,
	max_num_particles: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	filepaths = [f for f in os.listdir(split_dir) if f.endswith('.root')]
	selected = filepaths[:num_files * stride:stride] if num_files > 0 else filepaths
	all_xp, all_xj, all_y = [], [], []
	for fname in selected:
		xp, xj, y = read_file(os.path.join(split_dir, fname), max_num_particles=max_num_particles)
		all_xp.append(xp)
		all_xj.append(xj)
		all_y.append(y)
	X_particles = np.concatenate(all_xp, axis=0)
	X_jets = np.concatenate(all_xj, axis=0)
	y = np.concatenate(all_y, axis=0)
	return X_particles, X_jets, y


def _compute_norm_stats(X_particles: np.ndarray) -> Tuple[List[bool], Dict[str, Tuple[float, float]]]:
	Xp = X_particles.transpose(0, 2, 1).reshape(-1, X_particles.shape[2])
	Xp = Xp[Xp[:, 0] != 0]
	pT_mean, pT_std = Xp[:, 0].mean(), Xp[:, 0].std()
	eta_mean, eta_std = Xp[:, 1].mean(), Xp[:, 1].std()
	phi_mean, phi_std = Xp[:, 2].mean(), Xp[:, 2].std()
	E_mean, E_std = Xp[:, 3].mean(), Xp[:, 3].std()
	normalize = [True, False, False, True]
	norm_dict = {
		'pT': (float(pT_mean), float(pT_std)),
		'eta': (float(eta_mean), float(eta_std)),
		'phi': (float(phi_mean), float(phi_std)),
		'energy': (float(E_mean), float(E_std))
	}
	return normalize, norm_dict


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Evaluate ParticleTransformer from YAML config')

    # Model and configurations arguments
	parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
	parser.add_argument('--best-model', type=str, default=None, help='Path to best model weights (.pt)')
	parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to restore trainer state')
	parser.add_argument('--output-dir', type=str, default='./assets/results', help='Directory to save plots')

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

	model_cfg = ParticleTransformerConfig.from_dict(cfg['model'])
	train_cfg = TrainConfig.from_dict(cfg['train'])

	# Read in the data
	max_num_particles = model_cfg.max_num_particles
	Xp_test, Xj_test, y_test = _load_split(os.path.join(args.data_root, args.test_split), args.num_files, args.stride, max_num_particles)
	normalize, norm_dict = _compute_norm_stats(Xp_test)

	# Create the dataset
	if model_cfg.mask:
		test_dataset = JetClassDataset(Xp_test, y_test, normalize, norm_dict, mask_mode='first')
	else:
		test_dataset = JetClassDataset(Xp_test, y_test, normalize, norm_dict, mask_mode=None)

	# Initialize the model
	device = torch.device(train_cfg.device) if train_cfg.device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = ParticleTransformer(config=model_cfg).to(device)

	# Trainer stub for evaluation convenience
	if model_cfg.mask:
		trainer = MaskedModelTrainer(model=model, train_dataset=test_dataset, val_dataset=test_dataset, test_dataset=test_dataset, config=train_cfg)
	else:
		trainer = Trainer(model=model, train_dataset=test_dataset, val_dataset=test_dataset, test_dataset=test_dataset, metric=accuracy_metric_ce, config=train_cfg)

	# Restore weights
	if args.best_model:
		model_path = args.best_model
		print(f"Loading best model: {model_path}")
		trainer.load_best_model(model_path)
	elif args.checkpoint and os.path.exists(args.checkpoint):
		print(f"Restoring from checkpoint: {args.checkpoint}")
		try:
			trainer.load_checkpoint(args.checkpoint)
		except Exception as e:
			print(f"Error loading checkpoint: {e}")

	# Evaluate and plot
	os.makedirs(args.output_dir, exist_ok=True)
	
	if model_cfg.mask:
		test_loss, y_true, y_pred = trainer.evaluate()
		# Save reconstruction plots
		plt.ioff()
		plot_particle_reconstruction(y_true, y_pred)
		fig = plt.gcf()
		fig.savefig(os.path.join(args.output_dir, 'reconstruction_2d.png'), dpi=150)
		plt.close(fig)
	else:
		test_loss, test_metric, y_true, y_pred = trainer.evaluate(loss_type='cross_entropy', plot=None)
		# Save ROC and confusion matrix
		plt.ioff()
		plot_roc_curve(y_true, y_pred)
		fig = plt.gcf()
		fig.savefig(os.path.join(args.output_dir, 'roc.png'), dpi=150)
		plt.close(fig)
		plot_confusion_matrix(y_true, y_pred, labels=[str(i) for i in range(y_true.shape[1])])
		fig = plt.gcf()
		fig.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=150)
		plt.close(fig)


if __name__ == '__main__':
	main()