from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset


class JetClassDataset(Dataset):
    """
    A PyTorch dataset class to use JetClass for classification and self-supervised learning.

    Parameters
    ----------
    X_particles: np.ndarray
        The input particle features.
    y: np.ndarray
        The target labels.
    normalize: List[bool], optional
        A list indicating which features to normalize. Default is `[True, False, False, True]` (normalize pT and energy).
    norm_dict: Dict[str, Tuple[float, float]], optional
        A dictionary containing the normalization parameters (mean, std) for each feature.
    mask_mode: str, optional
        The masking mode to use. If not specified, no masking is applied.

    Returns
    -------
    Tuple[Tensor, ...]
        For self-supervised learning: a tuple containing the following elements:
        - The masked particle features tensor of shape (max_num_particles, num_particle_features).
        - The masked target labels tensor of shape (num_particle_features,).
        - The mask index tensor of shape (1,).
        
        For classification: a tuple containing the following elements:
        - The particle features tensor of shape (max_num_particles, num_particle_features).
        - The target labels tensor of shape (num_classes,).

    .. References::
        Huilin Qu, Congqiao Li, and Sitian Qian.
        [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772).
        In *Proceedings of the 39th International Conference on Machine Learning*, pages 18281-18292, 2022.

        Huilin Qu, Congqiao Li, and Sitian Qian.
        [JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics](https://zenodo.org/records/6619768).
        *Zenodo*, 2022.
    """
    def __init__(
        self,
        X_particles: np.ndarray,
        y: np.ndarray,
        normalize: List[bool] = [True, False, False, True],  # [pT, eta, phi, energy]
        norm_dict: Dict[str, Tuple[float, float]] = None,
        mask_mode: str = None
    ):
        self.X_particles = X_particles
        self.y = y
        self.normalize = normalize
        self.norm_dict = norm_dict
        self.mask_mode = mask_mode

    def __len__(self) -> int:
        return len(self.X_particles)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        particles = self.X_particles[idx].T  # (max_num_particles, num_particle_features)

        if self.mask_mode is not None:
            particles = particles.copy()
            masked_particles, masked_targets, mask_idx = self._mask_particle(particles, self.mask_mode)

            # Normalize features (in-place)
            feature_names = ['pT', 'eta', 'phi', 'energy']

            if self.norm_dict is not None:
                for i, feature in enumerate(feature_names):
                    if self.normalize[i]:
                        mean, std = self.norm_dict[feature]

                        if i == 0 or i == 3:  # pT or energy where values are strictly positive
                            masked_particles[:, i] = masked_particles[:, i] / mean
                            masked_targets[:, i] = masked_targets[:, i] / mean
                        else:
                            masked_particles[:, i] = (masked_particles[:, i] - mean) / std
                            masked_targets[:, i] = (masked_targets[:, i] - mean) / std

            return (
                torch.tensor(masked_particles, dtype=torch.float32),  # (max_num_particles, num_particle_features)
                torch.tensor(masked_targets.squeeze(0), dtype=torch.float32),  # (num_particle_features,)
                torch.tensor(mask_idx, dtype=torch.int64)  # (1,)
            )
        else:
            # Normalize features (in-place)
            feature_names = ['pT', 'eta', 'phi', 'energy']
            if self.norm_dict is not None:
                for i, feature in enumerate(feature_names):
                    if self.normalize[i]:
                        mean, std = self.norm_dict[feature]

                        if i == 0 or i == 3:  # pT or energy where values are strictly positive
                            particles[:, i] = particles[:, i] / mean
                        else:
                            particles[:, i] = (particles[:, i] - mean) / std

            tensor = torch.from_numpy(particles).float()  # (max_num_particles, num_particle_features)
            label = torch.from_numpy(self.y[idx]).float()  # (num_classes,)

            return tensor, label
        
    def _mask_particle(self, particles: np.ndarray, mode: str = 'random') -> Tuple[np.ndarray, np.ndarray, int]:
        valid_idx = np.where(np.any(particles != 0, axis=1))[0]
        
        if mode == 'random':
            mask_idx = np.array([np.random.choice(valid_idx)])
        elif mode == 'biased':
            total = np.sum(1 / (np.arange(0, particles.shape[0]) + 1))
            mask_idx = 127
            u, w = 0, 1

            while (u < w) or (mask_idx not in valid_idx):
                u = np.random.uniform(0, 1)
                mask_idx = np.random.randint(0, particles.shape[0])
                w = (1 / (mask_idx + 1)) / total
                
            mask_idx = np.array([mask_idx])
        elif mode == 'first':
            mask_idx = valid_idx[:1]
        
        masked_particles = particles.copy()
        masked_targets = masked_particles[mask_idx, :].copy()
        masked_particles[mask_idx, :] = 0.0

        return masked_particles, masked_targets, mask_idx