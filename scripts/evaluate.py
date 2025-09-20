import yaml
import argparse

import torch
import torch.multiprocessing as mp

from src.configs import LorentzParTConfig, TrainConfig
from src.engine import Trainer, MaskedModelTrainer
from src.models import LorentzParT
from src.utils import accuracy_metric_ce, set_seed, setup_ddp, cleanup_ddp
from src.utils.data import JetClassDataset, compute_norm_stats, load_npy_data
from src.utils.viz import plot_particle_reconstruction, plot_confusion_matrix, plot_roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LorentzParT from YAML config")

    # Model and configurations arguments
    parser.add_argument('--config-path', type=str, default='./configs/train_LorentzParT.yaml', help="Path to YAML config")
    parser.add_argument('--best-model-path', type=str, default='./logs/LorentzParT/best/pretrained_equilinear_clf.pt', help="Path to best model weights")

    # Data loading arguments
    parser.add_argument('--test-data-dir', type=str, default='./data/test_20M', help="Test data folder")

    return parser.parse_args()


@torch.no_grad()
def main(
    rank: int,
    world_size: int,
    config_path: str,
    best_model_path: str = None,
    test_data_dir: str = './data/test_20M'
):
    # Load the YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = LorentzParTConfig.from_dict(config['model'])
    train_config = TrainConfig.from_dict(config['train'])

    # Initialize multi-GPU processing
    setup_ddp(rank, world_size)

    # Read in the data
    X_test, _, y_test = load_npy_data(test_data_dir)
    normalize = [True, False, False, True]
    norm_dict = compute_norm_stats(X_test)

    # Create the dataset
    if model_config.mask:
        test_dataset = JetClassDataset(X_test, y_test, normalize, norm_dict, mask_mode='first')
    else:
        test_dataset = JetClassDataset(X_test, y_test, normalize, norm_dict, mask_mode=None)

    # Initialize the model
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = LorentzParT(config=model_config).to(device)

    # Trainer stub for evaluation convenience
    if model_config.mask:
        trainer = MaskedModelTrainer(
            model=model,
            train_dataset=test_dataset,
            val_dataset=test_dataset,
            test_dataset=test_dataset,
            device=device,
            config=train_config
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=test_dataset,
            val_dataset=test_dataset,
            test_dataset=test_dataset,
            device=device,
            metric=accuracy_metric_ce,
            config=train_config
        )

    # Load the best model
    print(f"Loading best model: {best_model_path}")
    trainer.load_best_model(best_model_path)

    # Evaluate the model
    if model_config.mask:
        test_loss, test_metric, y_true, y_pred = trainer.evaluate(plot_particle_reconstruction)
    else:
        test_loss, test_metric, y_true, y_pred = trainer.evaluate(
            loss_type='cross_entropy',
            plot=[plot_roc_curve, plot_confusion_matrix]
        )

    # Clean up distributed processing
    cleanup_ddp()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Reproducibility settings
    set_seed(42)

    # Multi-GPU processing
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            main,
            args(
                args.config_path,
                args.best_model_path,
                args.test_data_dir
            ),
            nprocs=world_size
        )
    else:
        # 1 GPU or CPU: run the same code on rank 0
        main(
            rank=0,
            world_size=1,
            config_path=args.config_path,
            best_model_path=args.best_model_path,
            test_data_dir=args.test_data_dir
        )