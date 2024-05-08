import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from membrain_pick.dataloading.diffusionnet_dataset import MemSegDiffusionNetDataset
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np


def custom_collate(batch):
    """Custom collate function to handle a complex data structure.
    
    Each sample is a dictionary containing numpy arrays and another dictionary
    with sparse matrices. Since we're using a batch size of 1, this function
    simplifies the handling of these structures.

    Args:
        batch: A list of samples, where each sample is the complex data structure
               described above.
    
    Returns:
        Processed batch ready for model input.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Unpack the single sample from the batch
    sample = batch[0]
    # Initialize a new dictionary to store the processed sample
    processed_sample = {}

    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            # Convert numpy arrays to tensors
            processed_sample[key] = torch.tensor(value).to(device)
        elif isinstance(value, dict):
            # For the nested dictionary, we assume it contains sparse matrices
            # and pass it through directly without modifications
            processed_sample[key] = {subkey: subvalue.to(device) for subkey, subvalue in value.items()}
        else:
            # Directly pass through any other types of values
            processed_sample[key] = value

    return processed_sample



class MemSegDiffusionNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_folder_train: str,
        csv_folder_val: str,
        csv_folder_test: Optional[str] = None,
        load_n_sampled_points: int = 2000,
        is_single_mb: bool = False, # For testing single membrane
        overfit: bool = False,
        force_recompute: bool = False,
        overfit_mb: bool = False,
        allpos:bool = False,
        use_psii: bool = True,
        use_b6f: bool = False,
        use_uk: bool = False,
        cache_dir: Optional[str] = None,
        augment_all: bool = True,
        aug_prob_to_one: bool = False,
        pixel_size: float = 1.0,
        max_tomo_shape: int = 928,
        k_eig: int = 128,
        batch_size: int = 1,
        num_workers: int = 16,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.csv_folder_train = csv_folder_train
        self.csv_folder_val = csv_folder_val
        self.csv_folder_test = csv_folder_test
        self.is_single_mb = is_single_mb

        self.load_n_sampled_points = load_n_sampled_points
        self.overfit = overfit
        self.force_recompute = force_recompute
        self.overfit_mb = overfit_mb
        self.cache_dir = cache_dir
        self.augment_all = augment_all
        self.pixel_size = pixel_size
        self.max_tomo_shape = max_tomo_shape
        self.allpos = allpos
        self.use_psii = use_psii
        self.use_b6f = use_b6f
        self.use_uk = use_uk
        self.aug_prob_to_one = aug_prob_to_one


        self.k_eig = k_eig
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Placeholder for the datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            if self.train_dataset is None:
                self.train_dataset = MemSegDiffusionNetDataset(
                    csv_folder=self.csv_folder_train,
                    train=True,
                    train_pct=1.0,
                    load_only_sampled_points=self.load_n_sampled_points,
                    max_tomo_shape=self.max_tomo_shape,
                    overfit=self.overfit,
                    force_recompute=self.force_recompute,
                    overfit_mb=self.overfit_mb,
                    cache_dir=self.cache_dir,
                    augment_all=self.augment_all,
                    aug_prob_to_one=self.aug_prob_to_one,
                    pixel_size=self.pixel_size,
                    k_eig=self.k_eig,
                    allpos=self.allpos,
                    use_psii=self.use_psii,
                    use_b6f=self.use_b6f,
                    use_uk=self.use_uk,
                )
            if self.val_dataset is None:
                self.val_dataset = MemSegDiffusionNetDataset(
                    csv_folder=self.csv_folder_val,
                    train=False,
                    train_pct=0.0,
                    load_only_sampled_points=self.load_n_sampled_points,
                    max_tomo_shape=self.max_tomo_shape,
                    overfit=self.overfit,
                    force_recompute=self.force_recompute,
                    overfit_mb=self.overfit_mb,
                    cache_dir=self.cache_dir,
                    augment_all=self.augment_all,
                    pixel_size=self.pixel_size,
                    k_eig=self.k_eig,
                    allpos=self.allpos,
                    use_psii=self.use_psii,
                    use_b6f=self.use_b6f,
                    use_uk=self.use_uk,
                )
            self.parameter_len = self.train_dataset.get_parameter_len()
        elif stage == 'test' or stage is None:
            if self.test_dataset is None:
                self.test_dataset = MemSegDiffusionNetDataset(
                    csv_folder=self.csv_folder_test,
                    train=False,
                    train_pct=0.0,
                    is_single_mb=self.is_single_mb,
                    load_only_sampled_points=self.load_n_sampled_points,
                    max_tomo_shape=self.max_tomo_shape,
                    overfit=self.overfit,
                    force_recompute=self.force_recompute,
                    overfit_mb=self.overfit_mb,
                    cache_dir=self.cache_dir,
                    augment_all=self.augment_all,
                    pixel_size=self.pixel_size,
                    k_eig=self.k_eig,
                    allpos=self.allpos,
                    use_psii=self.use_psii,
                    use_b6f=self.use_b6f,
                    use_uk=self.use_uk,
                    # test_mb="T1S1M18"
                )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate,
        )