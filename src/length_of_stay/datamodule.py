import lightning as L
import torch
from torch.utils.data import DataLoader, Sampler

from src.length_of_stay.dataset import LengthOfStayDataset


class CustomSampler(Sampler):
    def __init__(self, specific_indices, rest_indices):
        super().__init__()
        self.specific_indices = specific_indices
        self.rest_indices = rest_indices

    def __iter__(self):
        specific_iter = iter(self.specific_indices)
        rest_iter = iter(self.rest_indices)
        return iter(list(specific_iter) + list(rest_iter))

    def __len__(self):
        return len(self.specific_indices) + len(self.rest_indices)


class LengthOfStayDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, hparams: dict):
        super().__init__()
        self.data_dir = data_dir
        self.save_hyperparameters(hparams)

    def setup(self, stage: str) -> None:
        self.train_data = LengthOfStayDataset(
            self.data_dir,
            "train_listfile.csv",
            self.hparams.one_hot,
            self.hparams.normalize,
            self.hparams.discretize,
        )
        self.val_data = LengthOfStayDataset(
            self.data_dir,
            "val_listfile.csv",
            self.hparams.one_hot,
            self.hparams.normalize,
            self.hparams.discretize,
        )
        self.test_data = LengthOfStayDataset(
            self.data_dir,
            "test_listfile.csv",
            self.hparams.one_hot,
            self.hparams.normalize,
            self.hparams.discretize,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            collate_fn=pad_batch,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            collate_fn=pad_batch,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            collate_fn=pad_batch,
            num_workers=2,
        )


def pad_batch(batch):
    data, labels = zip(*batch)
    data = [torch.tensor(x, dtype=torch.int) for x in data]
    max_length = max([len(x) for x in data])
    padded_batch = torch.zeros(len(data), max_length, data[0].shape[1], dtype=torch.int)
    for i, x in enumerate(data):
        try:
            ## pad from the start
            padded_batch[i, -len(x) :] = x
        except:
            print(i, x.shape)
            continue
    del data
    return padded_batch, torch.tensor(labels, dtype=torch.long)
