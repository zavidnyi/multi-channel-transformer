import os.path

import lightning as L
import torch.nn.utils.rnn
from torch.utils.data import DataLoader

from src.mimic.constants import empty_measurements_row
from src.mimic.dataset import MimicTimeSeriesDataset


class MimicTimeSeriesDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.empty_measurements_row = torch.tensor(empty_measurements_row)

    def setup(self, stage: str) -> None:
        self.train_data = self.dataset_with_listfile("train_listfile.csv")
        self.val_data = self.dataset_with_listfile("val_listfile.csv")
        self.test_data = self.dataset_with_listfile("test_listfile.csv")

    def dataset_with_listfile(self, listfile: str) -> MimicTimeSeriesDataset:
        return MimicTimeSeriesDataset(
            self.hparams.processed_data_dir if self.hparams.processed_data_dir is not None else self.hparams.data_dir,
            self.hparams.processed_data_dir,
            os.path.join(self.hparams.listfile_dir, listfile),
            self.hparams.max_seq_len,
            self.hparams.one_hot,
            self.hparams.normalize,
            self.hparams.discretize,
            self.hparams.small
        )

    def padding_collate_fn(self, batch):
        if self.hparams.padding is None:
            return batch

        inputs, labels = zip(*batch)

        if self.hparams.padding == "numerical":
            inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)

        if self.hparams.padding == "categorical":
            # pad sequences with empty rows of measurements
            max_len_in_batch = max([len(x) for x in inputs])
            inputs = list(inputs)
            for i, x in enumerate(inputs):
                if len(x) < max_len_in_batch:
                    inputs[i] = torch.cat([x, self.empty_measurements_row.repeat(max_len_in_batch - len(x), 1)], dim=0)
            inputs = torch.stack(inputs)

        return inputs, torch.tensor(labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            collate_fn=self.padding_collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            collate_fn=self.padding_collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            # persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            collate_fn=self.padding_collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            # persistent_workers=True,
        )
