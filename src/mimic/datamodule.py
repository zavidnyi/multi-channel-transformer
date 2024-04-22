import lightning as L
from torch.utils.data import DataLoader

from src.mimic.dataset import MimicTimeSeriesDataset


class MimicTimeSeriesDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, hparams: dict):
        super().__init__()
        self.data_dir = data_dir
        self.save_hyperparameters(hparams)

    def setup(self, stage: str) -> None:
        self.train_data = MimicTimeSeriesDataset(
            self.data_dir,
            "train_listfile.csv",
            self.hparams.max_seq_len,
            self.hparams.one_hot,
            self.hparams.normalize,
            self.hparams.discretize,
        )
        self.val_data = MimicTimeSeriesDataset(
            self.data_dir,
            "val_listfile.csv",
            self.hparams.max_seq_len,
            self.hparams.one_hot,
            self.hparams.normalize,
            self.hparams.discretize,
        )
        self.test_data = MimicTimeSeriesDataset(
            self.data_dir,
            "test_listfile.csv",
            self.hparams.max_seq_len,
            self.hparams.one_hot,
            self.hparams.normalize,
            self.hparams.discretize,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.hparams.batch_size, num_workers=29
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.hparams.batch_size, num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.hparams.batch_size, num_workers=29
        )
