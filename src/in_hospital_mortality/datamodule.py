import lightning as L
from torch.utils.data import DataLoader

from src.in_hospital_mortality.dataset import InHospitalMortalityDataset


class InHospitalMortalityDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, hparams: dict):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = hparams.batch_size
        self.one_hot = hparams.one_hot

    def setup(self, stage: str) -> None:
        self.train_data = InHospitalMortalityDataset(
            self.data_dir, "train_listfile.csv", self.one_hot
        )
        self.val_data = InHospitalMortalityDataset(self.data_dir, "val_listfile.csv", self.one_hot)
        self.test_data = InHospitalMortalityDataset(self.data_dir, "test_listfile.csv", self.one_hot)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
