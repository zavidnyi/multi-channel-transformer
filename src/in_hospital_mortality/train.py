from argparse import ArgumentParser

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F

from src.in_hospital_mortality.datamodule import InHospitalMortalityDataModule
from src.sepsis.transformer_model import TransformerModel

torch.manual_seed(6469)

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--input_dim", type=int, default=48)
parser.add_argument("--embed_dim", type=int, default=64)
parser.add_argument("--output_dim", type=int, default=1)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--pos_weight", type=float, default=2.0)
parser.add_argument("--dropout", type=float, default=0.5)

args = parser.parse_args()


class LightningSimpleTransformerClassifier(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.aucpr = torchmetrics.classification.BinaryAveragePrecision()
        self.auroc = torchmetrics.classification.BinaryAUROC()
        self.recall = torchmetrics.classification.BinaryRecall()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        # self.model = torch.nn.LSTM(
        #     input_size=hparams.input_dim,
        #     hidden_size=hparams.embed_dim,
        #     num_layers=hparams.num_layers,
        #     batch_first=True,
        # )
        # self.model1 = torch.nn.Linear(hparams.embed_dim, hparams.output_dim)
        self.model = TransformerModel(
            input_dim=hparams.input_dim,
            embed_dim=hparams.embed_dim,
            output_dim=hparams.output_dim,
            num_layers=hparams.num_layers,
            num_heads=hparams.num_heads,
            dropout=hparams.dropout,
        )
        # self.model = MultiChannelTransformerClassifier(
        #     hparams.input_dim,
        #     1,
        #     hparams.output_dim,
        #     hparams.input_dim,
        #     hparams.num_layers,
        #     hparams.num_heads,
        # )

    def forward(self, x):
        return self.model(x)

    def loss_fn(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(
            y_hat,
            y,
            pos_weight=torch.tensor(self.hparams.pos_weight).to(self.device),
        )

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # input is (B, 48 features, 48 hours)
        # labels is (B, 0/1)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        outputs = torch.sigmoid(outputs)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.accuracy(outputs, labels), on_epoch=True)
        self.log("train_recall", self.recall(outputs, labels), on_epoch=True)
        self.log("train_auc", self.auroc(outputs, labels), on_epoch=True)
        self.log(
            "train_aucpr", self.aucpr(outputs, labels.to(torch.int)), on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", self.accuracy(y_hat, y), on_epoch=True)
        self.log("val_recall", self.recall(y_hat, y), on_epoch=True)
        self.log("val_auc", self.auroc(y_hat, y), on_epoch=True)
        self.log("val_aucpr", self.aucpr(y_hat, y.to(torch.int)), on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.accuracy(y_hat, y), on_epoch=True)
        self.log("test_recall", self.recall(y_hat, y), on_epoch=True)
        self.log("test_auc", self.auroc(y_hat, y), on_epoch=True)
        self.log("test_aucpr", self.aucpr(y_hat, y.to(torch.int)), on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


classifier = LightningSimpleTransformerClassifier(args)
trainer = L.Trainer(
    max_epochs=args.max_epochs,
    default_root_dir="models/in_hospital_mortality_simple",
    logger=TensorBoardLogger("models/in_hospital_mortality_simple"),
)
datamodule = InHospitalMortalityDataModule(
    "data/in-hospital-mortality", args.batch_size
)
trainer.fit(
    model=classifier,
    datamodule=datamodule,
)
trainer.test(datamodule=datamodule)
