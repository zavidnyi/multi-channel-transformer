import os
import re
from argparse import ArgumentParser
from datetime import datetime

import lightning as L
import torch
import torchmetrics
import torchmetrics.classification
from lightning.pytorch.loggers import TensorBoardLogger

from src.common.transformer_model import TransformerModel
from src.mimic.datamodule import MimicTimeSeriesDataModule
from src.multi_channel_transformer.multi_channel_transformer import (
    MultiChannelTransformerClassifier,
)

torch.manual_seed(6469)
torch.set_float32_matmul_precision("medium")

parser = ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="simple_transformer",
    choices=["simple_transformer", "multi_channel_transformer"],
)
parser.add_argument("--test", action="store_true")
parser.add_argument("--one_hot", action="store_true")
parser.add_argument("--discretize", action="store_true")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--input_dim", type=int, default=48)
parser.add_argument("--embed_dim", type=int, default=64)
parser.add_argument("--output_dim", type=int, default=9)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--num_heads", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--pos_weight", type=float, default=2.0)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--head_hidden_layers", type=int, default=2)
parser.add_argument("--head_hidden_dim", type=int, default=64)

args = parser.parse_args()

args.logdir = os.path.join(
    "logs",
    "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(
            (
                "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                for k, v in sorted(vars(args).items())
            )
        ),
    ),
)


class LightningSimpleTransformerClassifier(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.kappa = torchmetrics.classification.MulticlassCohenKappa(
            num_classes=9, weights="linear"
        )
        self.acc = torchmetrics.classification.MulticlassAccuracy(num_classes=9)
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        if hparams.model == "simple_transformer":
            self.model = TransformerModel(
                input_dim=hparams.input_dim,
                embed_dim=hparams.embed_dim,
                output_dim=hparams.output_dim,
                num_layers=hparams.num_layers,
                num_heads=hparams.num_heads,
                dropout=hparams.dropout,
                head_hidden_layers=hparams.head_hidden_layers,
                head_hidden_dimension=hparams.head_hidden_dim,
            )
        else:
            self.model = MultiChannelTransformerClassifier(
                vocabulary_size=132,
                embed_dim=hparams.embed_dim,
                output_dim=hparams.output_dim,
                number_of_channels=hparams.input_dim,
                number_of_layers=hparams.num_layers,
                number_of_heads=hparams.num_heads,
                dropout=hparams.dropout,
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        outputs = torch.softmax(outputs, dim=1)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log(
            "train_kappa", self.kappa(outputs, labels), on_epoch=True, prog_bar=True
        )
        self.log("train_acc", self.acc(outputs, labels), on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)
        outputs = torch.softmax(outputs, dim=1)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_kappa", self.kappa(outputs, y), on_epoch=True, prog_bar=True)
        self.log("val_acc", self.acc(outputs, y), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)
        outputs = torch.softmax(outputs, dim=1)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_kappa", self.kappa(outputs, y), on_epoch=True, prog_bar=True)
        self.log("test_acc", self.acc(outputs, y), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.1, patience=3, verbose=True
            ),
            "monitor": "val_kappa",
        }


classifier = LightningSimpleTransformerClassifier(args)
trainer = L.Trainer(
    log_every_n_steps=12,
    max_epochs=args.max_epochs,
    logger=TensorBoardLogger("models/length-of-stay", version=args.logdir),
    callbacks=[
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath="models/length-of-stay",
            filename="best",
            monitor="val_kappa",
            mode="max",
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_kappa",
            patience=5,
            mode="max",
        ),
    ],
)
args.max_seq_len = 24
datamodule = MimicTimeSeriesDataModule("data/length-of-stay", args)

if args.test:
    trainer.test(
        model=classifier,
        ckpt_path="models/length-of-stay/best-v33.ckpt",
        datamodule=datamodule,
    )
else:
    trainer.fit(
        model=classifier,
        datamodule=datamodule,
    )
