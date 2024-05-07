import os
import re
from argparse import ArgumentParser
from datetime import datetime

import lightning as L
import torch
import torchmetrics
import torchmetrics.classification
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from src.common.transformer_model import TransformerModel
from src.mimic.datamodule import MimicTimeSeriesDataModule
from src.multi_channel_transformer.multi_channel_transformer import (
    MultiChannelTransformerClassifier,
)



def init_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="simple_transformer",
        choices=["simple_transformer", "multi_channel_transformer", "lstm"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_checkpoint", type=str, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--padding",
        type=str,
        default=None,
        choices=["numerical", "categorical"],
    )
    parser.add_argument("--listfile_dir", type=str, default="data/length-of-stay")
    parser.add_argument("--data_dir", type=str, default="data/length-of-stay")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--processed_data_dir", type=str, default=None)
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

    return args


class LengthOfStayClassifier(L.LightningModule):
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
        elif hparams.model == "multi_channel_transformer":
            self.model = MultiChannelTransformerClassifier(
                vocabulary_size=132,
                embed_dim=hparams.embed_dim,
                output_dim=hparams.output_dim,
                number_of_channels=hparams.input_dim,
                number_of_layers=hparams.num_layers,
                number_of_heads=hparams.num_heads,
                dropout=hparams.dropout,
            )
        elif hparams.model == "lstm":
            self.lstm = torch.nn.LSTM(
                input_size=hparams.input_dim,
                hidden_size=hparams.embed_dim,
                num_layers=hparams.num_layers,
                batch_first=True,
                dropout=hparams.dropout,
            )
            self.dropout = torch.nn.Dropout(hparams.dropout)
            self.linear = torch.nn.Linear(hparams.embed_dim, hparams.output_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self.generic_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.generic_step(batch, batch_idx, "test")

    def generic_step(self, batch, batch_idx, prefix):
        inputs, labels = batch
        if self.hparams.model == "lstm":
            outputs, _ = self.lstm(inputs)
            outputs = self.dropout(outputs[:, -1, :])
            outputs = self.linear(outputs)
        else:
            outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        outputs = torch.softmax(outputs, dim=1)

        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(
            f"{prefix}_kappa", self.kappa(outputs, labels), on_epoch=True, prog_bar=True
        )
        self.log(
            f"{prefix}_acc", self.acc(outputs, labels), on_epoch=True, prog_bar=True
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3, verbose=True
            ),
            "monitor": "val_loss",
        }


if __name__ == '__main__':
    args = init_args()
    seed_everything(args.seed)
    classifier = LengthOfStayClassifier(args)
    time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    trainer = L.Trainer(
        log_every_n_steps=12,
        max_epochs=args.max_epochs,
        logger=TensorBoardLogger("models/length-of-stay", version=args.logdir),
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath="models/length-of-stay",
                filename=f"best_{args.model}_{time}",
                monitor="val_loss",
                mode="min",
            ),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            ),
        ],
    )
    # only the hours we need are stored on disk, so we need to take all of the measurements
    args.max_seq_len = 24 if args.aggregate else -1
    datamodule = MimicTimeSeriesDataModule("data/length-of-stay", args)

    if args.test_checkpoint is not None:
        trainer.test(
            model=classifier,
            ckpt_path=args.test_checkpoint,
            datamodule=datamodule,
        )
    else:
        trainer.fit(
            model=classifier,
            datamodule=datamodule,
        )
