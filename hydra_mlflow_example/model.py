import torch
from torch import nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.metrics.classification import F1
import mlflow


class CNN(pl.LightningModule):
    def __init__(self, opt: DictConfig, loss: str, arch: DictConfig) -> None:

        super().__init__()
        self.opt_conf = opt
        self.loss_conf = loss
        self.num_classes = arch.num_classes
        self.criterion = self._get_loss_function()
        self.f1_score_func = F1(num_classes=self.num_classes)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=arch.Conv1.input,
                      out_channels=arch.Conv1.output,
                      kernel_size=arch.Conv1.kernel,
                      stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=arch.Conv2.input,
                      out_channels=arch.Conv2.output,
                      kernel_size=arch.Conv2.kernel,
                      stride=1, padding=0),
            nn.BatchNorm2d(arch.Conv2.output),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=arch.MaxPool1.kernel, stride=arch.MaxPool1.stride),

            nn.Conv2d(in_channels=arch.Conv3.input,
                      out_channels=arch.Conv3.output,
                      kernel_size=arch.Conv3.kernel,
                      stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=arch.Conv4.input,
                      out_channels=arch.Conv4.output,
                      kernel_size=arch.Conv4.kernel,
                      stride=1, padding=0),
            nn.BatchNorm2d(arch.Conv4.output),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=arch.MaxPool2.kernel, stride=arch.MaxPool2.stride),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(arch.Linear.input, arch.Linear.output),
            nn.Linear(arch.Linear.output, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.clf(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        y_pred = torch.argmax(torch.sigmoid(pred).detach(), dim=1)
        return {"loss": loss, "y_pred": y_pred, "y_true": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        y_pred = torch.argmax(torch.sigmoid(pred).detach(), dim=1)
        return {"val_loss": loss, "y_pred": y_pred, "y_true": y}

    def _get_loss_function(self):
        if self.loss_conf == 'CEloss':
            return nn.CrossEntropyLoss()
        elif self.loss_confs == 'BCEloss':
            return nn.BCEWithLogitsLoss()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.opt_conf.name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.opt_conf.lr,
                                         betas=(self.opt_conf.betas.one, self.opt_conf.betas.two),
                                         weight_decay=self.opt_conf.weight_decay)
        elif self.opt_conf.name == 'AdamW':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.opt_conf.lr,
                                         betas=(self.opt_conf.betas.one, self.opt_conf.betas.two),
                                         weight_decay=self.opt_conf.weight_decay)
        elif self.opt_conf.name == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.opt_conf.lr,
                                            alpha=self.opt_conf.alpha, weight_decay=self.opt_conf.weight_decay,
                                            momentum=self.opt_conf.momentum)
        elif self.opt_conf.name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.opt_conf.lr,
                                        momentum=self.opt_conf.momentum, weight_decay=self.opt_conf.weight_decay,
                                        nesterov=self.opt_conf.nesterov)

        return optimizer

    def training_epoch_end(self, outputs):
        y_hat = torch.cat([x['y_pred'].view(-1) for x in outputs])
        y_true = torch.cat([x['y_true'].view(-1) for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().numpy()

        # Mlflow works with original python numbers
        avg_loss = float(avg_loss)
        f1_metric = float(self.f1_score_func(y_hat, y_true).numpy())

        print(f'Train: \n   Loss: {avg_loss}, F1 metric: {f1_metric}')
        mlflow.log_metric(key="train_loss", value=avg_loss)
        mlflow.log_metric("train_f1_score", f1_metric)

    def validation_epoch_end(self, outputs):
        y_hat = torch.cat([x['y_pred'].view(-1) for x in outputs])
        y_true = torch.cat([x['y_true'].view(-1) for x in outputs])
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().numpy()

        # Mlflow works with original python numbers
        avg_loss = float(avg_loss)
        f1_metric = float(self.f1_score_func(y_hat, y_true).numpy())

        print(f'Valid: \n   Loss: {avg_loss}, F1 metric: {f1_metric}')
        mlflow.log_metric(key="valid_loss", value=avg_loss)
        mlflow.log_metric("valid_f1_score", f1_metric)
