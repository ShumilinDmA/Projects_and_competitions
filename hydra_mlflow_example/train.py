import os
import random
import numpy as np
from six.moves import urllib
from omegaconf import DictConfig

# Pytorch
import torch
from model import CNN
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl

# MLflow
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Hydra
import hydra
from hydra import utils


# To overcome request without header
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


@hydra.main(config_name="config.yaml", config_path='./configs')
def main(cfg: DictConfig) -> None:

    original_pwd = utils.get_original_cwd()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    loss = cfg.loss
    opt = cfg.optimizer.optimizer
    arch = cfg.architecture.architecture
    model = CNN(opt=opt, loss=loss, arch=arch)
    train_ds = MNIST(original_pwd, train=True, download=True, transform=transforms.ToTensor())
    valid_ds = MNIST(original_pwd, train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # Set path where will be mlruns
    mlflow.set_tracking_uri('file://' + original_pwd + '/mlruns')
    client = MlflowClient()

    try:
        experiment_id = client.create_experiment("MNIST experiment")
    except MlflowException:  # If such experiment already exist
        experiment_id = client.get_experiment_by_name("MNIST experiment").experiment_id
    client.set_experiment_tag(experiment_id, "Version", "Training")

    trainer = pl.Trainer(gpus=cfg.num_gpus,
                         progress_bar_refresh_rate=23,
                         max_epochs=cfg.max_epoch)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_param("loss_function", loss)
        mlflow.log_param("optimizer", opt)
        mlflow.log_param("achitecture", arch)
        mlflow.log_param('batch_size', cfg.batch_size)
        mlflow.log_param('max_epoch', cfg.max_epoch)
        mlflow.log_param('num_gpu', cfg.num_gpus)
        mlflow.log_param('seed', cfg.seed)
        mlflow.log_artifact(f"{os.getcwd()}/.hydra/config.yaml")
        trainer.fit(model, train_loader, valid_loader)
        trainer.save_checkpoint(f"{run.info.run_id}.ckpt")
        mlflow.log_artifact(f"{run.info.run_id}.ckpt")


if __name__ == "__main__":
    main()
