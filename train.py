import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (train_test_split)
from torch.nn import functional as F
from torch.utils import data as torch_data

from config import Config
from datasets import Dataset
from model import Unet

DATA_DIRECTORY = Config.DATA_DIR
WEIGHTS_DIR = Config.WEIGHTS_DIR
TEMP_DIR = Config.TEMP_DIR

NUM_WORKERS = os.cpu_count() - 2
MRI_TYPES = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
IMAGE_SIZE = 256
BATCH_SIZE = 8
N_EPOCHS = 10
SEED = 23456
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(in_channels=IMAGE_SIZE, out_channels=1, init_features=32)

samples_to_exclude = [109, 123, 709]  # too much loosed samples
train_df = pd.read_csv(f"{DATA_DIRECTORY}/train_labels.csv")
train_df = train_df[~train_df.BraTS21ID.isin(samples_to_exclude)]

df_train, df_valid = train_test_split(train_df,
                                      test_size=0.2,
                                      random_state=SEED,
                                      stratify=train_df["MGMT_value"])


class Trainer:
    def __init__(self, model, device, optimizer, criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None

    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_time = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time = self.valid_epoch(valid_loader)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s            ",
                n_epoch, train_loss, train_time
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_auc, valid_time
            )

            # if True:
            # if self.best_valid_score < valid_auc:
            if self.best_valid_score > valid_loss:
                self.save_model(n_epoch, save_path, valid_loss, valid_auc)
                self.info_message(
                    "loss improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_score, valid_loss, self.lastmodel
                )
                self.best_valid_score = valid_loss
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device, dtype=torch.float)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)

            loss = self.criterion(outputs, targets)
            loss.backward()

            sum_loss += loss.detach().item()

            self.optimizer.step()

            message = 'Train Step {}/{}, train_loss: {:.4f}'
            self.info_message(message, step, len(train_loader), sum_loss / step, end="\r")

        return sum_loss / len(train_loader), int(time.time() - t)

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device, dtype=torch.float)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()
                y_all.extend(batch["y"].tolist())
                outputs_all.extend(torch.sigmoid(outputs).tolist())

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            self.info_message(message, step, len(valid_loader), sum_loss / step, end="\r")

        y_all = [1 if x > 0.5 else 0 for x in y_all]
        auc = roc_auc_score(y_all, outputs_all)

        return sum_loss / len(valid_loader), auc, int(time.time() - t)

    def save_model(self, n_epoch, save_path, loss, auc):
        self.lastmodel = f"{save_path}-e{n_epoch}-loss{loss:.3f}-auc{auc:.3f}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


def train_mri_type(model, df_train, df_valid, mri_type):
    if mri_type == "all":
        train_list = []
        valid_list = []
        for mri_type in MRI_TYPES:
            df_train.loc[:, "MRI_Type"] = mri_type
            train_list.append(df_train.copy())
            df_valid.loc[:, "MRI_Type"] = mri_type
            valid_list.append(df_valid.copy())

        df_train = pd.concat(train_list)
        df_valid = pd.concat(valid_list)
    else:
        df_train.loc[:, "MRI_Type"] = mri_type
        df_valid.loc[:, "MRI_Type"] = mri_type

    print(df_train.shape, df_valid.shape)

    train_data_retriever = Dataset(data_dir=DATA_DIRECTORY,
                                   paths=df_train["BraTS21ID"].values,
                                   targets=df_train["MGMT_value"].values,
                                   mri_types=df_train["MRI_Type"].values,
                                   image_size=IMAGE_SIZE
                                   )

    valid_data_retriever = Dataset(data_dir=DATA_DIRECTORY,
                                   paths=df_valid["BraTS21ID"].values,
                                   targets=df_valid["MGMT_value"].values,
                                   mri_types=df_valid["MRI_Type"].values,
                                   image_size=IMAGE_SIZE
                                   )

    train_loader = torch_data.DataLoader(dataset=train_data_retriever,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=NUM_WORKERS)
    valid_loader = torch_data.DataLoader(dataset=valid_data_retriever,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=NUM_WORKERS)

    # model = Model()
    model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    criterion = F.binary_cross_entropy_with_logits
    trainer = Trainer(model, device, optimizer, criterion)

    trainer.fit(epochs=N_EPOCHS,
                train_loader=train_loader,
                valid_loader=valid_loader,
                save_path=WEIGHTS_DIR / f"{model.__class__.__name__}_{mri_type}",
                patience=10)

    return trainer.lastmodel



modelfiles = None
if not modelfiles:
    modelfiles = [train_mri_type(model, df_train, df_valid, m) for m in MRI_TYPES]
    print(modelfiles)

df_valid.to_csv(TEMP_DIR / 'df_valid.csv', sep=';', index=False)
