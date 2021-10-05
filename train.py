import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (train_test_split)
from torch.nn import functional as torch_functional
from torch.utils import data as torch_data

from config import Config
from datasets import Dataset
from model import Model

DATA_DIRECTORY = Config.DATA_DIR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MRI_TYPES = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
SIZE = 256
NUM_IMAGES = 64
BATCH_SIZE = 4
N_EPOCHS = 10
SEED = 23456
LEARNING_RATE = 0.0008
LR_DECAY = 0.9

samples_to_exclude = [109, 123, 709]
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
                    "auc improved from {:.4f} to {:.4f}. Saved model to '{}'",
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
            X = batch["X"].to(self.device)
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
                X = batch["X"].to(self.device)
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


def train_mri_type(df_train, df_valid, mri_type):
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
    # display(df_train.head())
    # display(df_valid.head())

    train_data_retriever = Dataset(DATA_DIRECTORY,
                                   df_train["BraTS21ID"].values,
                                   df_train["MGMT_value"].values,
                                   df_train["MRI_Type"].values)

    valid_data_retriever = Dataset(DATA_DIRECTORY,
                                   df_valid["BraTS21ID"].values,
                                   df_valid["MGMT_value"].values,
                                   df_valid["MRI_Type"].values)

    train_loader = torch_data.DataLoader(train_data_retriever, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valid_loader = torch_data.DataLoader(valid_data_retriever, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch',
                           'unet',
                           in_channels=64,
                           out_channels=1,
                           init_features=32,
                           pretrained=False)
    # model = Model()
    model.to(device)

    # checkpoint = torch.load("best-model-all-auc0.555.pth")
    # model.load_state_dict(checkpoint["model_state_dict"])

    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = torch_functional.binary_cross_entropy_with_logits

    trainer = Trainer(
        model,
        device,
        optimizer,
        criterion
    )

    history = trainer.fit(epochs=10,
                          train_loader=train_loader,
                          valid_loader=valid_loader,
                          save_path=f"{mri_type}",
                          patience=10,
                          )

    return trainer.lastmodel


def predict(modelfile, df, mri_type, split):
    print("Predict:", modelfile, mri_type, df.shape)
    df.loc[:, "MRI_Type"] = mri_type
    data_retriever = Dataset(DATA_DIRECTORY,
                             df.index.values,
                             mri_type=df["MRI_Type"].values,
                             split=split)

    data_loader = torch_data.DataLoader(data_retriever,
                                        batch_size=4,
                                        shuffle=False,
                                        num_workers=8)

    model = Model()
    model.to(device)

    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_pred = []
    ids = []

    for e, batch in enumerate(data_loader, 1):
        print(f"{e}/{len(data_loader)}", end="\r")
        with torch.no_grad():
            tmp_pred = torch.sigmoid(model(batch["X"].to(device))).cpu().numpy().squeeze()
            if tmp_pred.size == 1:
                y_pred.append(tmp_pred)
            else:
                y_pred.extend(tmp_pred.tolist())
            ids.extend(batch["id"].numpy().tolist())

    preddf = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred})
    preddf = preddf.set_index("BraTS21ID")
    return preddf


modelfiles = None
if not modelfiles:
    modelfiles = [train_mri_type(df_train, df_valid, m) for m in MRI_TYPES]
    print(modelfiles)
