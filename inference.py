import os

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils import data as torch_data

from config import Config
from datasets import Dataset
from model import Unet as Model

DATA_DIRECTORY = Config.DATA_DIR
NUM_WORKERS = os.cpu_count() // 2
MRI_TYPES = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
SIZE = 256
NUM_IMAGES = 64
BATCH_SIZE = 4
SEED = 23456

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelfiles = ['FLAIR-e3-loss0.680-auc0.600.pth',
              'T1w-e2-loss0.674-auc0.620.pth',
              'T1wCE-e8-loss0.702-auc0.550.pth',
              'T2w-e4-loss0.735-auc0.530.pth']


def predict(modelfile, df, mri_type, split):
    print("Predict:", modelfile, mri_type, df.shape)
    df.loc[:, "MRI_Type"] = mri_type
    data_retriever = Dataset(data_dir=DATA_DIRECTORY,
                             paths=df.index.values,
                             mri_type=df["MRI_Type"].values,
                             split=split,
                             num_imgs=NUM_IMAGES,
                             img_size=SIZE)

    data_loader = torch_data.DataLoader(data_retriever,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)

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
            tmp_pred = torch.sigmoid(model(batch["X"].to(device, dtype=torch.float))).cpu().numpy().squeeze()
            if tmp_pred.size == 1:
                y_pred.append(tmp_pred)
            else:
                y_pred.extend(tmp_pred.tolist())
            ids.extend(batch["id"].numpy().tolist())

    preddf = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred})
    preddf = preddf.set_index("BraTS21ID")
    return preddf


df_valid = pd.read_csv('df_valid.csv', sep=';')
df_valid = df_valid.set_index("BraTS21ID")
df_valid["MGMT_pred"] = 0
for m, mtype in zip(modelfiles, MRI_TYPES):
    pred = predict(m, df_valid, mtype, "train")
    df_valid["MGMT_pred"] += pred["MGMT_value"]
df_valid["MGMT_pred"] /= len(modelfiles)
auc = roc_auc_score(df_valid["MGMT_value"], df_valid["MGMT_pred"])
print(f"Validation ensemble AUC: {auc:.4f}")


submission = pd.read_csv(f"{DATA_DIRECTORY}/sample_submission.csv", index_col="BraTS21ID")

submission["MGMT_value"] = 0
for m, mtype in zip(modelfiles, MRI_TYPES):
    pred = predict(m, submission, mtype, split="test")
    submission["MGMT_value"] += pred["MGMT_value"]

submission["MGMT_value"] /= len(modelfiles)
submission["MGMT_value"].to_csv("submission.csv")

# kaggle competitions submit -c journey-springfield \
# -f /content/gdrive/MyDrive/temp/simple_cnn_baseline.csv \
# -m "simple_cnn_baseline"
