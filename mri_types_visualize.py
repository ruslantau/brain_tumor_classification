import os
import cv2
import pydicom
import numpy as np
import pandas as pd

from PIL import Image
from config import Config
from pathlib import Path
from datasets import crop_voxel, normalize_contrast, resize_voxel, get_image_plane

DATA_DIRECTORY = Config.DATA_DIR
WEIGHTS_DIR = Config.WEIGHTS_DIR
TEMP_DIR = Config.TEMP_DIR

NUM_WORKERS = os.cpu_count() - 2
MRI_TYPES = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
IMAGE_SIZE = 256


def get_voxel(data_root, split, study_id, scan_type, fix_plane: bool = True):
    dcm_dir = data_root.joinpath(split, study_id, scan_type)
    dcm_paths = sorted(dcm_dir.glob("*.dcm"), key=lambda x: int(x.stem.split("-")[-1]))

    imgs = []
    positions = []
    for dcm_path in dcm_paths:
        img = pydicom.dcmread(str(dcm_path))
        imgs.append(img.pixel_array)
        positions.append(img.ImagePositionPatient)

    plane = get_image_plane(img)
    voxel = np.stack(imgs)
    reordered = False

    if fix_plane:
        # reorder planes if needed and rotate voxel
        if plane == "Coronal":
            if positions[0][1] < positions[-1][1]:
                voxel = voxel[::-1]
                reordered = True
                print(f"{study_id} {scan_type} {plane} reordered")
            voxel = voxel.transpose((1, 0, 2))
        elif plane == "Sagittal":
            if positions[0][0] < positions[-1][0]:
                voxel = voxel[::-1]
                reordered = True
                print(f"{study_id} {scan_type} {plane} reordered")
            voxel = voxel.transpose((1, 2, 0))
            voxel = np.rot90(voxel, 2, axes=(1, 2))
        elif plane == "Axial":
            if positions[0][2] > positions[-1][2]:
                voxel = voxel[::-1]
                reordered = True
                print(f"{study_id} {scan_type} {plane} reordered")
            voxel = np.rot90(voxel, 2)
        else:
            raise ValueError(f"Unknown plane {plane}")
    return voxel, plane, reordered


def viz_voxels(result, out_dir):
    patient_id = out_dir.name
    for i in range(IMAGE_SIZE):
        collage_imgs = []
        for mri_type in MRI_TYPES:
            img = result[f"{mri_type}_fixed"][i]
            img_initial = result[f"{mri_type}_initial"][i]

            row = patients_df.loc[(patient_id, mri_type)]
            img = cv2.putText(img, f"Fixed {row['plane']} {'Reordered' if row['reordered'] else ''}", (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 255)
            img_initial = cv2.putText(img_initial, f"Original", (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 255)
            stacked_img = np.vstack([img_initial, img])
            stacked_img = cv2.putText(stacked_img, mri_type, (stacked_img.shape[1]//2 - 50, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 255)
            collage_imgs.append(stacked_img)

        stacked_img = np.hstack(collage_imgs)
        im = Image.fromarray(stacked_img)
        out_dir.mkdir(exist_ok=True, parents=True)
        im.save(out_dir / f"{i}.png")


split = 'train'
patients_df_path = Path("temp") / f'patients_df_{split}.csv'
if patients_df_path.exists():
    patients_df = pd.read_csv(patients_df_path, sep=';')
    patients_df['id'] = patients_df['id'].astype(str).str.zfill(5)
    patients_df = patients_df.set_index(['id', 'mri_type'], drop=False)
else:
    patients_df = pd.DataFrame(map(lambda x: [x.parent.name, x.name], list((Config.DATA_DIR / split).glob('*/*'))), columns=['id', 'mri_type'])
    patients_df['plane'] = None
    patients_df['reordered'] = False
    patients_df = patients_df.set_index(['id', 'mri_type'], drop=False)

for patient_id in patients_df['id'].unique():
    patient_dir = Path("temp") / 'temp' / str(patient_id)
    if patient_dir.exists():
        print(f"Patient {patient_id} was skipped")
        continue

    result = dict()
    for mri_type in MRI_TYPES:
        for flag in [False, True]:
            voxel, plane, reordered = get_voxel(data_root=DATA_DIRECTORY,
                                                study_id=patient_id,
                                                scan_type=mri_type,
                                                split=split,
                                                fix_plane=flag)
            voxel = crop_voxel(voxel)
            voxel = normalize_contrast(voxel)
            voxel = resize_voxel(voxel, size=IMAGE_SIZE)
            if flag:
                patients_df.loc[(patient_id, mri_type), 'plane'] = plane
                patients_df.loc[(patient_id, mri_type), 'reordered'] = reordered
                result.update({f"{mri_type}_fixed": voxel})
            else:
                result.update({f"{mri_type}_initial": voxel})

    viz_voxels(result, out_dir=patient_dir)

    patients_df.to_csv(patients_df_path, sep=';', index=False)
