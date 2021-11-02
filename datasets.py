import cv2
import torch
import pydicom
import numpy as np

from torch.utils import data as torch_data


def get_image_plane(data):
    x1, y1, _, x2, y2, _ = [round(j) for j in data.ImageOrientationPatient]
    cords = [x1, y1, x2, y2]

    if cords == [1, 0, 0, 0]:
        return 'Coronal'
    elif cords == [1, 0, 0, 1]:
        return 'Axial'
    elif cords == [0, 1, 0, 0]:
        return 'Sagittal'
    else:
        return 'Unknown'


def get_voxel(data_root, split, study_id, scan_type):
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

    # reorder planes if needed and rotate voxel
    if plane == "Coronal":
        if positions[0][1] < positions[-1][1]:
            voxel = voxel[::-1]
            print(f"{study_id} {scan_type} {plane} reordered")
        voxel = voxel.transpose((1, 0, 2))
    elif plane == "Sagittal":
        if positions[0][0] < positions[-1][0]:
            voxel = voxel[::-1]
            print(f"{study_id} {scan_type} {plane} reordered")
        voxel = voxel.transpose((1, 2, 0))
        voxel = np.rot90(voxel, 2, axes=(1, 2))
    elif plane == "Axial":
        if positions[0][2] > positions[-1][2]:
            voxel = voxel[::-1]
            print(f"{study_id} {scan_type} {plane} reordered")
        voxel = np.rot90(voxel, 2)
    else:
        raise ValueError(f"Unknown plane {plane}")
    return voxel, plane


def normalize_contrast(voxel):
    if voxel.sum() == 0:
        return voxel
    voxel = voxel - np.min(voxel)
    voxel = voxel / np.max(voxel)
    voxel = (voxel * 255).astype(np.uint8)
    return voxel


def crop_voxel(voxel):
    if voxel.sum() == 0:
        return voxel
    keep = (voxel.mean(axis=(0, 1)) > 0)
    voxel = voxel[:, :, keep]
    keep = (voxel.mean(axis=(0, 2)) > 0)
    voxel = voxel[:, keep]
    keep = (voxel.mean(axis=(1, 2)) > 0)
    voxel = voxel[keep]
    return voxel


def resize_voxel(voxel, size: int = 64):
    output = np.zeros((size, size, size), dtype=np.uint8)

    if np.argmax(voxel.shape) == 0:
        for i, s in enumerate(np.linspace(0, voxel.shape[0] - 1, size)):
            output[i] = cv2.resize(voxel[int(s)], (size, size))
    elif np.argmax(voxel.shape) == 1:
        for i, s in enumerate(np.linspace(0, voxel.shape[1] - 1, size)):
            output[:, i] = cv2.resize(voxel[:, int(s)], (size, size))
    elif np.argmax(voxel.shape) == 2:
        for i, s in enumerate(np.linspace(0, voxel.shape[2] - 1, size)):
            output[:, :, i] = cv2.resize(voxel[:, :, int(s)], (size, size))

    return output


class Dataset(torch_data.Dataset):
    def __init__(self,
                 data_dir,
                 paths,
                 targets=None,
                 mri_types=None,
                 image_size: int = 256,
                 transforms=None):
        self.data_dir = data_dir
        self.paths = paths
        self.targets = targets
        self.mri_types = mri_types
        self.split = 'test' if targets is None else 'train'
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        scan_id = self.paths[index]
        voxel, plane = get_voxel(data_root=self.data_dir,
                                 study_id=str(scan_id).zfill(5),
                                 scan_type=self.mri_types[index],
                                 split=self.split)
        voxel = crop_voxel(voxel)
        voxel = normalize_contrast(voxel)
        voxel = resize_voxel(voxel, size=self.image_size)

        if self.transforms is not None:
            voxel = self.transforms(image=voxel)["image"]

        if self.targets is None:
            return {"X": voxel, "id": scan_id, "plane": plane}
        else:
            return {"X": voxel, "id": scan_id, "plane": plane, "y": torch.tensor(self.targets[index], dtype=torch.float)}
