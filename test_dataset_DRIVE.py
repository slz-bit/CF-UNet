import os
import pickle
import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np

class test_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None):

        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
        self.img_file = self._select_img(self.data_file)


    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
            # img = np.array(Image.open(file))[:, :, None]
            # img = torch.from_numpy(img).float()
            # img = img.permute(2, 0, 1)

        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()
            # gt = np.array(Image.open(file))[:, :, None]
            # gt = torch.from_numpy(gt).float()
            # gt = gt.permute(2, 0, 1)

        return img, gt

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list

    def __len__(self):
        return len(self.img_file)
