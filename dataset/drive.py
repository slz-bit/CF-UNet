from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box, random_rot_flip_onlyimg, random_rotate_onlyimg

from copy import deepcopy
import pickle
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms


class DRIVEDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name #acdc
        self.data_path = root # data_root: Your/ACDC/Path
        self.mode = mode
        self.size = size
        if mode == 'test':
            self.file_root = os.path.join(self.data_path, 'labeled')
            self.ids = os.listdir(self.file_root)
            self.img_file = self._select_img(self.ids)



        if mode == 'train_l' or mode == 'train_u':
            if mode == 'train_l':
                self.file_root = os.path.join(self.data_path, 'labeled')
                self.ids = os.listdir(self.file_root)
                self.img_file = self._select_img(self.ids)
                self.gt_file = self._select_gt(self.ids)
            if mode == 'train_u':
                self.file_root = os.path.join(self.data_path, 'unlabeled')
                self.ids = os.listdir(self.file_root)
                self.img_file = self._select_img(self.ids)
            # with open(id_path, 'r') as f:
            #     self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                # print('before')
                # print(self.img_file)
                self.img_file *= math.ceil(nsample / len(self.img_file))
                self.img_file = self.img_file[:nsample]
                # print('after:')
                # print(self.img_file)
                print('self.ids:'+ str(len(self.ids)))
                print('self.img_file:' + str(len(self.img_file)))
                print('nsample:' + str(nsample))
        else:
            # with open(r'split_drive\%s\val.txt' % name, 'r') as f:
            self.file_root = os.path.join(self.data_path, 'labeled')
            with open(r"D:\datasets\STARE\txt\val.txt", 'r') as f:
                self.ids = f.read().splitlines()
                self.img_file = self._select_img(self.ids)
            with open(r"D:\datasets\STARE\txt\train.txt", "r") as f1:
            # with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
    def __getitem__(self, item):
        img_file = self.img_file[item]
        # sample = h5py.File(os.path.join(self.root, id), 'r')
        with open(file=os.path.join(self.file_root, img_file), mode='rb') as file:
            img = pickle.load(file)
            img = np.squeeze(img, 0)

        if self.mode == 'val' or self.mode == 'train_l' or self.mode == 'test':
            gt_file = "gt" + img_file[3:]
            with open(file=os.path.join(self.file_root, gt_file), mode='rb') as file:
                mask = pickle.load(file)
                mask = np.squeeze(mask, 0)

        if self.mode == 'val' or self.mode == 'test':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(mask).long()

        if self.mode == 'train_l':
            if random.random() > 0.5:
                img, mask = random_rot_flip(img, mask)
                # print('middle img= ' + str(img.shape), 'mask= ' + str(mask.shape))
            elif random.random() > 0.5:
                img, mask = random_rotate(img, mask)
            # print('after img= '+str(img.shape), 'mask= '+str(mask.shape))
            # x, y = img.shape[1:]
            # img = zoom(img, (self.size / x, self.size / y), order=0)
            # mask = zoom(mask, (self.size / x, self.size / y), order=0)
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()
        else:
            if random.random() > 0.5:
                img = random_rot_flip_onlyimg(img)
                # print('middle img= ' + str(img.shape), 'mask= ' + str(mask.shape))
            elif random.random() > 0.5:
                img = random_rotate_onlyimg(img)


        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        return img, img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):

       return len(self.img_file)


    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                if file[-4:] != 'pkl':
                    file = file + '.pkl'
                img_list.append(file)
        return img_list

    def _select_gt(self, file_list):
        gt_list = []
        for file in file_list:
            if file[:2] == "gt":
                if file[-4:] != 'pkl':
                    file = file + '.pkl'
                gt_list.append(file)

        return gt_list