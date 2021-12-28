import torch.utils.data as data
import numpy as np
import math
import random

import docs.constant as const
import docs.config as cfg
from utils.data_utils import DataUtils


class AnimalDataset(data.Dataset):
    def __init__(self):
        self.__datas = []
        self.__labels = []
        self.__load_datas(const.DATASET_LOC)
        if cfg.RANDOM_ERASING:
            self.__random_erasing()

    def __getitem__(self, item):
        return self.__datas[item], self.__labels[item]

    def __len__(self):
        return len(self.__datas)

    def __load_datas(self, dir):
        for label, name in const.CLASS.items():
            print('Load {}'.format(name))
            cls_datas = DataUtils.load_images(dir + name + '/')
            self.__datas.extend(cls_datas)
            self.__labels.extend([label] * len(cls_datas))

    def __random_erasing(self):
        p = cfg.P
        area_ratio = cfg.AREA_RATIO
        aspect_ratio = cfg.ASPECT_RATIO
        length = len(self.__datas)
        for i in range(length):
            print('\r{}'.format(i), end='', flush=True)
            data = self.__datas[i]
            label = self.__labels[i]
            is_erase, img = self.__erasing_image(data, p, area_ratio, aspect_ratio)
            if not is_erase:
                continue
            self.__datas.append(img)
            self.__labels.append(label)

    def __erasing_image(self, img, p, area_ratio, aspect_ratio):
        p1 = np.random.rand()
        if p1 >= p:
            return False, img

        w = img.shape[1]
        h = img.shape[2]
        s = w * h
        while True:
            se = (((area_ratio[1] - area_ratio[0]) * np.random.rand()) + area_ratio[0]) * s
            re = (aspect_ratio[1] - aspect_ratio[0]) * np.random.rand()
            he = int(np.floor(math.sqrt(se * re)))
            we = int(np.floor(math.sqrt(se / re)))
            if he <= 0 or we <= 0:
                continue
            print('he:{}'.format(he))
            print('we:{}'.format(we))
            xe = np.random.randint(we)
            ye = np.random.randint(he)
            if xe + we <= w and ye + he <= h:
                new_img = img.copy()
                ie = np.random.randint(0, 255, (3, we, he))
                new_img[:, xe:xe+we, ye:ye+he] = ie
                return True, new_img
