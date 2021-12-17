import torch.utils.data as data

import docs.constant as const
from utils.data_utils import DataUtils


class AnimalDataset(data.Dataset):
    def __init__(self):
        self.__datas = []
        self.__labels = []
        self.__load_datas(const.DATASET_LOC)

    def __getitem__(self, item):
        return self.__datas[item], self.__labels[item]

    def __len__(self):
        return len(self.__datas)

    def __load_datas(self, dir):
        for label, name in const.CLASS.items():
            print('Load {}'.format(name))
            cls_datas = DataUtils.load_images(dir + name)
            self.__datas.extend(cls_datas)
            self.__labels.extend([label] * len(cls_datas))
