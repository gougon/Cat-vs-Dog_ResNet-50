from PyQt5 import QtWidgets
from ui.mainWindow import Ui_MainWindow
import sys
import torch
import torch.utils.data as data
from torch.utils.data import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import cv2

import docs.constant as const
import docs.config as cfg
from datasets.animal_dataset import AnimalDataset
from train.resnet_trainer import ResNetTrainer
from utils.data_utils import DataUtils


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        device = torch.device(device)

        dataset = AnimalDataset()
        train_indices, test_indices, validate_indices = self.make_dataset_indices(dataset)
        train_loader = self.make_dataloader(dataset, train_indices)
        test_loader = self.make_dataloader(dataset, test_indices)
        validate_loader = self.make_dataloader(dataset, validate_indices)

        self.trainer = ResNetTrainer(dataset, train_loader, test_loader, device)

        self.ui.showStructureButton.clicked.connect(lambda: self.click_show_structure_button())
        self.ui.showBoardButton.clicked.connect(lambda: self.click_show_board_button())
        self.ui.testButton.clicked.connect(lambda: self.click_test_button())
        self.ui.erasingButton.clicked.connect(lambda: self.click_erasing_button())

    def make_dataset_indices(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split1 = int(np.floor(cfg.TEST_SIZE * dataset_size))
        split2 = split1 + int(np.floor(cfg.VALIDATE_SIZE * dataset_size))
        np.random.shuffle(indices)
        return indices[split2:], indices[:split1], indices[split1:split2]

    def make_dataloader(self, dataset, indices):
        sampler = SubsetRandomSampler(indices)
        return data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler)

    def click_show_structure_button(self):
        self.trainer.show_structure()

    def click_show_board_button(self):
        img = cv2.imread(const.LOG_LOC + 'v1_img.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    def click_test_button(self):
        predicted, img = self.trainer.random_test()
        plt.title(const.CLASS[predicted])
        plt.imshow(img)
        plt.show()

    def click_erasing_button(self):
        m1 = torch.load(const.MODEL_LOC + 'm1.pt')
        m2 = torch.load(const.MODEL_LOC + 'm2.pt')
        acc1 = self.trainer.validate_model(m1)
        acc2 = self.trainer.validate_model(m2)
        plt.bar(['Before Random-Erasing', 'After Random-Erasing'], [acc1, acc2])
        plt.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
