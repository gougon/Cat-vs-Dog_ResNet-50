import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet50 import ResNet50
from models.residual_block import ResidualBlock
import docs.constant as const
import docs.config as cfg


class ResNetTrainer():
    def __init__(self, train_loader, test_loader, device):
        self.__train_loader = train_loader
        self.__test_loader = test_loader
        self.__device = device
        self.__criterion = nn.BCEWithLogitsLoss()
        self.__model = ResNet50(ResidualBlock, len(const.CLASS)).to(device)
        self.__optimizer = optim.SGD(self.__model.parameters(), lr=cfg.LR,
                                     momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    def train(self):
        for epoch in range(cfg.EPOCH):
            train_loss, train_acc = self.__train_once(epoch)
            test_loss, test_acc = self.__test_once(epoch)
            print(train_loss, train_acc)
            print(test_loss, test_acc)

    def __train_once(self, epoch):
        self.__model.train()

        sum_loss = 0
        total = 0
        correct = 0
        for i, (data, label) in enumerate(self.__train_loader):
            data, label = data.to(self.__device), label.to(self.__device)
            self.__optimizer.zero_grad()

            output = self.__model(data)
            loss = self.__criterion(output, label)
            loss.backward()
            self.__optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += data.size(0)
            correct += predicted.eq(label.data).cpu().sum()
        return sum_loss, correct / float(total)

    def __test_once(self, epoch):
        with torch.no_grad():
            self.__model.eval()

            sum_loss = 0
            total = 0
            correct = 0
            for i, (data, label) in enumerate(self.__test_loader):
                data, label = data.to(self.__device), label.to(self.__device)
                output = self.__model(data)
                loss = self.__criterion(output, label)

                sum_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += data.size(0)
                correct += predicted.eq(label.data).cpu().sum()
            return sum_loss, correct / float(total)

