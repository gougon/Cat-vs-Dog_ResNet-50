import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

from models.resnet50 import ResNet50
from models.residual_block import ResidualBlock
import docs.constant as const
import docs.config as cfg


class ResNetTrainer():
    def __init__(self, dataset, train_loader, test_loader, device):
        self.__dataset = dataset
        self.__train_loader = train_loader
        self.__test_loader = test_loader
        self.__device = device
        self.__criterion = nn.CrossEntropyLoss()
        self.__model = ResNet50(ResidualBlock, len(const.CLASS)).to(device)
        self.__optimizer = optim.SGD(self.__model.parameters(), lr=cfg.LR,
                                     momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    def show_structure(self):
        summary(self.__model, (3, 224, 224))

    def train(self):
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        for epoch in range(cfg.EPOCH):
            train_loss, train_acc = self.__train_once(epoch)
            test_loss, test_acc = self.__test_once(epoch)
            print('Epoch: {}'.format(epoch))
            print(train_loss, train_acc)
            print(test_loss, test_acc)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        self.__draw_once(train_losses, test_losses, train_accs, test_accs)
        torch.save(self.__model, const.MODEL_LOC + const.MODEL_NAME)

    def __train_once(self, epoch):
        self.__model.train()

        sum_loss = 0
        total = 0
        correct = 0
        for i, (data, label) in enumerate(self.__train_loader):
            i = i % 10
            loading = '-' * i
            print('\r' + loading, end='', flush=True)
            data, label = data.float().to(self.__device), label.to(self.__device)
            self.__optimizer.zero_grad()

            output = self.__model(data)
            loss = self.__criterion(output, label.long())
            loss.backward()
            self.__optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += data.size(0)
            correct += predicted.eq(label.data).cpu().sum()

            # if epoch % 1000 == 0:
            #     print('Loss: {}'.format(loss.item()))
            #     print('Accuracy: {}'.format(correct / float(total)))

        return sum_loss, (correct / float(total)).item()

    def __test_once(self, epoch):
        with torch.no_grad():
            self.__model.eval()

            sum_loss = 0
            total = 0
            correct = 0
            for i, (data, label) in enumerate(self.__test_loader):
                i = i % 10
                loading = '-' * i
                print('\r' + loading, end='', flush=True)
                data, label = data.float().to(self.__device), label.to(self.__device)
                output = self.__model(data)
                loss = self.__criterion(output, label)

                sum_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += data.size(0)
                correct += predicted.eq(label.data).cpu().sum()
            return sum_loss, (correct / float(total)).item()

    def __draw_once(self, train_losses, test_losses, train_accs, test_accs):
        self.__model.eval()
        writer = SummaryWriter(const.LOG_LOC)
        length = len(train_losses)
        for i in range(length):
            writer.add_scalar('Accuracy', train_accs[i], i)
            writer.add_scalar('Loss', train_losses[i], i)

    def random_test(self):
        model = torch.load(const.MODEL_LOC + const.MODEL_NAME)
        model.eval().cuda()
        img_idx = np.random.randint(len(self.__dataset))
        img, label = torch.Tensor([self.__dataset[img_idx][0]]).cuda(), torch.Tensor([self.__dataset[img_idx][1]]).cuda()
        _, predicted = torch.max(model(img).data, 1)
        img = img.cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return predicted.item(), img / 255

    def validate_model(self, model):
        loss, acc = self.__test_once(0)
        return acc * 100
