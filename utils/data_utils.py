import os
import cv2
import numpy as np
import torch

import docs.constant as const


class DataUtils():
    @staticmethod
    def load_images(dir, format='jpg'):
        def _load_image(imgs, filename):
            if format != filename[-3:]:
                return imgs
            img = cv2.imread(filename)
            if img is None:
                return imgs
            img = cv2.resize(img, const.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
            img = np.transpose(img, (2, 0, 1))
            imgs.append(img)
            return imgs
        imgs = []
        for dirpath, dirnames, filenames in os.walk(dir):
            for i, filename in enumerate(filenames):
                if const.DEBUG and i == 100:
                    break
                print('\r({}/{})'.format(i + 1, len(filenames)), flush=True, end='')
                imgs = _load_image(imgs, dirpath + filename)
        print('\n')
        return imgs
