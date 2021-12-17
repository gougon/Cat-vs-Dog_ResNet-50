import os
import cv2


class DataUtils():
    @staticmethod
    def load_images(dir, format='jpg'):
        def _load_image(imgs, filename):
            if format != filename[-3:]:
                return imgs
            imgs.append(cv2.imread(filename))
            return imgs
        imgs = []
        for dirpath, dirnames, filenames in os.walk(dir):
            for i, filename in enumerate(filenames):
                print('\r({}/{})'.format(i + 1, len(filenames)), flush=True, end='')
                imgs = _load_image(imgs, dirpath + filename)
        print('\n')
        return imgs


