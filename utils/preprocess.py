# Basic
import os
import sys
import random

import numpy as np
import pandas as pd

# Image reader
from skimage.io import imread, imshow, imread_collection, concatenate_images

# Using multiprocessing
from multiprocessing import Pool

class ImagePreproc:
    _size = None
    _channel = None
    _imgs = None
    _ids = None
    _masks = None
    _x = None
    _y = None
    
    def __init__(self, path='../data/train', size=128, channel=3, normalize=False):
        """
        It looks that the minimum size of images is (256, 256)
        The channel of the input images are all 4, but only the first 3 matters.
        For the 4th channel, all entries equal 255, which dose not make any differences.
        """
        self._size = size
        self._channel = channel
        self._imgs = []
        self._ids = []
        self._masks = []
        
        print "Extracting image info ..."
        for img_id in os.listdir(path):
            print img_id
            self._ids.append(img_id)
            img_file_list = os.listdir('{0}/{1}/images'.format(path, img_id))
            assert len(img_file_list) == 1, "Multiple images found in one images id folder."
            assert img_file_list[0] == img_id + '.png', "Image id and image name do not match."
            img = imread('{0}/{1}/images/{1}.png'.format(path, img_id, img_id))[:, :, :3]
            if normalize:
                ## Do normalization
                img = (img.astype(np.float32) - (img.max()/2.))/img.max()
            mask = np.zeros(img.shape[:2])
            for m in os.listdir('{0}/{1}/masks'.format(path, img_id)):
                _mask = imread('{0}/{1}/masks/{2}'.format(path, img_id, m))
                assert _mask.shape == img.shape[:2], "Image shape and mask shape do not match."
                mask = np.maximum(mask, _mask)
                mask = mask.astype(np.float32)/mask.max()
            self._imgs.append(img)
            self._masks.append(mask)
    
    # for multiprocessing usage
    def _crop_img(img, i, j, s):
        return img[i: i+s, j: j+s]
    def _crop_img_wrap(args):
        return _crop_img(*args)
    def crop(stride=4, num_threads=4):
        x = []
        y = []
        for img, mask in zip(self._imgs, self._masks):
            i_limt = img.shape[0] - self._size + 1
            j_limt = img.shape[1] - self._size + 1
            n_crop = int(i_limt/stride + 1)*int(j_limt/stride + 1)
            img_args = [[img, np.random.randint(i_limt), np.random.randint(j_limt), self._size] for i in range(n_crop)]
            mask_args = [[mask, i, j, s] for _ ,i ,j ,s in img_args]
            pool = Pool(num_threads)
            x += list(pool.map(_crop_img_wrap, img_args))
            y += list(pool.map(_crop_img_wrap, mask_args))
            pool.close()
            pool.join()
        return x, y
        
if __name__ == '__main__':
    ip = ImagePreproc(normalize=True)
        
        
        
    