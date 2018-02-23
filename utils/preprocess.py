# Basic
import os
import sys
import random
import time
import numpy as np
import pandas as pd

# Image reader
from skimage.io import imread, imshow, imread_collection, concatenate_images

# Using multiprocessing
from multiprocessing import Pool

# h5 io
from h5io import write_to_h5, load_from_h5



# for multiprocessing usage
def _crop(img, crop_size, i, j):
    return img[i: i+crop_size, j: j+crop_size]
def _crop_wrapper(args):
    return _crop(*args)


class ImagePreproc:
    _size = None
    _channel = None
    _imgs = None
    _ids = None
    _masks = None
    _x = None
    _y = None
    
    def __init__(self, path='../data/train', size=128, channel=3, normalize=False, augment=False):
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
        
        print("Extracting image info ...")
        start_time = time.time()
        for img_id in os.listdir(path):
            self._ids.append(img_id)
            img_file_list = os.listdir('{0}/{1}/images'.format(path, img_id))
            assert len(img_file_list) == 1, "Multiple images found in one images id folder."
            assert img_file_list[0] == img_id + '.png', "Image id and image name do not match."
            img = imread('{0}/{1}/images/{1}.png'.format(path, img_id, img_id))[:, :, :channel]
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
            if augment:
                self._ids.append(img_id)
                self._imgs.append(np.flipud(img))
                self._masks.append(np.flipud(mask))
                for rot in range(3):
                    img = np.rot90(img)
                    mask = np.rot90(mask)
                    self._ids.append(img_id)
                    self._ids.append(img_id)
                    self._imgs.append(img)
                    self._masks.append(mask)
                    self._imgs.append(np.flipud(img))
                    self._masks.append(np.flipud(mask))
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        print len(self._ids), len(self._imgs), len(self._masks)

    def get_cropped_set(self, stride=16, num_threads=4):
        print("Getting cropped images ...")
        start_time = time.time()
        x = []
        y = []
        file_idx = 0
        for img, mask in zip(self._imgs, self._masks):
            if img.shape[0] < self._size or img.shape[1] < self._size:
                continue
            i_limt = img.shape[0] - self._size + 1
            j_limt = img.shape[1] - self._size + 1
            img_args = [(img, self._size, i, j) for i in range(0, i_limt, stride) for j in range(0, j_limt, stride)]
            mask_args = [(mask, self._size, i, j) for _,_,i,j in img_args]
            pool = Pool(num_threads)
            tmp_x = list(pool.map(_crop_wrapper, img_args))
            tmp_y = list(pool.map(_crop_wrapper, mask_args))
            pool.close()
            pool.join()
            '''
            for im, ma in zip(tmp_x, tmp_y):
                filename = "data/" + str(file_idx) + ".h5"
                write_to_h5(filename, pd.DataFrame({
                    'image': im.reshape(self._size * self._size * self._channel),
                    'mask': np.pad(ma.reshape(self._size * self._size), (0, self._size * self._size * (self._channel-1)), 'constant')
                }))
                file_idx += 1
            '''
            x += tmp_x
            y += tmp_y
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        return np.array(x), np.array(y)
        
if __name__ == '__main__':
    ip = ImagePreproc(size=128, channel=3, normalize=True, augment=True)
    x, y = ip.get_cropped_set(stride=64, num_threads=16)
    print x.shape
    print y.shape
        
        
        
    