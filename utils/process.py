# Basic
import os
import sys
import random
import time
import numpy as np
import pandas as pd

# Image reader
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.morphology import label

# Using multiprocessing
from multiprocessing import Pool

class TestModel:
    def predict(self, A):
        return A[:, :, 0]

def _run_length_encoding(x, threshold):
    dots = np.where(x.T.flatten() >= threshold)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join([str(y) for y in run_lengths])

class ImagePrec:
    _size = None
    _channel = None
    ## Training set info
    _imgs = None
    _ids = None
    _masks = None
    ## Testing set info
    _test_imgs = None
    _test_ids = None
    _test_masks = None

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
        
        print("Extracting training image info ...")
        start_time = time.time()
        for img_id in os.listdir(path):
            self._ids.append(img_id)
            img_file_list = os.listdir('{0}/{1}/images'.format(path, img_id))
            assert len(img_file_list) == 1, "Multiple images found in one images id folder."
            assert img_file_list[0] == img_id + '.png', "Image id and image name do not match."
            img = imread('{0}/{1}/images/{1}.png'.format(path, img_id, img_id))[:, :, :channel]
            if normalize:
                ## Do normalization
                img = (img.astype(np.float32) - img.mean())/max(1., img.std())
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

    def get_batch_data(self, expand=1, seed=0):
        print("Getting cropped images ...")
        np.random.seed(seed);
        start_time = time.time()
        x = []
        y = []
        for img, mask in zip(self._imgs, self._masks):
            assert img.shape[0]>=self._size and img.shape[1]>=self._size, "There exist images whose size is smaller than cropped size"
            for k in range(expand):
                i = np.random.randint(img.shape[0] - self._size + 1)
                j = np.random.randint(img.shape[1] - self._size + 1)
                x.append(img[i: i+self._size, j: j+self._size])
                y.append(mask[i: i+self._size, j: j+self._size])
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        return {
            "x": np.array(x),
            "y": np.array(y)
        }

    def get_test_set(self, path='../data/test', normalize=False):
        self._test_ids = []
        self._test_imgs = []
        
        print("Extracting testing image info ...")
        start_time = time.time()
        for img_id in os.listdir(path):
            self._test_ids.append(img_id)
            img_file_list = os.listdir('{0}/{1}/images'.format(path, img_id))
            assert len(img_file_list) == 1, "Multiple images found in one images id folder."
            assert img_file_list[0] == img_id + '.png', "Image id and image name do not match."
            img = imread('{0}/{1}/images/{1}.png'.format(path, img_id, img_id))[:, :, :self._channel]
            if normalize:
                ## Do normalization
                img = (img.astype(np.float32) - img.mean())/max(1., img.std())
            self._test_imgs.append(img)
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        print len(self._test_ids), len(self._test_imgs)

    def predict(self, model, stride=16):
        """
        Generate the probability map
        """
        print "Getting predictions ..."
        start_time = time.time()
        self._test_masks = []
        for img in self._test_imgs:
            coordinates = [
                (min(i, img.shape[0]-self._size), min(j, img.shape[1]-self._size))
                for i in range(0, img.shape[0], stride)
                for j in range(0, img.shape[1], stride)
            ]
            test_batch = model.predict(np.array([img[:, :, 0][i: i+self._size, j: j+self._size] for i, j in coordinates]))
            mask = np.zeros(img.shape[:2])
            ratio = np.zeros(img.shape[:2])
            for (i, j), _mask in zip(coordinates, test_batch):
                mask[i: i+self._size, j: j+self._size] += _mask
                ratio[i: i+self._size, j: j+self._size] += np.ones((self._size, self._size))
            self._test_masks.append(mask.astype(np.float32)/ratio.astype(np.float32))
        print("Time Usage: {0} sec".format(str(time.time() - start_time)))
        return self._test_masks

    def encoding(self, threshold=0.5):
        print "Generating submission ..."
        assert len(self._test_ids) == len(self._test_masks), "Something is wrong!"
        return pd.DataFrame({
            "ImageId": self._test_ids,
            "EncodedPixels": [_run_length_encoding(x, threshold) for x in self._test_masks]
        })[["ImageId", "EncodedPixels"]]

if __name__ == '__main__':
    ip = ImagePrec(size=128, channel=3, normalize=True, augment=True)
    train = ip.get_batch_data(expand=16, seed=17)
    print train['x'].shape, train['y'].shape
    model = TestModel()
    ip.get_test_set(normalize=True)
    masks = ip.predict(model, stride=16)
    sub = ip.encoding(threshold=0.)
    sub.to_csv('test.csv', index=False)