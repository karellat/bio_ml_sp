import os
import sys
import urllib.request

import numpy as np
from PIL import Image

class MRI_DATA:
    H, W, C = 299, 299, 3
    LABELS = 0
    LABEL_NAME = {}

    class Dataset:
        def __init__(self, data, seed=42):
            self._data = data
            #self._data["images"] = self._data["images"].astype(np.float32) / 255
            self._size = len(self._data["images"])

            #self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key in self._data:
                    batch[key] = self._data[key][batch_perm]
                yield batch

    def process_image(self, path, label):
        im = Image.open(path)
        array = np.array(im, dtype=np.float16) / 255
        return (array, label, self.LABEL_NAME[label] != 'NonDemented')

    def loadata(self, dirpath, label):
        data = []
        for filepath in os.listdir(dirpath):
            path = os.path.join(dirpath, filepath)
            if os.path.isdir(path):
                data.extend(self.loadata(path, label))
            else:
                data.append(self.process_image(path, label))

        return data

    def process_data(self, data, shuffle=True):
        return_value = {'images': [], 'labels': [], 'categories': []}

        for item in data:
            return_value['images'].append(item[0])
            return_value['labels'].append(item[1])
            return_value['categories'].append(item[2])

        for item in ['images', 'labels', 'categories']:
            return_value[item] = np.array(return_value[item])
        
        return return_value

    def get_part_of_data(self, data, index, length):
        return_value = {}

        for item in ['images', 'labels', 'categories']:
            return_value[item] = data[item][index:index+length]

        return return_value

    def __init__(self, sizes=(70, 20, 10)):
        """Sizes are in order: train, dev, test.
        Initializes datasets.
        """
        assert sum(sizes) == 100, "Sum of sizes must equal 100."
        data = []

        for targetpath in ['Alzheimers', 'NonAlzheimers']:
            if not os.path.isdir(targetpath):
                print('Missing directory {}.'.format(targetpath))
                continue

            print('Processing directory {}.'.format(targetpath))
            for filepath in os.listdir(targetpath):
                dirpath = os.path.join(targetpath, filepath)
                if not os.path.isdir(dirpath) or filepath.lower().startswith('notinuse'):
                    print('Skipping directory {}.'.format(dirpath))
                    continue
                
                print('Processing directory {}.'.format(dirpath))
                self.LABEL_NAME[self.LABELS] = filepath
                data.extend(self.loadata(dirpath, self.LABELS))
                self.LABELS += 1

        print('Processing data.')
        data = self.process_data(data)

        print('Splitting data.')
        last = 0
        for idx, dataset in enumerate(['train', 'dev', 'test']):
            current = int((sizes[idx]/100) * len(data['labels']))
            part = self.get_part_of_data(data, last, current)
            setattr(self, dataset, self.Dataset(part))
            last += current