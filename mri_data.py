import os
import sys
import urllib.request

import pickle

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
        array = np.array(im, dtype=np.float32) / 255
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

    def exchange(self, data, oi, ni):
        tmp = data[oi]
        data[oi] = data[ni]
        data[ni] = tmp

    def process_data(self, data, shuffle=True):
        return_value = {'images': [], 'labels': [], 'categories': []}
        
        sizes = np.zeros(self.LABELS)

        for item in data:
            return_value['images'].append(item[0])
            return_value['labels'].append(item[1])
            sizes[item[1]] += 1
            return_value['categories'].append(item[2])

        for item in ['images', 'labels', 'categories']:
            return_value[item] = np.array(return_value[item])
        
        for i in range(32):
            permutation = np.random.permutation(len(return_value['images']))
            for oi, ni in enumerate(permutation):
                for data in ['images', 'labels', 'categories']:
                    self.exchange(return_value[data], oi, ni)

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
        if os.path.isdir('mri-data'):
            data = {'images': []}
            print('Reading data from pickle.')

            print('Images')
            for filename in os.listdir('mri-data'):
                if 'images' in filename:
                    with open(os.path.join('mri-data', filename), 'rb') as pfile:
                        data['images'].extend(pickle.load(pfile))
            data['images'] = np.array(data['images'], dtype=np.float32)

            print('Labels')
            with open('mri-data/labels.pickle', 'rb') as pfile:
                data['labels'] = pickle.load(pfile)
            data['labels'] = np.array(data['labels'])

            print('Categories')
            with open('mri-data/categories.pickle', 'rb') as pfile:
                data['categories'] = pickle.load(pfile)
            data['categories'] = np.array(data['categories'])

        else:
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

            print('Exporting data.')
            os.mkdir('mri-data')

            print('Images')
            N = len(data['images'])
            parts = 8
            n = N//8
            for i in range(parts):
                l = i*n
                h = (i+1)*n
                if i == parts-1:
                    h = N

                with open('mri-data/images-part{}.pickle'.format(i+1), 'wb+') as pfile:
                    pickle.dump(data['images'][l:h], pfile)

            print('Labels')
            with open('mri-data/labels.pickle', 'wb+') as pfile:
                pickle.dump(data['labels'], pfile)

            print('Categories')
            with open('mri-data/categories.pickle', 'wb+') as pfile:
                pickle.dump(data['categories'], pfile)

        print('Splitting data.')
        last = 0
        for idx, dataset in enumerate(['train', 'dev', 'test']):
            current = int((sizes[idx]/100) * len(data['labels']))
            part = self.get_part_of_data(data, last, current)
            setattr(self, dataset, self.Dataset(part))
            last += current
