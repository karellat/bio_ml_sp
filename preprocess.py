import os
import cv2

for oripath in ['NonAlzheimers']: #, 'data_01', 'data_02', 'data_03', 'data_04']:
    for dirpath2 in os.listdir(oripath):
        for dirpath in os.listdir(os.path.join(oripath,dirpath2)):
            path = os.path.join(oripath,dirpath2, dirpath)
            if not os.path.isdir(path):
                print('Skipping dir {}'.format(path))
                continue

            print('Processing dir {}'.format(path))
            for filepath in os.listdir(path):
                fpath = os.path.join(path, filepath)
                image = cv2.imread(fpath)
                resized = cv2.resize(image, (299, 299), cv2.INTER_CUBIC)
                cv2.imwrite(fpath, resized)
