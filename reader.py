import glob
from pathlib import Path
import os
import tensorflow as tf
import random

def preprocess_image(image,shape=(299,299,3)):
    image = tf.image.decode_jpeg(image,channels=shape[2])
    image = tf.image.resize(image, (shape[0], shape[1]))
    image /= 255 # normalize image

    return image

def load_and_preprocess_image(path, shape=(299,299,3)):
    image = tf.io.read_file(path)
    return preprocess_image(image, shape)

def read_images(seed=42):
    data_root = Path("./data")
    all_image_paths = list(data_root.glob('*/*/*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.seed(seed)
    random.shuffle(all_image_paths)

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_names = dict((name, index) for index,name in enumerate(label_names))
    all_image_labels = [label_names[Path(path).parent.parent.parent.name] for path in
            all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.one_hot(tf.cast(all_image_labels,
        tf.int64), 2))
    

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    image_count = len(all_image_paths)
    return image_label_ds, image_count
