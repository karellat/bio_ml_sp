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

def read_images(sizes=(0.7,0.15,0.15),batch_size=64):
    data_root = Path("./data")
    all_image_paths = list(data_root.glob('*/*/*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.seed(42)
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_names = dict((name, index) for index,name in enumerate(label_names))
    all_image_labels = [label_names[Path(path).parent.parent.parent.name] for path in
            all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))


    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # Train test val split
    train_size = int(sizes[0] * image_count)
    val_size = int(sizes[1] * image_count)
    test_size = int(sizes[2] * image_count)

    train_dataset = image_label_ds.take(train_size)
    test_dataset = image_label_ds.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)

    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
