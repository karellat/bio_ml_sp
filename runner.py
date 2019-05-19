import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub
from mri_data import MRI_DATA

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", tfhub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

import argparse
import datetime
import os
import re
from model import Network
#  Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, type=str,
    help="Exponentia decay")
parser.add_argument("--learning_rate", default=0.01, type=float,
    help="Initial learning rate")
parser.add_argument("--learning_rate_final", default=None, type=float,
    help="Final learning rate")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--nn", default="",type=str, help="Shared convolution layers")
parser.add_argument("--model",
        default="https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/3",
        type=str, help="Transfer learning model")
parser.add_argument("--model_output", default=2048,type=int,
    help="Output dim of transfer model")
parser.add_argument("--labels", default="categories",type=str)
parser.add_argument("--classes", default=2, type=int, help="Number of epochs.")
args = parser.parse_args()

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))

# Images
images = MRI_DATA()

# Network
network = Network(args)
network.train(images, args)

# Generate test set annotations, but in args.logdir to allow parallel execution.
with open(os.path.join(args.logdir, "images_test.txt"), "w", encoding="utf-8") as out_file:
    for probs in network.predict(images.test.data["images"], args):
        print(np.argmax(probs), file=out_file)

