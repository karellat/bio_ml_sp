#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub
from mri_data import MRI_DATA

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", tfhub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

class Network:

    @staticmethod
    def get_layer(arg, inputs):
      C_args = arg.split('-')
      if arg.startswith('C-'):
          return tf.keras.layers.Conv2D(
                  int(C_args[1]),
                  int(C_args[2]),
                  int(C_args[3]),
                  padding=C_args[4],
                  activation="relu")(inputs)
      elif arg.startswith('CB-'):
          new_layer = tf.keras.layers.Conv2D(
                  int(C_args[1]),
                  int(C_args[2]),
                  int(C_args[3]),
                  padding=C_args[4],
                  use_bias=False)(inputs)
          new_layer = tf.keras.layers.BatchNormalization()(new_layer)
          return tf.keras.layers.Activation("relu")(new_layer)
      elif arg.startswith('M-'):
         return tf.keras.layers.MaxPool2D(
             int(C_args[1]),
             int(C_args[2]))(inputs)
      elif arg.startswith('R-'):
          assert len(arg[3:-1].split(';')) != 0
          new_layer = inputs
          print(arg[3:-1])
          for a in arg[3:-1].split(';'):
              new_layer = get_layer(a, new_layer)
          return tf.keras.layers.Add()([new_layer, inputs])
      elif arg.startswith('D-'):
          return tf.keras.layers.Dense(
             int(C_args[1]),
              activation="relu")(inputs)
      elif arg.startswith('F'):
          return tf.keras.layers.Flatten()(inputs)
      elif arg.startswith('Dr'):
          return tf.keras.layers.Dropout(rate=0.5)(inputs)
      else:
        raise Exception('Unknown cnn argument {}'.format(arg))

    def __init__(self, args):
        input1 = tf.keras.layers.Input(shape=[299,299,3])

        transfer_model = tfhub.KerasLayer(
                "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/3",
                output_shape=[2048],
                trainable=False)(input1,training=False)
        hidden = transfer_model

        for l in filter(None, args.nn.split(",")):
            hidden = self.get_layer(l, hidden)

        flatten = tf.keras.layers.Flatten()(hidden)
        output1 = tf.keras.layers.Dense(args.classes, activation='softmax')(flatten)

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir,
                update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

        self.model = tf.keras.Model(inputs=[input1], outputs=[output1])

        # Learning rate
        if args.decay is None:
            learning_rate = args.learning_rate
        elif args.decay == 'polynomial':
            learning_rate = tf.optimizers.schedules.PolynomialDecay(
            args.learning_rate,
            decay_steps=args.epochs * images.train.size / args.batch_size,
            end_learning_rate=args.learning_rate_final)

        elif args.decay == 'exponential':
            learning_rate = tf.optimizers.schedules.ExponentialDecay(
            args.learning_rate,
            decay_rate=args.learning_rate_final / args.learning_rate,
            decay_steps=args.epochs * images.train.size / args.batch_size)
        else:
            learning_rate = None

        self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.metrics.CategoricalAccuracy()])

    def train(self, images, args):
        self.model.fit(
                x = images.train.data["images"],
                y = images.train.data[args.labels],
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=(images.dev.data["images"],
                    images.dev.data[args.labels])
                )

    def predict(self, data_images, args):
        self.model.predict(data_images)

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

     # Parse arguments
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
    network.train(caltech42, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "images_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(caltech42.test.data["images"], args):
            print(np.argmax(probs), file=out_file)

