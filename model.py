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


        self.model = tf.keras.Model(inputs=[input1], outputs=[output1])

        print(self.model.summary())

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

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir,
                update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

        self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

    def train(self, images, args):

        self.model.fit(
                x = images.train.data["images"],
                y = images.train.data[args.labels],
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=(images.dev.data["images"],
                    images.dev.data[args.labels]),
                callbacks=[self.tb_callback],
                )

    def predict(self, data_images, args):
        self.model.predict(data_images)

