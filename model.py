#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from reader import read_images

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

    def __init__(self,args):
        input1 = tf.keras.layers.Input(shape=[299,299,3])

        transfer_model = tfhub.KerasLayer(
                args.model,
                output_shape=[args.model_output],
                trainable=False)(input1, training=False)
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
            decay_steps=args.epochs * train_size / args.batch_size,
            end_learning_rate=args.learning_rate_final)

        elif args.decay == 'exponential':
            learning_rate = tf.optimizers.schedules.ExponentialDecay(
            args.learning_rate,
            decay_rate=args.learning_rate_final / args.learning_rate,
            decay_steps=args.epochs * train_size / args.batch_size)
        else:
            learning_rate = None

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir,
                update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

        self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

    def train(self,train_data, val_data, args):
        self.model.fit(
            x=data.batch(args.batch_size),
            validation_data=val_data.batch(args.batch_size),
            epochs=args.epochs,
            callbacks=[self.tb_callback]
        )
    def predict(self, data_images, args):
        return  self.model.predict(data_images)

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re
    #  Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
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
    sizes = (0.7,0.15,0.15)
    data, data_size  = read_images()

    train_size = int(sizes[0] * data_size)
    val_size = int(sizes[1] * data_size)
    test_size = int(sizes[2] * data_size)

    train = data.take(train_size)
    test = data.skip(train_size)
    dev = test.skip(val_size)
    test = test.take(test_size)
    # Network
    network = Network(args)
    print("Train size {}, dev size {}, test size {}".format(train_size,
        val_size, test_size) 
    network.train(train, dev, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
#    with open(os.path.join(args.logdir, "images_test.txt"), "w", encoding="utf-8") as out_file:
#        for probs in network.predict(images.test.data["images"], args):
#            print(np.argmax(probs), file=out_file)

