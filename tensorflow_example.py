"""Example to demonstrate how to integrate brainwave text notification into TensorFlow training pipeline."""
import os

import tensorflow as tf

from brainwave.backend import AmazonSES
from brainwave.notification import AccuracyAndLossTextMessage


MODEL_NAME = "toy"
NUM_CLASSES = 10
BATCH_SIZE = 256
TRAIN_BATCHES = 500
VAL_BATCHES = 10
LEN_FEATURES = 512
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3


def input_fn():
    features = tf.random_normal(shape=[BATCH_SIZE, LEN_FEATURES], dtype=tf.float32)
    labels = tf.random_uniform(shape=[BATCH_SIZE], dtype=tf.int32, maxval=NUM_CLASSES)
    return features, labels


def model_fn(features):
    out = tf.layers.dense(features, units=256)
    out = tf.layers.dense(out, units=128)
    out = tf.layers.dense(out, units=NUM_CLASSES)
    return out


def model_spec(features, labels, is_training=False, learning_rate=LEARNING_RATE, reuse=tf.AUTO_REUSE):
    spec = {}
    with tf.variable_scope(MODEL_NAME, reuse=reuse):
        logits = model_fn(features)
        predictions = tf.argmax(logits, axis=1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if is_training:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)
        spec["train_op"] = train_op
    with tf.variable_scope("metrics"):
        metrics = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions),
            "loss": tf.metrics.mean(loss)
        }
    spec["metrics"] = metrics
    spec["init_metrics"] = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    )
    spec["update_metrics"] = tf.group(*[op for _, op in metrics.values()])
    return spec


def main():
    features, labels = input_fn()
    train_spec = model_spec(features=features, labels=labels, is_training=True)
    val_spec = model_spec(features=features, labels=labels)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_val_acc = 0

        # Initialize Amazon Simple Notification Service backend for sending texts
        texter = AmazonSNS()

        for epoch in range(NUM_EPOCHS):
            # training
            sess.run(train_spec["init_metrics"])
            for _ in range(TRAIN_BATCHES):
                sess.run([train_spec["train_op"], train_spec["update_metrics"]])
            train_metrics = sess.run({k: v[0] for k, v in train_spec["metrics"].items()})

            # validation
            sess.run(val_spec["init_metrics"])
            for _ in range(VAL_BATCHES):
                sess.run(val_spec["update_metrics"])
            val_metrics = sess.run({k: v[0] for k, v in val_spec["metrics"].items()})
            print("Epoch: {}".format(epoch))
            print("Train: {}".format(train_metrics))
            print("Validation: {}".format(val_metrics))
            print()

            # only send notification when validation accuracy gets better
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                
                # send a templated text message
                text_message = AccuracyAndLossTextMessage(
                    model=MODEL_NAME,
                    epoch=epoch,
                    train_accuracy=train_metrics["accuracy"],
                    train_loss=train_metrics["loss"],
                    val_accuracy=val_metrics["accuracy"],
                    val_loss=val_metrics["loss"]
                )
                try:
                    to_number = os.environ["BRAINWAVE_PHONE_NUMBER"]
                except KeyError:
                    raise KeyError("Phone number not found. Did you source contacts.env?")
                texter.send_text(to_number=to_number, message=text_message)


if __name__ == "__main__":
    main()
